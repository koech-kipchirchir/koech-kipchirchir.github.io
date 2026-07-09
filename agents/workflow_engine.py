"""
AIOS Workflow Engine
====================

Production-grade DAG-based workflow engine for defining and executing
directed task graphs with sequential/parallel execution, conditional
branches, retries, rollback hooks, checkpoints, resume, and metrics.

Usage::

    engine = WorkflowEngine()

    spec = WorkflowSpec(
        id="my-workflow",
        steps=[
            WorkflowStep(id="fetch", agent="research", task="Fetch data",
                         next_on_success="process"),
            WorkflowStep(id="process", agent="coding", task="Process data",
                         next_on_success=["validate", "archive"],
                         rollback=RollbackHook(step_id="process-undo", task="Undo process")),
            WorkflowStep(id="validate", agent="coding", task="Validate results"),
            WorkflowStep(id="archive", agent="research", task="Archive output"),
        ],
    )

    result = await engine.run(spec)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

logger = logging.getLogger("aios.workflow")

# ---------------------------------------------------------------------------
# Lazy logger
# ---------------------------------------------------------------------------

_WF_LOGGER: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _WF_LOGGER
    if _WF_LOGGER is None:
        _WF_LOGGER = logging.getLogger("aios.workflow")
    return _WF_LOGGER


def _structured_log(level: int, event: str, **kwargs: Any) -> None:
    record = {"event": event, "ts": time.time()}
    record.update(kwargs)
    _get_logger().log(level, "%s", json.dumps(record, default=str))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"


class ConditionOperator(Enum):
    EQUALS = "=="
    NOT_EQUALS = "!="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    GREATER_THAN = ">"
    LESS_THAN = "<"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"
    IS_NONE = "is_none"
    NOT_NONE = "not_none"


class BackoffStrategy(Enum):
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RetryPolicy:
    """Retry policy for a step."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff: BackoffStrategy = BackoffStrategy.EXPONENTIAL_JITTER
    retryable_on: list[str] = field(default_factory=lambda: ["TimeoutError", "ConnectionError"])
    jitter: float = 0.1


@dataclass
class RollbackHook:
    """A rollback step executed when a workflow or step fails."""

    step_id: str
    task: str = ""
    agent: str = "coding"
    context: dict[str, Any] = field(default_factory=dict)
    timeout: float = 60.0
    ignore_errors: bool = True


@dataclass
class Condition:
    """Conditional branch evaluation."""

    source_step_id: str = ""
    variable: str = ""                    # context key to evaluate
    operator: ConditionOperator = ConditionOperator.IS_TRUE
    value: Any = True
    then_step_ids: list[str] = field(default_factory=list)
    else_step_ids: list[str] = field(default_factory=list)

    def evaluate(self, context_value: Any) -> bool:
        if self.operator == ConditionOperator.IS_TRUE:
            return bool(context_value)
        if self.operator == ConditionOperator.IS_FALSE:
            return not bool(context_value)
        if self.operator == ConditionOperator.IS_NONE:
            return context_value is None
        if self.operator == ConditionOperator.NOT_NONE:
            return context_value is not None
        if self.operator == ConditionOperator.EQUALS:
            return context_value == self.value
        if self.operator == ConditionOperator.NOT_EQUALS:
            return context_value != self.value
        if self.operator == ConditionOperator.CONTAINS:
            return self.value in context_value if context_value is not None else False
        if self.operator == ConditionOperator.NOT_CONTAINS:
            return self.value not in context_value if context_value is not None else True
        if self.operator == ConditionOperator.GREATER_THAN:
            return (context_value is not None) and (context_value > self.value)
        if self.operator == ConditionOperator.LESS_THAN:
            return (context_value is not None) and (context_value < self.value)
        return False


@dataclass
class StepResult:
    """Result of a single workflow step."""

    step_id: str
    status: StepStatus = StepStatus.PENDING
    output: str = ""
    error: str = ""
    duration_ms: float = 0.0
    retry_count: int = 0
    agent_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""

    @property
    def success(self) -> bool:
        return self.status == StepStatus.COMPLETED


@dataclass
class WorkflowResult:
    """Overall result of a workflow execution."""

    workflow_id: str = ""
    status: WorkflowStatus = WorkflowStatus.PENDING
    step_results: dict[str, StepResult] = field(default_factory=dict)
    error: str = ""
    duration_ms: float = 0.0
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    skipped_steps: int = 0
    context: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    completed_at: str = ""

    @property
    def progress(self) -> float:
        if self.total_steps == 0:
            return 1.0
        done = self.completed_steps + self.failed_steps + self.skipped_steps
        return done / self.total_steps


@dataclass
class WorkflowStep:
    """A single step in a workflow DAG."""

    id: str
    agent: str = "coding"
    task: str = ""
    depends_on: list[str] = field(default_factory=list)
    next_on_success: list[str] | str | None = None   # auto-resolved if None
    next_on_failure: list[str] | str | None = None
    timeout: float = 300.0
    retry: RetryPolicy | None = None
    rollback: RollbackHook | None = None
    condition: Condition | None = None
    context: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    max_execution: int = 0                           # 0 = unlimited


@dataclass
class WorkflowSpec:
    """Complete workflow definition."""

    id: str = ""
    name: str = ""
    description: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)
    conditions: list[Condition] = field(default_factory=list)
    start_step_id: str = ""
    timeout: float = 3600.0
    default_retry: RetryPolicy | None = None
    max_parallel: int = 4
    context: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _generate_id()
        if not self.name:
            self.name = self.id
        # Auto-resolve start_step_id (first step with no dependencies)
        if not self.start_step_id and self.steps:
            for s in self.steps:
                if not s.depends_on:
                    self.start_step_id = s.id
                    break
            if not self.start_step_id and self.steps:
                self.start_step_id = self.steps[0].id

    def get_step(self, step_id: str) -> WorkflowStep | None:
        return next((s for s in self.steps if s.id == step_id), None)

    @property
    def step_ids(self) -> list[str]:
        return [s.id for s in self.steps]

    @property
    def adjacency(self) -> dict[str, list[str]]:
        """Build adjacency list from depends_on."""
        adj: dict[str, list[str]] = {s.id: [] for s in self.steps}
        for s in self.steps:
            for dep in s.depends_on:
                if dep in adj:
                    adj[dep].append(s.id)
        return adj


@dataclass
class WorkflowMetrics:
    """Performance metrics for a workflow run."""

    workflow_id: str = ""
    total_duration_ms: float = 0.0
    step_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    retry_count: int = 0
    parallel_batches: int = 0
    avg_step_duration_ms: float = 0.0
    max_step_duration_ms: float = 0.0
    min_step_duration_ms: float = 0.0
    checkpoint_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Checkpoint:
    """Serializable workflow state for resume."""

    workflow_id: str = ""
    spec: dict[str, Any] = field(default_factory=dict)
    step_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    completed_step_ids: list[str] = field(default_factory=list)
    failed_step_ids: list[str] = field(default_factory=list)
    started_step_ids: list[str] = field(default_factory=list)
    status: str = "running"
    created_at: str = ""
    version: int = 1

    def to_json(self) -> str:
        return json.dumps({
            "workflow_id": self.workflow_id,
            "spec": _serialize_spec(self.spec),
            "step_results": _serialize_step_results(self.step_results),
            "context": _serialize_context(self.context),
            "completed_step_ids": self.completed_step_ids,
            "failed_step_ids": self.failed_step_ids,
            "started_step_ids": self.started_step_ids,
            "status": self.status,
            "created_at": self.created_at,
            "version": self.version,
        }, default=str, indent=2)

    @classmethod
    def from_json(cls, data: str | dict[str, Any]) -> Checkpoint:
        if isinstance(data, str):
            data = json.loads(data)
        return cls(**data)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _current_ms() -> float:
    return time.perf_counter() * 1000


def _serialize_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return spec


def _serialize_step_results(results: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return results


def _serialize_context(ctx: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for k, v in ctx.items():
        try:
            json.dumps(v)
            cleaned[k] = v
        except (TypeError, ValueError):
            cleaned[k] = str(v)
    return cleaned


def _compute_backoff(retry_count: int, policy: RetryPolicy) -> float:
    delay = policy.base_delay
    if policy.backoff == BackoffStrategy.FIXED:
        delay = policy.base_delay
    elif policy.backoff == BackoffStrategy.LINEAR:
        delay = policy.base_delay * (retry_count + 1)
    elif policy.backoff == BackoffStrategy.EXPONENTIAL:
        delay = policy.base_delay * (2 ** retry_count)
    elif policy.backoff == BackoffStrategy.EXPONENTIAL_JITTER:
        delay = policy.base_delay * (2 ** retry_count)
        import random
        delay += random.uniform(-policy.jitter, policy.jitter)
        delay = max(0, delay)
    return min(delay, policy.max_delay)


# ---------------------------------------------------------------------------
# Step executor (injectable)
# ---------------------------------------------------------------------------


class StepExecutor:
    """Executes a single workflow step by invoking an agent.

    Override `execute_step` to integrate with your agent runtime.
    """

    async def execute_step(self, step: WorkflowStep, context: dict[str, Any],
                           step_result: StepResult) -> StepResult:
        """Execute a step. Override this to call your agent system."""
        _structured_log(logging.INFO, "step.executing",
                        step_id=step.id, agent=step.agent, task=step.task[:80])
        start = _current_ms()
        step_result.started_at = _now()
        step_result.status = StepStatus.RUNNING

        try:
            resolved_task = _interpolate(step.task, context)
            await asyncio.sleep(0.05)  # simulate agent call
            output = f"[{step.agent}] Executed: {resolved_task}"
            step_result.output = output
            step_result.status = StepStatus.COMPLETED
            context[step.id] = {"output": output, "success": True}
        except Exception as exc:
            step_result.error = str(exc)
            step_result.status = StepStatus.FAILED
            context[step.id] = {"output": "", "success": False, "error": str(exc)}

        step_result.completed_at = _now()
        step_result.duration_ms = _current_ms() - start
        return step_result


# ---------------------------------------------------------------------------
# WorkflowEngine
# ---------------------------------------------------------------------------


class WorkflowEngine:
    """Production-grade workflow engine supporting DAGs, conditionals,
    retries, rollbacks, checkpoints, and resume.

    Usage::

        engine = WorkflowEngine()
        spec = WorkflowSpec(steps=[...])
        result = await engine.run(spec)
    """

    def __init__(
        self,
        executor: StepExecutor | None = None,
        checkpoint_dir: str | None = None,
        max_parallel: int = 4,
    ) -> None:
        self._executor = executor or StepExecutor()
        self._checkpoint_dir = checkpoint_dir
        self._max_parallel = max_parallel
        self._running_workflows: dict[str, asyncio.Task[Any]] = {}
        self._results: dict[str, WorkflowResult] = {}
        self._lock = asyncio.Lock()
        self._progress_callbacks: list[Callable[[str, float, StepResult | None], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, spec: WorkflowSpec, context: dict[str, Any] | None = None) -> WorkflowResult:
        """Execute a workflow spec and return the result."""
        wf_id = spec.id or _generate_id()
        wf_result = WorkflowResult(
            workflow_id=wf_id,
            status=WorkflowStatus.RUNNING,
            total_steps=len(spec.steps),
            started_at=_now(),
            context=dict(spec.context),
        )
        if context:
            wf_result.context.update(context)

        task = asyncio.create_task(
            self._execute_workflow(spec, wf_result),
            name=f"wf-{wf_id}",
        )
        async with self._lock:
            self._running_workflows[wf_id] = task
            self._results[wf_id] = wf_result

        try:
            return await task
        except asyncio.CancelledError:
            wf_result.status = WorkflowStatus.CANCELLED
            return wf_result
        finally:
            async with self._lock:
                self._running_workflows.pop(wf_id, None)

    async def run_sequential(self, steps: list[WorkflowStep], context: dict[str, Any] | None = None) -> WorkflowResult:
        """Run a list of steps sequentially as a single workflow."""
        spec = WorkflowSpec(
            id=_generate_id(),
            name="sequential",
            steps=steps,
            max_parallel=1,
            context=context or {},
        )
        for i, s in enumerate(steps):
            if i > 0:
                s.depends_on = [steps[i - 1].id]
        return await self.run(spec)

    async def run_parallel(self, steps: list[WorkflowStep], context: dict[str, Any] | None = None) -> WorkflowResult:
        """Run a list of steps in parallel as a single workflow."""
        spec = WorkflowSpec(
            id=_generate_id(),
            name="parallel",
            steps=steps,
            max_parallel=len(steps),
            context=context or {},
        )
        return await self.run(spec)

    async def resume(self, checkpoint_id: str) -> WorkflowResult | None:
        """Resume a workflow from a saved checkpoint."""
        checkpoint = await self._load_checkpoint(checkpoint_id)
        if not checkpoint:
            _structured_log(logging.ERROR, "checkpoint.load.failed", checkpoint_id=checkpoint_id)
            return None

        spec_dict = checkpoint.spec
        spec = WorkflowSpec(**spec_dict)

        wf_result = WorkflowResult(
            workflow_id=checkpoint.workflow_id,
            status=WorkflowStatus.RUNNING,
            total_steps=len(spec.steps),
            started_at=checkpoint.created_at,
            context=checkpoint.context,
        )

        # Restore step results
        for step_id, sr_dict in checkpoint.step_results.items():
            wf_result.step_results[step_id] = StepResult(**sr_dict)

        _structured_log(logging.INFO, "workflow.resuming",
                        workflow_id=checkpoint.workflow_id,
                        completed=len(checkpoint.completed_step_ids))

        task = asyncio.create_task(
            self._execute_workflow(spec, wf_result, checkpoint=checkpoint),
            name=f"wf-resume-{checkpoint.workflow_id}",
        )
        async with self._lock:
            self._running_workflows[checkpoint.workflow_id] = task
            self._results[checkpoint.workflow_id] = wf_result

        try:
            return await task
        except asyncio.CancelledError:
            wf_result.status = WorkflowStatus.CANCELLED
            return wf_result
        finally:
            async with self._lock:
                self._running_workflows.pop(checkpoint.workflow_id, None)

    def get_result(self, workflow_id: str) -> WorkflowResult | None:
        return self._results.get(workflow_id)

    async def cancel(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        async with self._lock:
            task = self._running_workflows.get(workflow_id)
            if task and not task.done():
                task.cancel()
                if workflow_id in self._results:
                    self._results[workflow_id].status = WorkflowStatus.CANCELLED
                return True
            return False

    def on_progress(self, callback: Callable[[str, float, StepResult | None], None]) -> None:
        """Register a progress callback: (workflow_id, progress, last_step_result)."""
        self._progress_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def get_progress(self, workflow_id: str) -> float:
        result = self._results.get(workflow_id)
        return result.progress if result else 0.0

    def get_step_status(self, workflow_id: str, step_id: str) -> StepStatus | None:
        result = self._results.get(workflow_id)
        if result and step_id in result.step_results:
            return result.step_results[step_id].status
        return None

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self, workflow_id: str) -> WorkflowMetrics | None:
        result = self._results.get(workflow_id)
        if not result:
            return None
        durations = [
            sr.duration_ms for sr in result.step_results.values()
            if sr.duration_ms > 0
        ]
        return WorkflowMetrics(
            workflow_id=workflow_id,
            total_duration_ms=result.duration_ms,
            step_count=result.total_steps,
            completed_count=result.completed_steps,
            failed_count=result.failed_steps,
            skipped_count=result.skipped_steps,
            avg_step_duration_ms=sum(durations) / len(durations) if durations else 0.0,
            max_step_duration_ms=max(durations) if durations else 0.0,
            min_step_duration_ms=min(durations) if durations else 0.0,
        )

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    async def save_checkpoint(self, wf_id: str) -> str | None:
        """Save a checkpoint for a running workflow. Returns checkpoint path."""
        if not self._checkpoint_dir:
            return None
        result = self._results.get(wf_id)
        if not result:
            return None

        checkpoint = Checkpoint(
            workflow_id=wf_id,
            spec={"id": wf_id, "name": result.metadata.get("name", "")},
            step_results={
                sid: {
                    "step_id": sr.step_id, "status": sr.status.value,
                    "output": sr.output, "error": sr.error,
                    "duration_ms": sr.duration_ms, "retry_count": sr.retry_count,
                    "agent_name": sr.agent_name,
                }
                for sid, sr in result.step_results.items()
            },
            context=_serialize_context(result.context),
            completed_step_ids=[
                sid for sid, sr in result.step_results.items()
                if sr.status == StepStatus.COMPLETED
            ],
            failed_step_ids=[
                sid for sid, sr in result.step_results.items()
                if sr.status in (StepStatus.FAILED, StepStatus.TIMEOUT)
            ],
            status=result.status.value,
            created_at=_now(),
        )

        return await self._write_checkpoint(checkpoint)

    def list_checkpoints(self) -> list[str]:
        if not self._checkpoint_dir:
            return []
        p = Path(self._checkpoint_dir)
        if not p.exists():
            return []
        return sorted([str(f) for f in p.glob("*.json")], reverse=True)

    async def delete_checkpoint(self, path: str) -> bool:
        try:
            Path(path).unlink(missing_ok=True)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _execute_workflow(
        self,
        spec: WorkflowSpec,
        wf_result: WorkflowResult,
        checkpoint: Checkpoint | None = None,
    ) -> WorkflowResult:
        log = _get_logger()
        _structured_log(logging.INFO, "workflow.started",
                        workflow_id=wf_result.workflow_id, name=spec.name,
                        steps=len(spec.steps), max_parallel=spec.max_parallel)

        deadline = time.monotonic() + spec.timeout
        wf_start = _current_ms()
        completed: set[str] = set()
        failed: set[str] = set()
        skipped: set[str] = set()

        # Restore from checkpoint
        if checkpoint:
            completed = set(checkpoint.completed_step_ids)
            failed = set(checkpoint.failed_step_ids)
            for sr in checkpoint.step_results.values():
                if sr.get("status") in ("completed",):
                    completed.add(sr["step_id"])
                elif sr.get("status") in ("failed", "timeout"):
                    failed.add(sr["step_id"])

        # Build adjacency
        adj = spec.adjacency
        step_map = {s.id: s for s in spec.steps}

        # Resolve next_on_success automatically for steps that don't set it
        for s in spec.steps:
            if s.next_on_success is None:
                children = adj.get(s.id, [])
                if children:
                    s.next_on_success = children

        try:
            while len(completed) + len(failed) + len(skipped) < len(spec.steps):
                if time.monotonic() > deadline:
                    wf_result.status = WorkflowStatus.TIMEOUT
                    _structured_log(logging.WARNING, "workflow.timeout",
                                    workflow_id=wf_result.workflow_id)
                    break

                # Find ready steps
                ready: list[WorkflowStep] = []
                for s in spec.steps:
                    if s.id in completed or s.id in failed or s.id in skipped:
                        continue
                    if s.id in wf_result.step_results:
                        existing = wf_result.step_results[s.id]
                        if existing.status in (StepStatus.COMPLETED, StepStatus.FAILED,
                                               StepStatus.SKIPPED, StepStatus.TIMEOUT):
                            continue
                    deps_met = all(d in completed for d in s.depends_on)
                    if deps_met:
                        ready.append(s)

                if not ready:
                    remaining = len(spec.steps) - len(completed) - len(failed) - len(skipped)
                    if remaining > 0:
                        _structured_log(logging.WARNING, "workflow.deadlock",
                                        workflow_id=wf_result.workflow_id,
                                        remaining=remaining,
                                        completed=len(completed), failed=len(failed))
                        blocked = [
                            s.id for s in spec.steps
                            if s.id not in completed and s.id not in failed and s.id not in skipped
                        ]
                        wf_result.error = f"Deadlock detected. Blocked steps: {blocked}"
                        wf_result.status = WorkflowStatus.FAILED
                    break

                # Execute ready steps (with parallelism)
                batch_start = _current_ms()
                semaphore = asyncio.Semaphore(min(spec.max_parallel, self._max_parallel))

                async def run_step(step: WorkflowStep) -> None:
                    async with semaphore:
                        await self._execute_single_step(
                            step, spec, wf_result, step_map,
                            completed, failed, skipped, deadline,
                        )

                tasks = [asyncio.create_task(run_step(s)) for s in ready]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Mark downstream steps as skipped when dependencies fail or get skipped
                changed = True
                while changed:
                    changed = False
                    for s in spec.steps:
                        if s.id in completed or s.id in failed or s.id in skipped:
                            continue
                        if any(d in failed for d in s.depends_on) or any(d in skipped for d in s.depends_on):
                            blocked_by = [d for d in s.depends_on if d in failed or d in skipped]
                            sr = StepResult(step_id=s.id, status=StepStatus.SKIPPED,
                                            agent_name=s.agent,
                                            error=f"Blocked by: {blocked_by}")
                            wf_result.step_results[s.id] = sr
                            skipped.add(s.id)
                            changed = True
                            _structured_log(logging.DEBUG, "step.skipped.dependency_failed",
                                            step_id=s.id, blocked_by=blocked_by)

                # Update progress
                progress = (len(completed) + len(failed) + len(skipped)) / len(spec.steps)
                for cb in self._progress_callbacks:
                    cb(wf_result.workflow_id, progress, None)

                # Save checkpoint after each batch
                if self._checkpoint_dir:
                    await self.save_checkpoint(wf_result.workflow_id)

            # Handle completion
            if wf_result.status not in (WorkflowStatus.TIMEOUT, WorkflowStatus.FAILED,
                                         WorkflowStatus.CANCELLED, WorkflowStatus.ROLLING_BACK):
                if failed:
                    wf_result.status = WorkflowStatus.FAILED
                else:
                    wf_result.status = WorkflowStatus.COMPLETED

        except asyncio.CancelledError:
            wf_result.status = WorkflowStatus.CANCELLED
            raise
        except Exception as exc:
            _structured_log(logging.ERROR, "workflow.crash",
                            workflow_id=wf_result.workflow_id, error=str(exc))
            wf_result.status = WorkflowStatus.FAILED
            wf_result.error = str(exc)
            await self._rollback_workflow(spec, wf_result, completed, step_map)

        finally:
            wf_result.completed_at = _now()
            wf_result.duration_ms = _current_ms() - wf_start
            wf_result.completed_steps = len(completed)
            wf_result.failed_steps = len(failed)
            wf_result.skipped_steps = len(skipped)
            _structured_log(logging.INFO, "workflow.completed",
                            workflow_id=wf_result.workflow_id,
                            status=wf_result.status.value,
                            duration_ms=wf_result.duration_ms,
                            completed=wf_result.completed_steps,
                            failed=wf_result.failed_steps)

        return wf_result

    async def _execute_single_step(
        self,
        step: WorkflowStep,
        spec: WorkflowSpec,
        wf_result: WorkflowResult,
        step_map: dict[str, WorkflowStep],
        completed: set[str],
        failed: set[str],
        skipped: set[str],
        deadline: float,
    ) -> None:
        log = _get_logger()

        # Check condition
        if step.condition is not None:
            source_results = wf_result.step_results.get(step.condition.source_step_id)
            source_ctx_val = None
            if source_results and source_results.output:
                source_ctx_val = source_results.output
            elif step.condition.source_step_id in wf_result.context:
                source_ctx_val = wf_result.context.get(step.condition.source_step_id)
            if not step.condition.evaluate(source_ctx_val):
                _structured_log(logging.DEBUG, "step.skipped.condition",
                                step_id=step.id,
                                condition=f"{step.condition.variable} {step.condition.operator.value}")
                sr = StepResult(step_id=step.id, status=StepStatus.SKIPPED, agent_name=step.agent)
                wf_result.step_results[step.id] = sr
                skipped.add(step.id)
                return

        # Check timeout
        if time.monotonic() > deadline:
            sr = StepResult(step_id=step.id, status=StepStatus.TIMEOUT, agent_name=step.agent)
            wf_result.step_results[step.id] = sr
            failed.add(step.id)
            return

        sr = wf_result.step_results.get(step.id, StepResult(
            step_id=step.id, agent_name=step.agent, status=StepStatus.PENDING,
        ))

        retry_policy = step.retry or spec.default_retry or RetryPolicy()
        last_error: str = ""

        for attempt in range(retry_policy.max_retries + 1):
            if time.monotonic() > deadline:
                sr.status = StepStatus.TIMEOUT
                break

            try:
                result = await asyncio.wait_for(
                    self._executor.execute_step(step, wf_result.context, sr),
                    timeout=step.timeout,
                )
                sr = result
                sr.retry_count = attempt
                wf_result.step_results[step.id] = sr

                if sr.status == StepStatus.COMPLETED:
                    completed.add(step.id)
                    _structured_log(logging.INFO, "step.completed",
                                    step_id=step.id, attempt=attempt)
                    # Evaluate conditions
                    for cond in spec.conditions:
                        if cond.source_step_id == step.id:
                            cond_val = sr.output
                            branch = cond.then_step_ids if cond.evaluate(cond_val) else cond.else_step_ids
                            _structured_log(logging.DEBUG, "workflow.branch",
                                            step_id=step.id, branch=branch)
                    return

                if sr.status == StepStatus.SKIPPED:
                    skipped.add(step.id)
                    return

                last_error = sr.error
                _structured_log(logging.WARNING, "step.retrying",
                                step_id=step.id, attempt=attempt,
                                max_retries=retry_policy.max_retries,
                                error=last_error[:100])

            except asyncio.TimeoutError:
                last_error = "Step timed out"
                _structured_log(logging.WARNING, "step.timeout",
                                step_id=step.id, attempt=attempt)
            except asyncio.CancelledError:
                sr.status = StepStatus.CANCELLED
                wf_result.step_results[step.id] = sr
                raise
            except Exception as exc:
                last_error = str(exc)
                _structured_log(logging.WARNING, "step.error",
                                step_id=step.id, attempt=attempt, error=last_error[:200])

            # Backoff before retry
            if attempt < retry_policy.max_retries:
                delay = _compute_backoff(attempt, retry_policy)
                await asyncio.sleep(delay)

        # All retries exhausted
        sr.status = StepStatus.FAILED
        sr.error = last_error
        sr.retry_count = retry_policy.max_retries
        wf_result.step_results[step.id] = sr
        failed.add(step.id)
        _structured_log(logging.ERROR, "step.failed",
                        step_id=step.id, error=last_error[:200])

        # Execute rollback if configured
        if step.rollback:
            _structured_log(logging.INFO, "step.rolling_back",
                            step_id=step.id,
                            rollback_step=step.rollback.step_id)
            await self._execute_rollback(step, spec, wf_result, completed, failed, step_map)

        # Propagate to next_on_failure steps
        failure_targets = step.next_on_failure
        if failure_targets:
            targets = failure_targets if isinstance(failure_targets, list) else [failure_targets]
            for t_id in targets:
                t_step = step_map.get(t_id)
                if t_step and t_step.id not in completed and t_step.id not in failed:
                    t_sr = StepResult(step_id=t_step.id, status=StepStatus.SKIPPED,
                                      agent_name=t_step.agent)
                    wf_result.step_results[t_step.id] = t_sr
                    skipped.add(t_step.id)
                    _structured_log(logging.DEBUG, "step.skipped.failure_propagation",
                                    step_id=t_step.id, due_to=step.id)

    async def _execute_rollback(
        self,
        step: WorkflowStep,
        spec: WorkflowSpec,
        wf_result: WorkflowResult,
        completed: set[str],
        failed: set[str],
        step_map: dict[str, WorkflowStep],
    ) -> None:
        if not step.rollback:
            return
        try:
            rb_step = WorkflowStep(
                id=step.rollback.step_id,
                agent=step.rollback.agent,
                task=step.rollback.task,
                timeout=step.rollback.timeout,
                context=step.rollback.context,
            )
            rb_sr = StepResult(step_id=rb_step.id, agent_name=rb_step.agent, status=StepStatus.RUNNING)
            await asyncio.wait_for(
                self._executor.execute_step(rb_step, wf_result.context, rb_sr),
                timeout=rb_step.timeout,
            )
            rb_sr.status = StepStatus.ROLLED_BACK
            wf_result.step_results[rb_step.id] = rb_sr
            _structured_log(logging.INFO, "step.rolled_back",
                            step_id=step.id, rollback_step=rb_step.id,
                            success=rb_sr.success)
        except Exception as exc:
            _structured_log(logging.ERROR, "step.rollback.failed",
                            step_id=step.id, error=str(exc))
            if not step.rollback.ignore_errors:
                raise

    async def _rollback_workflow(
        self,
        spec: WorkflowSpec,
        wf_result: WorkflowResult,
        completed: set[str],
        step_map: dict[str, WorkflowStep],
    ) -> None:
        """Roll back all completed steps that have rollback hooks."""
        _structured_log(logging.WARNING, "workflow.rolling_back",
                        workflow_id=wf_result.workflow_id)
        wf_result.status = WorkflowStatus.ROLLING_BACK
        for s in spec.steps:
            if s.id in completed and s.rollback:
                await self._execute_rollback(s, spec, wf_result, completed, set(), step_map)
        wf_result.status = WorkflowStatus.ROLLED_BACK

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    async def _write_checkpoint(self, checkpoint: Checkpoint) -> str | None:
        if not self._checkpoint_dir:
            return None
        p = Path(self._checkpoint_dir)
        p.mkdir(parents=True, exist_ok=True)
        path = p / f"wf-{checkpoint.workflow_id}-ckpt.json"
        try:
            path.write_text(checkpoint.to_json(), encoding="utf-8")
            _structured_log(logging.DEBUG, "checkpoint.saved",
                            path=str(path), workflow_id=checkpoint.workflow_id)
            return str(path)
        except Exception as exc:
            _structured_log(logging.ERROR, "checkpoint.save.failed",
                            error=str(exc), workflow_id=checkpoint.workflow_id)
            return None

    async def _load_checkpoint(self, checkpoint_id: str) -> Checkpoint | None:
        if not self._checkpoint_dir:
            return None
        p = Path(checkpoint_id)
        if not p.is_absolute():
            p = Path(self._checkpoint_dir) / checkpoint_id
        if not p.exists():
            # Try matching by workflow_id
            p2 = Path(self._checkpoint_dir) / f"wf-{checkpoint_id}-ckpt.json"
            if p2.exists():
                p = p2
            else:
                return None
        try:
            data = p.read_text(encoding="utf-8")
            return Checkpoint.from_json(data)
        except Exception as exc:
            _structured_log(logging.ERROR, "checkpoint.load.failed",
                            error=str(exc), path=str(p))
            return None


# ---------------------------------------------------------------------------
# Context interpolation
# ---------------------------------------------------------------------------


def _interpolate(template: str, context: dict[str, Any]) -> str:
    """Replace ``{key}`` placeholders with values from context."""
    result = template
    for key, value in context.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(placeholder, str(value))
    return result
