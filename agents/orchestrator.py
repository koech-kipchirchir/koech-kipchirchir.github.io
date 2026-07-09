"""
AIOS Multi-Agent Orchestrator
==============================

Production-grade orchestration framework for coordinating multiple AI agents
with support for dynamic registration, discovery, lifecycle management,
priority scheduling, parallel/sequential workflows, retry policies,
timeout handling, result aggregation, and metrics collection.

Typical usage::

    orchestrator = AgentOrchestrator()
    orchestrator.register_agent("coder", CodingAgent(config))
    orchestrator.register_agent("researcher", ResearchAgent(config))

    result = await orchestrator.execute("coder", "Write a Fibonacci function")
    results = await orchestrator.execute_parallel([
        ("researcher", "Latest AI news"),
        ("coder", "Implement sorting algorithm"),
    ])
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, AsyncIterator, Callable, Optional

from agents.base_agent import BaseAgent, AgentResult, AgentStatus

# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------

_ORCHESTRATOR_LOGGER: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _ORCHESTRATOR_LOGGER
    if _ORCHESTRATOR_LOGGER is None:
        _ORCHESTRATOR_LOGGER = logging.getLogger("aios.orchestrator")
    return _ORCHESTRATOR_LOGGER


def _structured_log(logger: logging.Logger, level: int, event: str, **kwargs: Any) -> None:
    record = {"event": event, "timestamp": time.time()}
    record.update(kwargs)
    logger.log(level, "%s", json.dumps(record, default=str))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Priority(Enum):
    """Task priority levels. Higher values preempt lower values."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(Enum):
    """Lifecycle status of a scheduled task."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMED_OUT = auto()


class WorkflowStatus(Enum):
    """Lifecycle status of a workflow execution."""

    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    PARTIALLY_COMPLETED = auto()
    CANCELLED = auto()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetryPolicy:
    """
    Configuration for task retry behaviour.

    Attributes:
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Initial delay in seconds before first retry (default 1.0).
        max_delay: Maximum delay in seconds between retries (default 60.0).
        exponential_backoff: If True, delay doubles after each attempt.
        jitter: Random jitter fraction added to delay (default 0.1).
        retryable_exceptions: Tuple of exception types that trigger a retry.
            If None, all exceptions are retried.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: float = 0.1
    retryable_exceptions: tuple[type[Exception], ...] | None = None


@dataclass
class Task:
    """
    A single unit of work to be executed by an agent.

    Attributes:
        id: Unique task identifier.
        agent_name: Name of the target agent.
        task: Natural-language task description.
        context: Optional contextual data passed to the agent.
        priority: Scheduling priority.
        timeout: Maximum execution time in seconds.
        retry_policy: Retry configuration. None means no retries.
        status: Current task status.
        result: Completed result (populated after execution).
    """

    id: str = ""
    agent_name: str = ""
    task: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    timeout: float = 120.0
    retry_policy: RetryPolicy | None = None
    status: TaskStatus = TaskStatus.PENDING
    result: AgentResult | None = None

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:16]


@dataclass
class WorkflowStep:
    """
    A single step in a workflow DAG.

    Attributes:
        id: Step identifier (used for dependency references).
        agent_name: Agent to execute this step.
        task_template: String template for the task. Supports ``{context.key}``
            interpolation from prior step results.
        depends_on: List of step IDs that must complete before this step runs.
        timeout: Per-step timeout override.
        retry_policy: Per-step retry override. None inherits workflow default.
        priority: Per-step priority override.
        context: Static context merged into step execution.
    """

    id: str = ""
    agent_name: str = ""
    task_template: str = ""
    depends_on: list[str] = field(default_factory=list)
    timeout: float | None = None
    retry_policy: RetryPolicy | None = None
    priority: Priority | None = None
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class Workflow:
    """
    A directed acyclic graph of steps executed by the orchestrator.

    Attributes:
        name: Workflow identifier.
        description: Human-readable description.
        steps: Ordered list of workflow steps.
        timeout: Overall workflow timeout (default 600 s).
        default_retry_policy: Fallback retry policy for steps without one.
    """

    name: str = ""
    description: str = ""
    steps: list[WorkflowStep] = field(default_factory=list)
    timeout: float = 600.0
    default_retry_policy: RetryPolicy | None = None


@dataclass
class OrchestratorConfig:
    """
    Global configuration for the :class:`AgentOrchestrator`.

    Attributes:
        default_timeout: Default task timeout in seconds (default 120).
        default_retry_policy: Default retry policy for tasks.
        max_parallel_tasks: Maximum concurrent task executions.
        enable_metrics: Collect and expose execution metrics.
        structured_logging: Emit JSON-structured log records.
        track_history: Keep per-session result history.
    """

    default_timeout: float = 120.0
    default_retry_policy: RetryPolicy | None = None
    max_parallel_tasks: int = 10
    enable_metrics: bool = True
    structured_logging: bool = True
    track_history: bool = True


@dataclass
class AgentMetadata:
    """
    Descriptive metadata registered alongside an agent.

    Attributes:
        name: Agent identifier (matches BaseAgent.config.name).
        description: Human-readable description of the agent's purpose.
        version: Agent version string.
        capabilities: List of capability tags for discovery.
        dependencies: Names of agents this agent depends on.
        timeout: Recommended execution timeout for this agent.
        requires_permission: Whether tool execution requires approval.
    """

    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    capabilities: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    timeout: float = 120.0
    requires_permission: bool = False


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

@dataclass
class AgentMetrics:
    """Aggregated metrics for a single agent."""

    total_tasks: int = 0
    successful: int = 0
    failed: int = 0
    timed_out: int = 0
    total_duration_ms: float = 0.0
    min_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    total_tokens: int = 0
    last_execution: float = 0.0


@dataclass
class OrchestratorMetrics:
    """Aggregated orchestrator-wide metrics snapshot."""

    total_tasks: int = 0
    total_successful: int = 0
    total_failed: int = 0
    total_timed_out: int = 0
    avg_duration_ms: float = 0.0
    agents: dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Thread-safe collector for task execution metrics.

    Collects per-agent and aggregate metrics including execution counts,
    durations, token usage, and success/failure rates.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._agents: dict[str, AgentMetrics] = defaultdict(AgentMetrics)
        self._records: list[dict[str, Any]] = []
        self._max_records: int = 10_000

    async def record(
        self,
        agent_name: str,
        success: bool,
        duration_ms: float,
        tokens_used: int = 0,
        timed_out: bool = False,
    ) -> None:
        async with self._lock:
            m = self._agents[agent_name]
            m.total_tasks += 1
            m.total_duration_ms += duration_ms
            m.total_tokens += tokens_used
            m.last_execution = time.time()

            if m.min_duration_ms == 0 or duration_ms < m.min_duration_ms:
                m.min_duration_ms = duration_ms
            if duration_ms > m.max_duration_ms:
                m.max_duration_ms = duration_ms

            if timed_out:
                m.timed_out += 1
                m.failed += 1
            elif success:
                m.successful += 1
            else:
                m.failed += 1

            self._records.append({
                "agent": agent_name,
                "success": success,
                "duration_ms": duration_ms,
                "tokens": tokens_used,
                "timed_out": timed_out,
                "timestamp": time.time(),
            })
            if len(self._records) > self._max_records:
                self._records = self._records[-self._max_records:]

    async def snapshot(self) -> OrchestratorMetrics:
        async with self._lock:
            total = 0
            suc = 0
            fail = 0
            to = 0
            dur = 0.0
            agents = {}
            for name, m in self._agents.items():
                total += m.total_tasks
                suc += m.successful
                fail += m.failed
                to += m.timed_out
                dur += m.total_duration_ms
                agents[name] = {
                    "total_tasks": m.total_tasks,
                    "successful": m.successful,
                    "failed": m.failed,
                    "timed_out": m.timed_out,
                    "avg_duration_ms": round(m.total_duration_ms / m.total_tasks, 2) if m.total_tasks else 0.0,
                    "min_duration_ms": m.min_duration_ms,
                    "max_duration_ms": m.max_duration_ms,
                    "total_tokens": m.total_tokens,
                }
            return OrchestratorMetrics(
                total_tasks=total,
                total_successful=suc,
                total_failed=fail,
                total_timed_out=to,
                avg_duration_ms=round(dur / total, 2) if total else 0.0,
                agents=agents,
            )

    async def agent_metrics(self, agent_name: str) -> AgentMetrics | None:
        async with self._lock:
            return self._agents.get(agent_name)

    async def clear(self) -> None:
        async with self._lock:
            self._agents.clear()
            self._records.clear()


# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """
    Registry for agent registration, discovery, and lifecycle tracking.

    Supports dynamic registration with rich metadata, capability-based
    discovery, dependency validation, and lifecycle state tracking.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._metadata: dict[str, AgentMetadata] = {}
        self._lifecycle: dict[str, AgentStatus] = {}
        self._lock = asyncio.Lock()
        self._log = _get_logger()

    async def register(
        self,
        agent: BaseAgent,
        metadata: AgentMetadata | None = None,
    ) -> None:
        """Register an agent with optional metadata."""
        async with self._lock:
            name = agent.name
            if name in self._agents:
                self._log.warning("Overwriting existing agent: %s", name)
            self._agents[name] = agent
            self._metadata[name] = metadata or AgentMetadata(name=name)
            self._lifecycle[name] = AgentStatus.IDLE
            _structured_log(
                self._log, logging.INFO, "agent_registered",
                agent=name, capabilities=(
                    metadata.capabilities if metadata else []
                ),
            )

    async def unregister(self, name: str) -> bool:
        """Remove an agent from the registry. Returns True if removed."""
        async with self._lock:
            if name not in self._agents:
                return False
            del self._agents[name]
            self._metadata.pop(name, None)
            self._lifecycle.pop(name, None)
            _structured_log(self._log, logging.INFO, "agent_unregistered", agent=name)
            return True

    def get(self, name: str) -> BaseAgent | None:
        """Retrieve an agent by name."""
        return self._agents.get(name)

    def get_metadata(self, name: str) -> AgentMetadata | None:
        """Retrieve metadata for a registered agent."""
        return self._metadata.get(name)

    async def discover(self, capability: str) -> list[str]:
        """
        Discover agents that match a given capability.

        Performs case-insensitive substring matching against capability tags.
        """
        cap_lower = capability.lower()
        results: list[str] = []
        async with self._lock:
            for name, meta in self._metadata.items():
                if any(cap_lower in c.lower() for c in meta.capabilities):
                    results.append(name)
            for name in self._agents:
                if name not in results:
                    meta = self._metadata.get(name)
                    if meta is None and cap_lower in name.lower():
                        results.append(name)
        return results

    @property
    def available_agents(self) -> list[str]:
        """Return a snapshot of registered agent names."""
        return list(self._agents.keys())

    async def lifecycle_status(self, name: str) -> AgentStatus | None:
        """Get the lifecycle status of a registered agent."""
        return self._lifecycle.get(name)

    async def set_lifecycle(self, name: str, status: AgentStatus) -> None:
        """Update the lifecycle status of a registered agent."""
        async with self._lock:
            self._lifecycle[name] = status

    async def validate_dependencies(self, name: str) -> list[str]:
        """
        Return a list of unresolved dependencies for the given agent.
        An empty list means all dependencies are satisfied.
        """
        meta = self._metadata.get(name)
        if meta is None:
            return []
        missing = []
        async with self._lock:
            for dep in meta.dependencies:
                if dep not in self._agents:
                    missing.append(dep)
        return missing


# ---------------------------------------------------------------------------
# Task scheduler
# ---------------------------------------------------------------------------

class _ScheduledTask:
    """Internal wrapper for scheduling priority queue."""

    def __init__(
        self,
        task: Task,
        priority: int,
        sequence: int,
    ) -> None:
        self.task = task
        self.priority = priority
        self.sequence = sequence

    def __lt__(self, other: _ScheduledTask) -> bool:
        if self.priority == other.priority:
            return self.sequence < other.sequence
        return self.priority > other.priority


class TaskScheduler:
    """
    Priority-based task scheduler with concurrency control.

    Manages a priority queue of tasks, respects max concurrency limits,
    and supports task cancellation and status tracking.
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        self._queue: asyncio.PriorityQueue[_ScheduledTask] = asyncio.PriorityQueue()
        self._running: dict[str, Task] = {}
        self._completed: dict[str, Task] = {}
        self._cancelled: set[str] = set()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._sequence: int = 0
        self._lock = asyncio.Lock()
        self._log = _get_logger()

    async def schedule(self, task: Task) -> None:
        """Add a task to the scheduling queue."""
        async with self._lock:
            self._sequence += 1
            priority_val = task.priority.value if isinstance(task.priority, Priority) else int(task.priority)
        await self._queue.put(_ScheduledTask(task, priority_val, self._sequence))
        _structured_log(
            self._log, logging.INFO, "task_scheduled",
            task_id=task.id, agent=task.agent_name,
            priority=task.priority.name if isinstance(task.priority, Priority) else priority_val,
        )

    async def acquire(self) -> Task | None:
        """Acquire the next ready task from the queue (blocks until available)."""
        while True:
            try:
                st = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                return None
            if st.task.id in self._cancelled:
                self._cancelled.discard(st.task.id)
                st.task.status = TaskStatus.CANCELLED
                self._completed[st.task.id] = st.task
                continue
            async with self._lock:
                self._running[st.task.id] = st.task
            st.task.status = TaskStatus.RUNNING
            return st.task

    async def complete(self, task: Task) -> None:
        """Mark a task as completed and release its slot."""
        async with self._lock:
            self._running.pop(task.id, None)
            self._completed[task.id] = task
        self._semaphore.release()
        _structured_log(
            self._log, logging.INFO, "task_completed",
            task_id=task.id, status=task.status.name,
        )

    async def cancel(self, task_id: str) -> bool:
        """Cancel a pending or running task. Returns True if found."""
        async with self._lock:
            if task_id in self._running:
                task = self._running[task_id]
                task.status = TaskStatus.CANCELLED
                return True
            if task_id not in self._cancelled:
                self._cancelled.add(task_id)
                return True
            return False

    def task_status(self, task_id: str) -> TaskStatus | None:
        """Get the current status of a task by ID."""
        if task_id in self._running:
            return self._running[task_id].status
        if task_id in self._completed:
            return self._completed[task_id].status
        return None

    def running_count(self) -> int:
        return len(self._running)

    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def semaphore(self) -> asyncio.Semaphore:
        return self._semaphore


# ---------------------------------------------------------------------------
# Result aggregator
# ---------------------------------------------------------------------------

class AggregationStrategy(Enum):
    """Strategies for combining multiple agent results."""

    CONCAT = auto()
    MERGE = auto()
    BEST_OF = auto()
    VOTE = auto()
    SEQUENTIAL = auto()


class ResultAggregator:
    """
    Combines multiple AgentResult objects using configurable strategies.

    Supports concatenation, merging into a single dict, best-of selection
    (highest confidence), majority voting, and sequential accumulation.
    """

    @staticmethod
    def concat(results: list[AgentResult], separator: str = "\n\n---\n\n") -> AgentResult:
        """Concatenate all outputs into a single result."""
        if not results:
            return AgentResult(success=False, output="", agent_name="aggregator")
        combined = separator.join(r.output for r in results)
        total_tokens = sum(r.tokens_used for r in results)
        total_duration = sum(r.duration_ms for r in results)
        all_success = all(r.success for r in results)
        return AgentResult(
            success=all_success,
            output=combined,
            agent_name="aggregator",
            duration_ms=total_duration,
            tokens_used=total_tokens,
        )

    @staticmethod
    def merge(results: list[AgentResult]) -> AgentResult:
        """Merge structured outputs (JSON dicts) into a single result."""
        if not results:
            return AgentResult(success=False, output="{}", agent_name="aggregator")
        merged: dict[str, Any] = {}
        total_tokens = 0
        total_duration = 0.0
        all_success = True
        for r in results:
            total_tokens += r.tokens_used
            total_duration += r.duration_ms
            if not r.success:
                all_success = False
            try:
                data = json.loads(r.output)
                if isinstance(data, dict):
                    merged.update(data)
                else:
                    merged[f"result_{len(merged)}"] = data
            except (json.JSONDecodeError, TypeError):
                merged[f"result_{len(merged)}"] = r.output
        return AgentResult(
            success=all_success,
            output=json.dumps(merged, default=str),
            agent_name="aggregator",
            duration_ms=total_duration,
            tokens_used=total_tokens,
        )

    @staticmethod
    def best_of(
        results: list[AgentResult],
        score_key: str = "confidence",
    ) -> AgentResult:
        """Select the result with the highest metadata score."""
        if not results:
            return AgentResult(success=False, output="", agent_name="aggregator")
        scored = [
            (r, r.metadata.get(score_key, 0.0) if isinstance(r.metadata, dict) else 0.0)
            for r in results
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[0][0]

    @staticmethod
    def vote(results: list[AgentResult]) -> AgentResult:
        """Majority-vote among boolean or categorical outputs."""
        if not results:
            return AgentResult(success=False, output="", agent_name="aggregator")
        outputs = [r.output.strip().lower() for r in results]
        from collections import Counter
        counter = Counter(outputs)
        winner = counter.most_common(1)[0][0]
        winner_result = next(r for r in results if r.output.strip().lower() == winner)
        return winner_result

    @staticmethod
    def sequential(results: list[AgentResult]) -> AgentResult:
        """Accumulate results sequentially, passing context between steps."""
        if not results:
            return AgentResult(success=False, output="", agent_name="aggregator")
        full_output: list[str] = []
        total_tokens = 0
        total_duration = 0.0
        all_success = True
        for r in results:
            full_output.append(r.output)
            total_tokens += r.tokens_used
            total_duration += r.duration_ms
            if not r.success:
                all_success = False
        return AgentResult(
            success=all_success,
            output="\n".join(full_output),
            agent_name="aggregator",
            duration_ms=total_duration,
            tokens_used=total_tokens,
        )


# ---------------------------------------------------------------------------
# Workflow executor
# ---------------------------------------------------------------------------

class WorkflowExecutor:
    """
    Executes workflows defined as DAGs of steps with dependency resolution,
    parallel execution, timeout enforcement, and result aggregation.

    Steps are executed respecting their ``depends_on`` declarations.
    Steps with no unmet dependencies run in parallel up to the concurrency
    limit. The executor supports context propagation between steps via
    ``task_template`` string interpolation.
    """

    def __init__(self, orchestrator: AgentOrchestrator) -> None:
        self._orchestrator = orchestrator
        self._log = _get_logger()

    async def execute(
        self,
        workflow: Workflow,
        initial_context: dict[str, Any] | None = None,
    ) -> list[AgentResult]:
        """
        Execute a workflow DAG.

        Args:
            workflow: The workflow definition to execute.
            initial_context: Global context available to all step templates.

        Returns:
            List of AgentResult objects in step-defined order.
        """
        context: dict[str, Any] = dict(initial_context or {})
        context["workflow_name"] = workflow.name
        results: dict[str, AgentResult] = {}
        step_map: dict[str, WorkflowStep] = {s.id: s for s in workflow.steps}
        completed: set[str] = set()
        failed: set[str] = set()

        _structured_log(
            self._log, logging.INFO, "workflow_started",
            workflow=workflow.name, steps=len(workflow.steps),
        )

        overall_deadline = time.time() + workflow.timeout

        while len(completed) + len(failed) < len(workflow.steps):
            if time.time() > overall_deadline:
                _structured_log(
                    self._log, logging.ERROR, "workflow_timed_out",
                    workflow=workflow.name,
                )
                break

            ready: list[WorkflowStep] = []
            for step in workflow.steps:
                if step.id in completed or step.id in failed:
                    continue
                deps_met = all(d in completed for d in step.depends_on)
                if deps_met:
                    ready.append(step)

            if not ready and len(completed) + len(failed) < len(workflow.steps):
                _structured_log(
                    self._log, logging.WARNING, "workflow_deadlock",
                    workflow=workflow.name,
                    completed=len(completed), failed=len(failed),
                )
                break

            coros = [self._execute_step(step, context, step_map, results, workflow) for step in ready]
            step_results = await asyncio.gather(*coros, return_exceptions=True)

            for step, sr in zip(ready, step_results):
                if isinstance(sr, Exception):
                    _structured_log(
                        self._log, logging.ERROR, "step_failed",
                        step=step.id, error=str(sr),
                    )
                    failed.add(step.id)
                elif sr is not None:
                    results[step.id] = sr
                    context[step.id] = sr.output
                    context.setdefault("results", {})[step.id] = sr.output
                    if sr.success:
                        completed.add(step.id)
                    else:
                        failed.add(step.id)

        ordered_results = [results[s.id] for s in workflow.steps if s.id in results]

        _structured_log(
            self._log, logging.INFO, "workflow_completed",
            workflow=workflow.name,
            completed=len(completed), failed=len(failed),
            results=len(ordered_results),
        )
        return ordered_results

    async def _execute_step(
        self,
        step: WorkflowStep,
        context: dict[str, Any],
        step_map: dict[str, WorkflowStep],
        prior_results: dict[str, AgentResult],
        workflow: Workflow,
    ) -> AgentResult | None:
        task_text = self._interpolate(step.task_template, context)
        step_timeout = step.timeout or workflow.timeout / max(len(workflow.steps), 1)
        retry = step.retry_policy or workflow.default_retry_policy

        try:
            result = await self._orchestrator.execute(
                agent_name=step.agent_name,
                task=task_text,
                context={**step.context, **context},
                timeout=step_timeout,
                retry_policy=retry,
                priority=step.priority if step.priority is not None else Priority.MEDIUM,
            )
        except asyncio.TimeoutError:
            return AgentResult(
                success=False,
                output="",
                agent_name=step.agent_name,
                error="Step timed out",
                metadata={"step": step.id, "timed_out": True},
            )

        if result.metadata is None:
            result.metadata = {}
        if isinstance(result.metadata, dict):
            result.metadata["step"] = step.id
        return result

    @staticmethod
    def _interpolate(template: str, context: dict[str, Any]) -> str:
        """Replace ``{key}`` placeholders with values from context."""
        result = template
        for key, value in context.items():
            placeholder = "{" + key + "}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        return result


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Production-grade multi-agent orchestrator.

    Coordinates agent registration, task scheduling, workflow execution,
    result aggregation, metrics collection, and structured logging.

    Typical usage::

        orchestrator = AgentOrchestrator()
        orchestrator.register_agent("coder", CodingAgent(AgentConfig(name="coder")))

        # Single task
        result = await orchestrator.execute("coder", "Write hello world")

        # Parallel tasks
        results = await orchestrator.execute_parallel([
            ("researcher", "AI news"),
            ("coder", "sort algorithm"),
        ])

        # Workflow
        workflow = Workflow(name="research-then-code", steps=[...])
        results = await orchestrator.run_workflow(workflow)
    """

    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        registry: AgentRegistry | None = None,
        scheduler: TaskScheduler | None = None,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.registry = registry or AgentRegistry()
        self.scheduler = scheduler or TaskScheduler(self.config.max_parallel_tasks)
        self.metrics = MetricsCollector() if self.config.enable_metrics else None
        self.workflow_executor = WorkflowExecutor(self)
        self._results: dict[str, list[AgentResult]] = {}
        self._log = _get_logger()

    # ------------------------------------------------------------------
    # Agent registration & discovery
    # ------------------------------------------------------------------

    async def register_agent(
        self,
        agent: BaseAgent,
        metadata: AgentMetadata | None = None,
    ) -> None:
        """Register an agent with optional metadata for discovery."""
        await self.registry.register(agent, metadata)
        _structured_log(
            self._log, logging.INFO, "orchestrator_agent_registered",
            agent=agent.name,
        )

    async def unregister_agent(self, name: str) -> bool:
        """Unregister an agent by name."""
        return await self.registry.unregister(name)

    async def discover_agents(self, capability: str) -> list[str]:
        """Discover agents matching a capability keyword."""
        return await self.registry.discover(capability)

    def get_agent(self, name: str) -> BaseAgent | None:
        """Retrieve a registered agent by name."""
        return self.registry.get(name)

    async def get_agent_metadata(self, name: str) -> AgentMetadata | None:
        """Get metadata for a registered agent."""
        return self.registry.get_metadata(name)

    @property
    def available_agents(self) -> list[str]:
        """Return names of all registered agents."""
        return self.registry.available_agents

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    async def start_agent(self, name: str) -> bool:
        """Mark an agent as active."""
        agent = self.registry.get(name)
        if agent is None:
            return False
        agent.status = AgentStatus.IDLE
        await self.registry.set_lifecycle(name, AgentStatus.IDLE)
        return True

    async def stop_agent(self, name: str) -> bool:
        """Mark an agent as idle and cancel any running tasks."""
        agent = self.registry.get(name)
        if agent is None:
            return False
        agent.cancel()
        agent.status = AgentStatus.IDLE
        await self.registry.set_lifecycle(name, AgentStatus.IDLE)
        return True

    async def agent_status(self, name: str) -> AgentStatus | None:
        """Get the lifecycle status of an agent."""
        return await self.registry.lifecycle_status(name)

    # ------------------------------------------------------------------
    # Task execution
    # ------------------------------------------------------------------

    async def execute(
        self,
        agent_name: str,
        task: str,
        context: dict[str, Any] | None = None,
        priority: Priority | int = Priority.MEDIUM,
        timeout: float | None = None,
        retry_policy: RetryPolicy | None = None,
    ) -> AgentResult:
        """
        Execute a single task on a named agent with optional retry and timeout.

        Args:
            agent_name: Target agent identifier.
            task: Natural-language task description.
            context: Contextual data passed to the agent.
            priority: Scheduling priority.
            timeout: Maximum execution time (overrides config default).
            retry_policy: Retry policy for this task.

        Returns:
            The AgentResult produced by the agent.

        Raises:
            ValueError: If the agent is not found.
            asyncio.TimeoutError: If the task times out and no retry recovers.
        """
        agent = self.registry.get(agent_name)
        if agent is None:
            raise ValueError(f"Agent not found: {agent_name}")

        effective_timeout = timeout if timeout is not None else self.config.default_timeout
        effective_retry = retry_policy if retry_policy is not None else self.config.default_retry_policy

        if priority is None:
            priority = Priority.MEDIUM
        elif not isinstance(priority, Priority):
            priority = Priority(priority)

        t = Task(
            agent_name=agent_name,
            task=task,
            context=dict(context or {}),
            priority=priority,
            timeout=effective_timeout,
            retry_policy=effective_retry,
        )

        await self.scheduler.schedule(t)
        result = await self._execute_with_retries(agent, t)

        if self.config.track_history:
            sid = agent.session_id
            if sid not in self._results:
                self._results[sid] = []
            self._results[sid].append(result)

        if self.metrics:
            await self.metrics.record(
                agent_name=agent_name,
                success=result.success,
                duration_ms=result.duration_ms,
                tokens_used=result.tokens_used,
                timed_out=(t.status == TaskStatus.TIMED_OUT),
            )

        return result

    async def _execute_with_retries(self, agent: BaseAgent, task: Task) -> AgentResult:
        last_error: str = ""
        max_attempts = 1
        policy = task.retry_policy

        if policy is not None:
            max_attempts = 1 + policy.max_retries

        for attempt in range(1, max_attempts + 1):
            start = time.perf_counter()

            try:
                async with self.scheduler.semaphore:
                    acquired = await self.scheduler.acquire()
                    if acquired is None or acquired.id != task.id:
                        pass

                    result = await asyncio.wait_for(
                        agent.execute(task.task, task.context),
                        timeout=task.timeout,
                    )

                elapsed = (time.perf_counter() - start) * 1000
                result.duration_ms = elapsed

                task.result = result
                task.status = TaskStatus.COMPLETED if result.success else TaskStatus.FAILED
                await self.scheduler.complete(task)

                if not result.success and attempt < max_attempts:
                    last_error = result.error
                    delay = self._backoff_delay(attempt, policy)
                    _structured_log(
                        self._log, logging.WARNING, "task_retrying",
                        task_id=task.id, attempt=attempt,
                        max_attempts=max_attempts, delay=delay,
                        error=result.error,
                    )
                    await asyncio.sleep(delay)
                    continue

                return result

            except asyncio.TimeoutError:
                elapsed = (time.perf_counter() - start) * 1000
                task.status = TaskStatus.TIMED_OUT
                await self.scheduler.complete(task)

                if attempt < max_attempts:
                    delay = self._backoff_delay(attempt, policy)
                    _structured_log(
                        self._log, logging.WARNING, "task_timeout_retrying",
                        task_id=task.id, attempt=attempt,
                        max_attempts=max_attempts, delay=delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                _structured_log(
                    self._log, logging.ERROR, "task_timed_out",
                    task_id=task.id, agent=task.agent_name,
                    timeout=task.timeout,
                )
                return AgentResult(
                    success=False,
                    output="",
                    agent_name=task.agent_name,
                    duration_ms=elapsed,
                    error=f"Task timed out after {task.timeout}s",
                    metadata={"timed_out": True, "attempts": attempt},
                )

            except Exception as exc:
                elapsed = (time.perf_counter() - start) * 1000
                last_error = str(exc)

                should_retry = True
                if policy and policy.retryable_exceptions:
                    should_retry = isinstance(exc, policy.retryable_exceptions)

                if should_retry and attempt < max_attempts:
                    delay = self._backoff_delay(attempt, policy)
                    _structured_log(
                        self._log, logging.WARNING, "task_error_retrying",
                        task_id=task.id, attempt=attempt,
                        max_attempts=max_attempts, delay=delay,
                        error=last_error,
                    )
                    await asyncio.sleep(delay)
                    continue

                task.status = TaskStatus.FAILED
                task.result = AgentResult(
                    success=False,
                    output="",
                    agent_name=task.agent_name,
                    duration_ms=elapsed,
                    error=last_error,
                )
                await self.scheduler.complete(task)
                return task.result

        return AgentResult(
            success=False,
            output="",
            agent_name=task.agent_name,
            error=last_error,
        )

    @staticmethod
    def _backoff_delay(attempt: int, policy: RetryPolicy | None) -> float:
        if policy is None:
            return 0.0
        delay = policy.base_delay
        if policy.exponential_backoff:
            delay *= 2.0 ** (attempt - 1)
        delay = min(delay, policy.max_delay)
        if policy.jitter > 0:
            import random
            delay *= 1.0 + random.uniform(-policy.jitter, policy.jitter)
        return delay

    # ------------------------------------------------------------------
    # Parallel execution
    # ------------------------------------------------------------------

    async def execute_parallel(
        self,
        tasks: list[tuple[str, str]],
        context: dict[str, Any] | None = None,
        priority: Priority | int = Priority.MEDIUM,
        timeout: float | None = None,
    ) -> list[AgentResult]:
        """
        Execute multiple tasks in parallel across named agents.

        Args:
            tasks: List of ``(agent_name, task)`` tuples.
            context: Shared context for all tasks.
            priority: Scheduling priority for all tasks.
            timeout: Per-task timeout override.

        Returns:
            List of AgentResult objects (order matches input order).
        """
        coros = [
            self.execute(
                agent_name=name,
                task=task,
                context=context,
                priority=priority,
                timeout=timeout,
            )
            for name, task in tasks
        ]
        return await asyncio.gather(*coros, return_exceptions=True)

    # ------------------------------------------------------------------
    # Sequential execution (pipeline)
    # ------------------------------------------------------------------

    async def execute_sequential(
        self,
        pipeline: list[tuple[str, str]],
        initial_context: dict[str, Any] | None = None,
        priority: Priority | int = Priority.MEDIUM,
    ) -> AgentResult:
        """
        Execute tasks sequentially, passing output context between steps.

        Each step receives the accumulated context containing prior step
        results accessible via ``{step_index}`` or ``{results}`` keys.

        Args:
            pipeline: List of ``(agent_name, task_template)`` tuples.
            initial_context: Base context for the first step.
            priority: Scheduling priority for all steps.

        Returns:
            Aggregated AgentResult containing all step outputs.
        """
        context: dict[str, Any] = dict(initial_context or {})
        context["results"] = {}
        results: list[AgentResult] = []

        for i, (agent_name, task_template) in enumerate(pipeline):
            task_text = task_template
            for key, value in context.items():
                placeholder = "{" + key + "}"
                if placeholder in task_text:
                    task_text = task_text.replace(placeholder, str(value))

            result = await self.execute(
                agent_name=agent_name,
                task=task_text,
                context=dict(context),
                priority=priority,
            )
            results.append(result)
            context[str(i)] = result.output
            context["results"][str(i)] = {
                "agent": agent_name,
                "output": result.output,
                "success": result.success,
            }

        return ResultAggregator.concat(results)

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    async def run_workflow(
        self,
        workflow: Workflow,
        initial_context: dict[str, Any] | None = None,
    ) -> list[AgentResult]:
        """
        Execute a full workflow DAG.

        Args:
            workflow: The workflow definition with steps and dependencies.
            initial_context: Global context accessible in step templates.

        Returns:
            List of AgentResult objects in workflow step order.
        """
        return await self.workflow_executor.execute(workflow, initial_context)

    # ------------------------------------------------------------------
    # Results & history
    # ------------------------------------------------------------------

    def get_session_results(self, session_id: str) -> list[AgentResult]:
        """Return all results recorded for a given session."""
        return self._results.get(session_id, [])

    def clear_session(self, session_id: str) -> None:
        """Remove result history for a session."""
        self._results.pop(session_id, None)

    def clear_all_sessions(self) -> None:
        """Remove all session result history."""
        self._results.clear()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    async def get_metrics(self) -> OrchestratorMetrics | None:
        """Return a snapshot of current execution metrics."""
        if self.metrics is None:
            return None
        return await self.metrics.snapshot()

    async def clear_metrics(self) -> None:
        """Reset all collected metrics."""
        if self.metrics:
            await self.metrics.clear()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully shut down the orchestrator, cancelling pending tasks."""
        _structured_log(self._log, logging.INFO, "orchestrator_shutdown")
        for name in self.registry.available_agents:
            await self.stop_agent(name)
        self.clear_all_sessions()
