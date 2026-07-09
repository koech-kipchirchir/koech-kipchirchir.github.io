"""
AIOS Planner Agent
==================

Production-grade planner that decomposes complex goals into executable task
plans with dependency estimation, prioritization, tool/memory/RAG integration,
and failure-driven re-planning.

Typical usage::

    planner = PlannerAgent(AgentConfig(name="planner"))
    plan = await planner.execute("Build a web app with user authentication")

    # Inspect the plan
    for task in plan.tasks:
        print(f"{task.id}: {task.description} [{task.priority.name}]")

    # Re-plan after a failure
    new_plan = await planner.replan(plan, failed_task_id="task_3", error="Module not found")
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from agents.base_agent import AgentConfig, AgentResult, BaseAgent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("aios.agent.planner")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TaskPriority(Enum):
    """Priority level for a planned task."""

    CRITICAL = 0
    HIGH = 1
    MEDIUM = 2
    LOW = 3


class TaskState(Enum):
    """Execution state of a planned task."""

    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ToolRequirement:
    """Describes a tool a task may need during execution."""

    tool_name: str = ""
    purpose: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryQuery:
    """Describes a memory retrieval operation a task should perform."""

    query: str = ""
    session_id: str = ""
    top_k: int = 5


@dataclass
class RAGQuery:
    """Describes a RAG retrieval operation a task should perform."""

    query: str = ""
    top_k: int = 5
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlannedTask:
    """
    A single unit of work within an execution plan.

    Attributes:
        id: Unique identifier for this task.
        description: Natural-language description of the work.
        agent_type: Agent type best suited for this task.
        priority: Scheduling priority.
        depends_on: IDs of tasks that must complete before this one.
        estimated_duration_seconds: Rough execution time estimate.
        requires_tools: List of tools this task likely needs.
        requires_memory: Whether this task should retrieve conversation memory.
        memory_query: Optional structured memory query parameters.
        requires_rag: Whether this task should perform RAG retrieval.
        rag_query: Optional structured RAG query parameters.
        context: Additional contextual data for the executing agent.
        state: Current execution state of this task.
        error: Error message if the task failed.
    """

    id: str = ""
    description: str = ""
    agent_type: str = "research"
    priority: TaskPriority = TaskPriority.MEDIUM
    depends_on: list[str] = field(default_factory=list)
    estimated_duration_seconds: float = 30.0
    requires_tools: list[ToolRequirement] = field(default_factory=list)
    requires_memory: bool = False
    memory_query: MemoryQuery | None = None
    requires_rag: bool = False
    rag_query: RAGQuery | None = None
    context: dict[str, Any] = field(default_factory=dict)
    state: TaskState = TaskState.PENDING
    error: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f"task_{uuid.uuid4().hex[:8]}"


@dataclass
class ExecutionPlan:
    """
    A complete execution plan produced by the PlannerAgent.

    Attributes:
        goal: The original high-level goal or task description.
        tasks: Ordered list of planned tasks.
        parallel_groups: Groups of task IDs that can execute concurrently.
        critical_path: Sequence of task IDs on the critical path.
        estimated_total_duration_seconds: Sum of critical-path durations.
        created_at: Unix timestamp of plan creation.
        version: Plan version (incremented on re-plan).
        metadata: Arbitrary additional data.
    """

    goal: str = ""
    tasks: list[PlannedTask] = field(default_factory=list)
    parallel_groups: list[list[str]] = field(default_factory=list)
    critical_path: list[str] = field(default_factory=list)
    estimated_total_duration_seconds: float = 0.0
    created_at: float = 0.0
    version: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to a JSON-compatible dictionary."""
        return {
            "goal": self.goal,
            "version": self.version,
            "created_at": self.created_at,
            "estimated_total_duration_seconds": self.estimated_total_duration_seconds,
            "total_tasks": len(self.tasks),
            "tasks": [
                {
                    "id": t.id,
                    "description": t.description,
                    "agent_type": t.agent_type,
                    "priority": t.priority.name,
                    "depends_on": list(t.depends_on),
                    "estimated_duration_seconds": t.estimated_duration_seconds,
                    "requires_tools": [
                        {"tool_name": r.tool_name, "purpose": r.purpose}
                        for r in t.requires_tools
                    ],
                    "requires_memory": t.requires_memory,
                    "requires_rag": t.requires_rag,
                    "state": t.state.value,
                }
                for t in self.tasks
            ],
            "parallel_groups": [list(g) for g in self.parallel_groups],
            "critical_path": list(self.critical_path),
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Planner configuration
# ---------------------------------------------------------------------------


@dataclass
class PlannerConfig:
    """
    Configuration for the PlannerAgent.

    Attributes:
        default_agent_type: Fallback agent type when classification is unclear.
        min_task_length: Minimum character length for a task description.
        max_tasks: Maximum number of tasks a plan can contain.
        duration_per_task_seconds: Default estimated duration per task.
        enable_tool_detection: Automatically detect tool requirements.
        enable_memory_detection: Automatically detect memory requirements.
        enable_rag_detection: Automatically detect RAG requirements.
        use_llm_decomposition: Use an LLM for decomposition (if available).
        llm_model: Model name to use for LLM-based decomposition.
    """

    default_agent_type: str = "research"
    min_task_length: int = 10
    max_tasks: int = 20
    duration_per_task_seconds: float = 30.0
    enable_tool_detection: bool = True
    enable_memory_detection: bool = True
    enable_rag_detection: bool = True
    use_llm_decomposition: bool = False
    llm_model: str = "gpt-4o"


# ---------------------------------------------------------------------------
# Agent-type classifier
# ---------------------------------------------------------------------------

_AGENT_KEYWORDS: dict[str, list[str]] = {
    "coding": [
        "code", "write", "implement", "program", "function", "script",
        "debug", "refactor", "compile", "build", "develop", "software",
        "api", "endpoint", "class", "method", "algorithm", "data structure",
    ],
    "research": [
        "research", "search", "find", "look up", "what is", "who is",
        "explain", "tell me about", "analyze", "investigate", "study",
        "report", "summarize", "information", "knowledge",
    ],
    "math": [
        "calculate", "compute", "math", "equation", "sum", "solve",
        "derivative", "integral", "statistics", "probability", "algebra",
        "geometry", "formula", "numerical", "quantitative",
    ],
    "cybersecurity": [
        "security", "vulnerability", "threat", "attack", "exploit",
        "pentest", "firewall", "encrypt", "decrypt", "malware",
        "authentication", "authorization", "audit", "compliance",
    ],
    "trading": [
        "trade", "stock", "market", "invest", "portfolio", "crypto",
        "price", "asset", "equity", "bond", "forex", "option",
        "future", "dividend", "volatility", "risk",
    ],
    "vision": [
        "image", "picture", "photo", "visual", "detect", "recognize",
        "see", "look at", "video", "frame", "object detection",
        "segmentation", "ocr", "face", "scene",
    ],
    "voice": [
        "speech", "audio", "voice", "record", "transcribe", "speak",
        "listen", "sound", "noise", "microphone", "speaker",
        "text to speech", "speech to text",
    ],
    "planner": [
        "plan", "decompose", "organize", "break down", "strategy",
        "steps", "workflow", "pipeline", "schedule", "orchestrate",
    ],
}


def _classify_agent_type(task_description: str) -> str:
    """Keyword-score a task description against known agent types."""
    lower = task_description.lower()
    best_type = "research"
    best_score = 0
    for agent_type, keywords in _AGENT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_score:
            best_score = score
            best_type = agent_type
    return best_type


# ---------------------------------------------------------------------------
# Tool / Memory / RAG detection
# ---------------------------------------------------------------------------

_TOOL_KEYWORDS: dict[str, list[str]] = {
    "calculator": ["calculate", "compute", "math", "sum", "equation", "formula"],
    "web_search": ["search web", "look up", "find online", "internet", "google"],
    "python_executor": ["run code", "execute python", "script", "program"],
    "terminal": ["shell", "command", "terminal", "bash", "execute"],
    "filesystem": ["file", "read file", "write file", "directory", "save"],
    "weather": ["weather", "temperature", "forecast"],
    "datetime": ["date", "time", "timezone", "current time"],
}


def _detect_tools(task_description: str) -> list[ToolRequirement]:
    """Detect likely tool requirements from task description."""
    lower = task_description.lower()
    tools: list[ToolRequirement] = []
    for tool_name, keywords in _TOOL_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            tools.append(ToolRequirement(tool_name=tool_name, purpose=f"Auto-detected for: {task_description[:60]}"))
    return tools


_MEMORY_KEYWORDS = [
    "remember", "recall", "previous", "history", "context",
    "conversation", "memory", "what did i", "last time",
]

_RAG_KEYWORDS = [
    "document", "knowledge base", "corpus", "rag", "search document",
    "find in", "look up in", "retrieve from", "vector search",
]


def _detect_memory(task_description: str) -> bool:
    return any(kw in task_description.lower() for kw in _MEMORY_KEYWORDS)


def _detect_rag(task_description: str) -> bool:
    return any(kw in task_description.lower() for kw in _RAG_KEYWORDS)


# ---------------------------------------------------------------------------
# Dependency graph utilities
# ---------------------------------------------------------------------------


def _build_dependency_graph(tasks: list[PlannedTask]) -> dict[str, list[str]]:
    """Build adjacency list of task dependencies: task_id -> list of dependents."""
    graph: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        for dep_id in task.depends_on:
            graph[dep_id].append(task.id)
    return dict(graph)


def _topological_sort(tasks: list[PlannedTask]) -> list[PlannedTask]:
    """Return tasks topologically sorted by their dependency DAG."""
    in_degree: dict[str, int] = {t.id: 0 for t in tasks}
    for t in tasks:
        for dep_id in t.depends_on:
            in_degree[t.id] = in_degree.get(t.id, 0) + 1

    queue = deque([t for t in tasks if in_degree.get(t.id, 0) == 0])
    sorted_tasks: list[PlannedTask] = []
    task_map = {t.id: t for t in tasks}

    while queue:
        task = queue.popleft()
        sorted_tasks.append(task)
        for dependent_id in _build_dependency_graph(tasks).get(task.id, []):
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0 and dependent_id in task_map:
                queue.append(task_map[dependent_id])

    remaining = len(tasks) - len(sorted_tasks)
    if remaining:
        logger.warning("Dependency cycle detected; %d tasks excluded from sort", remaining)
    return sorted_tasks


def _find_critical_path(tasks: list[PlannedTask]) -> list[str]:
    """
    Find the critical path through the task DAG using longest-path on a DAG.

    Returns task IDs on the critical path (longest estimated duration chain).
    """
    task_map = {t.id: t for t in tasks}
    graph = _build_dependency_graph(tasks)
    topo = _topological_sort(tasks)

    dist: dict[str, float] = {t.id: 0.0 for t in tasks}
    predecessor: dict[str, str | None] = {t.id: None for t in tasks}

    for t in topo:
        for dep_id in graph.get(t.id, []):
            candidate = dist[t.id] + task_map[dep_id].estimated_duration_seconds
            if candidate > dist[dep_id]:
                dist[dep_id] = candidate
                predecessor[dep_id] = t.id

    end_task = max(tasks, key=lambda t: dist[t.id])
    path: list[str] = []
    current: str | None = end_task.id
    while current is not None:
        path.append(current)
        current = predecessor[current]
    path.reverse()
    return path


def _find_parallel_groups(tasks: list[PlannedTask]) -> list[list[str]]:
    """
    Group tasks that can run concurrently.

    Two tasks can run in parallel if neither depends on the other and they
    share no transitive dependency chain.
    """
    task_map = {t.id: t for t in tasks}
    deps_of: dict[str, set[str]] = {}
    for t in tasks:
        visited: set[str] = set()
        queue = deque(t.depends_on)
        while queue:
            d = queue.popleft()
            if d in visited:
                continue
            visited.add(d)
            queue.extend(task_map[d].depends_on if d in task_map else [])
        deps_of[t.id] = visited

    used: set[str] = set()
    groups: list[list[str]] = []
    remaining = [t.id for t in tasks]

    iteration = 0
    while remaining and iteration < len(tasks):
        iteration += 1
        ready = [tid for tid in remaining if deps_of[tid].issubset(used)]
        if not ready:
            break
        groups.append(ready)
        used.update(ready)
        remaining = [t for t in remaining if t not in ready]

    return groups


# ---------------------------------------------------------------------------
# Plan validator
# ---------------------------------------------------------------------------


def _validate_plan(plan: ExecutionPlan) -> list[str]:
    """Validate a plan and return a list of issues (empty = valid)."""
    issues: list[str] = []
    task_ids = {t.id for t in plan.tasks}

    for t in plan.tasks:
        for dep_id in t.depends_on:
            if dep_id not in task_ids:
                issues.append(f"Task '{t.id}' depends on unknown task '{dep_id}'")
        if not t.description:
            issues.append(f"Task '{t.id}' has empty description")
        if not t.agent_type:
            issues.append(f"Task '{t.id}' has no agent type")

    graph_ids = set()
    for group in plan.parallel_groups:
        graph_ids.update(group)
    for t in plan.tasks:
        if t.id not in graph_ids:
            issues.append(f"Task '{t.id}' not assigned to any parallel group")

    return issues


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------


class PlannerAgent(BaseAgent):
    """
    Agent that decomposes complex goals into structured, executable plans.

    Capabilities:

    * **Task decomposition** - break goals into discrete, ordered tasks
    * **Dependency estimation** - infer which tasks block others
    * **Prioritization** - assign priority based on critical-path analysis
    * **Parallel-group detection** - identify concurrent execution opportunities
    * **Tool detection** - flag tasks that need calculator, search, filesystem, etc.
    * **Memory detection** - flag tasks that should retrieve conversation history
    * **RAG detection** - flag tasks that should search indexed documents
    * **Re-planning** - adjust a plan when a task fails
    * **Validation** - check plan integrity (no cycles, missing deps, etc.)
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        planner_config: PlannerConfig | None = None,
    ) -> None:
        if config is None:
            config = AgentConfig(name="planner", system_prompt=self._default_system_prompt())
        super().__init__(config)
        self._planner_config = planner_config or PlannerConfig()
        self._plan_history: list[ExecutionPlan] = []

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    @staticmethod
    def _default_system_prompt() -> str:
        return (
            "You are a planning agent that decomposes complex goals into "
            "executable task plans. For each goal, produce a structured plan "
            "with ordered, dependency-aware tasks. Identify which tasks can "
            "run in parallel, what agent type each task needs, and what tools "
            "or data sources (memory, RAG) each task requires."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Decompose a goal into a structured ExecutionPlan.

        Args:
            task: The high-level goal or task to decompose.
            context: Optional context (may include prior plans, agent registry
                listing, or user preferences).

        Returns:
            AgentResult with ``output`` containing the JSON-serialized plan
            and ``metadata["plan"]`` holding the :class:`ExecutionPlan` object.
        """
        start = time.perf_counter()
        self.status = self.status
        ctx = context or {}

        try:
            plan = await self._build_plan(task, ctx)
            self._plan_history.append(plan)
            plan_json = json.dumps(plan.to_dict(), indent=2)
            duration = (time.perf_counter() - start) * 1000

            logger.info(
                "Plan created: goal=%s tasks=%d parallel_groups=%d critical_path_len=%d estimated=%.1fs",
                task[:60], len(plan.tasks), len(plan.parallel_groups),
                len(plan.critical_path), plan.estimated_total_duration_seconds,
            )

            return AgentResult(
                success=True,
                output=plan_json,
                agent_name=self.name,
                duration_ms=duration,
                tokens_used=sum(len(t.description.split()) for t in plan.tasks),
                metadata={
                    "plan": plan,
                    "plan_dict": plan.to_dict(),
                    "task_count": len(plan.tasks),
                    "parallel_groups": len(plan.parallel_groups),
                    "critical_path_length": len(plan.critical_path),
                    "estimated_duration_seconds": plan.estimated_total_duration_seconds,
                },
            )
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            logger.error("Plan creation failed: %s", exc)
            return AgentResult(
                success=False,
                output="",
                agent_name=self.name,
                duration_ms=duration,
                error=str(exc),
            )

    async def replan(
        self,
        previous_plan: ExecutionPlan,
        failed_task_id: str,
        error: str = "",
        context: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        """
        Generate a revised plan after a task failure.

        The failed task and any tasks that depend on it are re-decomposed
        or re-routed to different agent types. The plan version is
        incremented.

        Args:
            previous_plan: The plan that is being revised.
            failed_task_id: ID of the task that failed.
            error: Error message from the failure.
            context: Additional context for re-planning.

        Returns:
            A new :class:`ExecutionPlan` with the revision applied.
        """
        logger.info(
            "Re-planning after failure: task=%s error=%s version=%d",
            failed_task_id, error[:80], previous_plan.version,
        )

        task_map = {t.id: t for t in previous_plan.tasks}
        failed_task = task_map.get(failed_task_id)
        if failed_task is None:
            raise ValueError(f"Task '{failed_task_id}' not found in previous plan")

        # Identify tasks to re-plan: the failed task + its dependents
        affected = self._collect_dependents(previous_plan, failed_task_id)
        surviving = [t for t in previous_plan.tasks if t.id not in affected]

        # Re-decompose the affected goal fragment
        failed_description = failed_task.description
        if error:
            failed_description = f"{failed_description} (previous error: {error})"

        new_ctx = dict(context or {})
        new_ctx["_replan_reason"] = error
        new_ctx["_failed_agent"] = failed_task.agent_type
        new_plan = await self._build_plan(failed_description, new_ctx)

        # Re-merge: surviving tasks + new tasks from the re-plan
        merged_tasks = list(surviving) + list(new_plan.tasks)
        merged_tasks = self._renumber_ids(merged_tasks)

        result = self._assemble_plan(previous_plan.goal, merged_tasks)
        result.version = previous_plan.version + 1
        result.metadata["replanned_from"] = previous_plan.version
        result.metadata["failed_task_id"] = failed_task_id
        result.metadata["failure_reason"] = error

        self._plan_history.append(result)
        logger.info(
            "Re-plan complete: tasks=%d (survived=%d, new=%d) version=%d",
            len(result.tasks), len(surviving), len(new_plan.tasks), result.version,
        )
        return result

    def get_plan_history(self) -> list[ExecutionPlan]:
        """Return all plans generated by this agent."""
        return list(self._plan_history)

    # ------------------------------------------------------------------
    # Plan construction
    # ------------------------------------------------------------------

    async def _build_plan(
        self,
        goal: str,
        context: dict[str, Any],
    ) -> ExecutionPlan:
        """Full plan construction pipeline."""
        raw_tasks = self._decompose(goal, context)
        typed_tasks = self._assign_agent_types(raw_tasks)
        detected_tasks = self._detect_requirements(typed_tasks)
        dependency_tasks = self._estimate_dependencies(detected_tasks, goal)
        prioritized_tasks = self._assign_priorities(dependency_tasks)
        plan = self._assemble_plan(goal, prioritized_tasks)
        return plan

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def _decompose(
        self,
        goal: str,
        context: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Break a goal into raw task descriptions.

        Uses sentence-boundary splitting with context-aware merging of
        short fragments. Returns a list of ``{"description": str}`` dicts.
        """
        cfg = self._planner_config
        context_str = json.dumps(context, default=str) if context else ""

        raw_descriptions: list[str] = []

        if cfg.use_llm_decomposition:
            raw_descriptions = self._llm_decompose(goal, context_str)
        else:
            raw_descriptions = self._rule_decompose(goal)

        # Merge descriptions that are too short to be standalone tasks
        merged: list[str] = []
        buffer = ""
        for desc in raw_descriptions:
            desc = desc.strip()
            if not desc:
                continue
            if not buffer:
                buffer = desc
            elif len(desc) < cfg.min_task_length:
                buffer += ". " + desc
            else:
                if len(buffer) >= cfg.min_task_length:
                    merged.append(buffer)
                else:
                    merged.append(buffer + ". " + desc) if merged else merged.append(buffer + ". " + desc)
                buffer = desc
        if buffer and len(buffer) >= cfg.min_task_length:
            merged.append(buffer)
        elif buffer and merged:
            merged[-1] += ". " + buffer
        elif buffer:
            merged.append(buffer)

        # Respect max_tasks
        if len(merged) > cfg.max_tasks:
            merged = merged[:cfg.max_tasks]

        return [{"description": d} for d in merged]

    def _rule_decompose(self, goal: str) -> list[str]:
        """Rule-based decomposition: split on sentence boundaries and conjunctions."""
        cleaned = goal.replace("?", ".").replace("!", ".")
        for sep in [" and then ", " then ", " subsequently "]:
            cleaned = cleaned.replace(sep, ". ")

        sentences = [s.strip() for s in cleaned.split(".") if s.strip() and len(s.strip()) > 5]

        # If only one sentence, try harder to split
        if len(sentences) <= 1:
            for sep in [", then ", ", followed by ", ", after which "]:
                if sep in cleaned:
                    parts = [p.strip() for p in cleaned.split(sep) if p.strip()]
                    if len(parts) > 1:
                        return parts

        return sentences if sentences else [goal]

    def _llm_decompose(self, goal: str, context_str: str) -> list[str]:
        """LLM-based decomposition (requires external LLM integration)."""
        logger.info("LLM decomposition requested but not available; falling back to rule-based")
        return self._rule_decompose(goal)

    # ------------------------------------------------------------------
    # Agent type assignment
    # ------------------------------------------------------------------

    def _assign_agent_types(
        self,
        raw_tasks: list[dict[str, Any]],
    ) -> list[PlannedTask]:
        """Classify each raw task to the best-matching agent type."""
        cfg = self._planner_config
        tasks: list[PlannedTask] = []
        for i, rt in enumerate(raw_tasks):
            desc = rt["description"]
            agent_type = _classify_agent_type(desc)
            tasks.append(PlannedTask(
                id=f"task_{i + 1}",
                description=desc,
                agent_type=agent_type if agent_type else cfg.default_agent_type,
                estimated_duration_seconds=cfg.duration_per_task_seconds,
            ))
        return tasks

    # ------------------------------------------------------------------
    # Tool / Memory / RAG detection
    # ------------------------------------------------------------------

    def _detect_requirements(
        self,
        tasks: list[PlannedTask],
    ) -> list[PlannedTask]:
        """Augment tasks with detected tool, memory, and RAG requirements."""
        cfg = self._planner_config
        for task in tasks:
            if cfg.enable_tool_detection:
                task.requires_tools = _detect_tools(task.description)
            if cfg.enable_memory_detection:
                task.requires_memory = _detect_memory(task.description)
            if cfg.enable_rag_detection:
                task.requires_rag = _detect_rag(task.description)
        return tasks

    # ------------------------------------------------------------------
    # Dependency estimation
    # ------------------------------------------------------------------

    def _estimate_dependencies(
        self,
        tasks: list[PlannedTask],
        goal: str,
    ) -> list[PlannedTask]:
        """
        Infer which tasks depend on which.

        Heuristics:

        * A task whose description references a prior task number or
          pronoun likely depends on that task.
        * A task that starts with "then", "next", "after", "finally" is
          sequential to the previous task.
        * RAG retrieval tasks are front-loaded with no dependencies.
        * Tool-execution tasks are leaf nodes (depend on prior analysis).
        """
        for i, task in enumerate(tasks):
            lower = task.description.lower()

            # RAG / memory tasks go first
            if task.requires_rag or task.requires_memory:
                continue

            # "then / next / after / finally" -> sequential
            if any(lower.startswith(w) for w in ["then ", "next ", "after ", "finally "]):
                if i > 0:
                    task.depends_on.append(tasks[i - 1].id)
                continue

            # Tool tasks depend on analysis tasks that came before
            if task.requires_tools:
                for j in range(i - 1, -1, -1):
                    if not tasks[j].requires_tools:
                        task.depends_on.append(tasks[j].id)
                        break
                continue

            # Reference to prior step numbers
            for j in range(i):
                prior_label = f"task {j + 1}"
                if prior_label in lower:
                    task.depends_on.append(tasks[j].id)

            # If no dependencies found, depend on immediate predecessor
            if not task.depends_on and i > 0:
                task.depends_on.append(tasks[i - 1].id)

        return tasks

    # ------------------------------------------------------------------
    # Prioritization
    # ------------------------------------------------------------------

    def _assign_priorities(self, tasks: list[PlannedTask]) -> list[PlannedTask]:
        """
        Assign priority based on critical-path position and dependency weight.

        * Tasks on the critical path get HIGH or CRITICAL priority.
        * Tasks with many dependents get HIGH priority.
        * Leaf-node, short-duration tasks get LOW priority.
        * Everything else is MEDIUM.
        """
        if not tasks:
            return tasks

        task_map = {t.id: t for t in tasks}
        try:
            critical_path = _find_critical_path(tasks)
        except Exception:
            critical_path = [t.id for t in tasks]

        critical_set = set(critical_path)
        dependents_count: dict[str, int] = defaultdict(int)
        for t in tasks:
            for dep_id in t.depends_on:
                dependents_count[dep_id] += 1

        for t in tasks:
            if t.id in critical_set and len(critical_path) > 1:
                # Core of the critical path
                if critical_path.index(t.id) < len(critical_path) // 2:
                    t.priority = TaskPriority.CRITICAL
                else:
                    t.priority = TaskPriority.HIGH
            elif dependents_count.get(t.id, 0) >= 3:
                t.priority = TaskPriority.HIGH
            elif dependents_count.get(t.id, 0) == 0 and not t.depends_on:
                t.priority = TaskPriority.LOW
            else:
                t.priority = TaskPriority.MEDIUM

        return tasks

    # ------------------------------------------------------------------
    # Plan assembly
    # ------------------------------------------------------------------

    def _assemble_plan(
        self,
        goal: str,
        tasks: list[PlannedTask],
    ) -> ExecutionPlan:
        """Construct a full ExecutionPlan from a list of tasks."""
        critical_path: list[str] = []
        parallel_groups: list[list[str]] = []

        try:
            topo = _topological_sort(tasks)
            ordered_ids = [t.id for t in topo]
            task_map = {t.id: t for t in tasks}
            ordered_tasks = [task_map[tid] for tid in ordered_ids if tid in task_map]
            critical_path = _find_critical_path(ordered_tasks)
            parallel_groups = _find_parallel_groups(ordered_tasks)
        except Exception as exc:
            logger.warning("Plan assembly heuristics failed: %s", exc)
            ordered_tasks = list(tasks)
            critical_path = [t.id for t in ordered_tasks]
            parallel_groups = [[t.id] for t in ordered_tasks]

        critical_duration = sum(
            task_map.get(tid, tasks[0]).estimated_duration_seconds
            for tid in critical_path if tid in task_map
        )

        plan = ExecutionPlan(
            goal=goal,
            tasks=ordered_tasks,
            parallel_groups=parallel_groups,
            critical_path=critical_path,
            estimated_total_duration_seconds=critical_duration,
            created_at=time.time(),
            version=1,
            metadata={},
        )

        issues = _validate_plan(plan)
        if issues:
            plan.metadata["validation_issues"] = issues
            for issue in issues:
                logger.warning("Plan validation: %s", issue)

        return plan

    # ------------------------------------------------------------------
    # Re-plan helpers
    # ------------------------------------------------------------------

    def _collect_dependents(
        self,
        plan: ExecutionPlan,
        task_id: str,
    ) -> set[str]:
        """Collect the ID of a failed task plus all transitive dependents."""
        task_map = {t.id: t for t in plan.tasks}
        reverse_deps: dict[str, list[str]] = defaultdict(list)
        for t in plan.tasks:
            for dep_id in t.depends_on:
                reverse_deps[dep_id].append(t.id)

        affected: set[str] = set()
        queue = deque([task_id])
        while queue:
            tid = queue.popleft()
            if tid in affected:
                continue
            affected.add(tid)
            queue.extend(reverse_deps.get(tid, []))
        return affected

    @staticmethod
    def _renumber_ids(tasks: list[PlannedTask]) -> list[PlannedTask]:
        """Re-assign sequential IDs and update dependency references."""
        old_to_new: dict[str, str] = {}
        for i, t in enumerate(tasks):
            old_id = t.id
            new_id = f"task_{i + 1}"
            old_to_new[old_id] = new_id
            t.id = new_id
        for t in tasks:
            t.depends_on = [old_to_new.get(d, d) for d in t.depends_on if old_to_new.get(d, d)]
        return tasks
