from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from agents.base_agent import AgentResult
from agents.coding_agent import CodingAgent
from agents.cybersecurity_agent import CybersecurityAgent
from agents.math_agent import MathAgent
from agents.planner_agent import PlannerAgent
from agents.research_agent import ResearchAgent
from agents.router import AgentRouter
from agents.trading_agent import TradingAgent
from agents.vision_agent import VisionAgent
from agents.voice_agent import VoiceAgent

logger = logging.getLogger("aios.agent.manager")


class AgentManager:
    def __init__(self) -> None:
        self._router = AgentRouter()
        self._sessions: dict[str, list[AgentResult]] = {}
        self._logger = logging.getLogger("aios.agent.manager")
        self._register_default_agents()

    def _register_default_agents(self) -> None:
        self._router.register("planner", PlannerAgent())
        self._router.register("coding", CodingAgent())
        self._router.register("research", ResearchAgent(), is_default=True)
        self._router.register("math", MathAgent())
        self._router.register("cybersecurity", CybersecurityAgent())
        self._router.register("trading", TradingAgent())
        self._router.register("vision", VisionAgent())
        self._router.register("voice", VoiceAgent())
        self._logger.info("Registered %s default agents", len(self._router.available_agents))

    def register_agent(self, name: str, agent: Any, is_default: bool = False) -> None:
        self._router.register(name, agent, is_default)

    async def execute(self, task: str, agent_name: str | None = None, context: dict[str, Any] | None = None) -> AgentResult:
        agent = self._router.route_to(task, agent_name) if agent_name else self._router.route(task)
        result = await agent.execute(task, context)
        session_id = agent.session_id
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(result)
        return result

    async def execute_plan(self, task: str, context: dict[str, Any] | None = None) -> list[AgentResult]:
        planner = self._router.route_to(task, "planner")
        plan_result = await planner.execute(task, context)
        if not plan_result.success:
            return [plan_result]

        import json
        plan = json.loads(plan_result.output)
        steps = plan.get("steps", [])
        parallel_groups = plan.get("parallel_possible", [])

        results: list[AgentResult] = []
        for group in parallel_groups:
            tasks_in_group = []
            for step_num in group:
                step_info = next((s for s in steps if s["step"] == step_num), None)
                if step_info:
                    tasks_in_group.append(
                        self.execute(step_info["input"], step_info["agent"], context)
                    )
            if len(tasks_in_group) > 1:
                group_results = await asyncio.gather(*tasks_in_group)
                results.extend(group_results)
            elif tasks_in_group:
                results.append(await tasks_in_group[0])

        for step in steps:
            if not any(r.metadata.get("step") == step["step"] for r in results if r.metadata):
                result = await self.execute(step["input"], step["agent"], context)
                result.metadata["step"] = step["step"]
                results.append(result)

        return results

    async def execute_parallel(
        self, tasks: list[tuple[str, str | None]], context: dict[str, Any] | None = None
    ) -> list[AgentResult]:
        coros = [self.execute(task, agent_name, context) for task, agent_name in tasks]
        return await asyncio.gather(*coros)

    def get_session_history(self, session_id: str) -> list[AgentResult]:
        return self._sessions.get(session_id, [])

    @property
    def available_agents(self) -> list[str]:
        return self._router.available_agents
