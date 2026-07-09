from __future__ import annotations

import logging
from typing import Any, Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger("aios.agent.router")


class AgentRouter:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}
        self._default_agent: str | None = None
        self._logger = logging.getLogger("aios.agent.router")

    def register(self, name: str, agent: BaseAgent, is_default: bool = False) -> None:
        self._agents[name] = agent
        if is_default:
            self._default_agent = name
        self._logger.info("Registered agent: %s (default=%s)", name, is_default)

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)
        if self._default_agent == name:
            self._default_agent = next(iter(self._agents.keys())) if self._agents else None

    def route(self, task: str) -> BaseAgent:
        task_lower = task.lower()
        priorities: list[tuple[str, int]] = []

        for name, agent in self._agents.items():
            score = self._score_agent(name, agent, task_lower)
            if score > 0:
                priorities.append((name, score))

        if not priorities:
            if self._default_agent and self._default_agent in self._agents:
                return self._agents[self._default_agent]
            raise ValueError(f"No agent found for task: {task[:50]}...")

        priorities.sort(key=lambda x: -x[1])
        best = self._agents[priorities[0][0]]
        self._logger.info("Routed task to agent: %s (score=%s)", best.name, priorities[0][1])
        return best

    def route_to(self, task: str, agent_name: str) -> BaseAgent:
        agent = self._agents.get(agent_name)
        if agent is None:
            raise ValueError(f"Agent not found: {agent_name}")
        return agent

    def get(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    @property
    def available_agents(self) -> list[str]:
        return list(self._agents.keys())

    @staticmethod
    def _score_agent(name: str, agent: BaseAgent, task_lower: str) -> int:
        keywords_map: dict[str, list[str]] = {
            "coding": ["code", "write", "implement", "program", "function", "script", "debug", "refactor"],
            "research": ["research", "search", "find", "look up", "what is", "who is", "explain", "tell me about"],
            "planner": ["plan", "decompose", "organize", "break down", "strategy", "steps"],
            "math": ["calculate", "compute", "math", "equation", "sum", "solve", "derivative", "integral"],
            "cybersecurity": ["security", "vulnerability", "threat", "attack", "exploit", "pentest", "firewall"],
            "trading": ["trade", "stock", "market", "invest", "portfolio", "crypto", "price"],
            "vision": ["image", "picture", "photo", "visual", "detect", "recognize", "see", "look at"],
            "voice": ["speech", "audio", "voice", "record", "transcribe", "speak", "listen"],
        }

        keywords = keywords_map.get(name, [])
        score = sum(1 for kw in keywords if kw in task_lower)
        return score
