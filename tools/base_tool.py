from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolInput:
    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutput:
    success: bool
    data: Any = None
    error: str = ""
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    def __init__(self) -> None:
        self._logger = logging.getLogger(f"aios.tool.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    def description(self) -> str:
        return ""

    @property
    def parameters(self) -> dict[str, Any]:
        return {}

    @abstractmethod
    async def execute(self, inp: ToolInput) -> ToolOutput:
        ...

    async def validate(self, inp: ToolInput) -> str | None:
        return None

    @property
    def requires_permission(self) -> bool:
        return False
