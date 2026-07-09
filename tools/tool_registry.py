from __future__ import annotations

import logging
from typing import Any

from tools.base_tool import BaseTool

logger = logging.getLogger("aios.tool.registry")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._logger = logging.getLogger("aios.tool.registry")

    def register(self, tool: BaseTool) -> None:
        if tool.name in self._tools:
            self._logger.warning("Overwriting existing tool: %s", tool.name)
        self._tools[tool.name] = tool
        self._logger.info("Registered tool: %s", tool.name)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)
        self._logger.info("Unregistered tool: %s", name)

    def get(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "requires_permission": tool.requires_permission,
            }
            for tool in self._tools.values()
        ]

    @property
    def available_tools(self) -> list[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
