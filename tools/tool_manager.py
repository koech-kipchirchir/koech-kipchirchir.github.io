from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from tools.base_tool import BaseTool, ToolInput, ToolOutput
from tools.calculator import CalculatorTool
from tools.datetime_tool import DateTimeTool
from tools.filesystem import FileSystemTool
from tools.python_executor import PythonExecutorTool
from tools.terminal import TerminalTool
from tools.tool_registry import ToolRegistry
from tools.weather import WeatherTool
from tools.web_search import WebSearchTool

logger = logging.getLogger("aios.tool.manager")


class ToolManager:
    def __init__(self) -> None:
        self._registry = ToolRegistry()
        self._permission_cache: dict[str, bool] = {}
        self._logger = logging.getLogger("aios.tool.manager")
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        tools: list[BaseTool] = [
            CalculatorTool(),
            DateTimeTool(),
            FileSystemTool(),
            PythonExecutorTool(),
            TerminalTool(),
            WeatherTool(),
            WebSearchTool(),
        ]
        for tool in tools:
            self._registry.register(tool)
        self._logger.info("Registered %s default tools", len(tools))

    def register_tool(self, tool: BaseTool) -> None:
        self._registry.register(tool)

    def unregister_tool(self, name: str) -> None:
        self._registry.unregister(name)

    async def execute(
        self, tool_name: str, arguments: dict[str, Any], permission_override: bool | None = None
    ) -> ToolOutput:
        tool = self._registry.get(tool_name)
        if tool is None:
            return ToolOutput(success=False, error=f"Tool not found: {tool_name}")

        inp = ToolInput(arguments=arguments)

        if permission_override is not False:
            validation_error = await tool.validate(inp)
            if validation_error:
                return ToolOutput(success=False, error=validation_error)

        if tool.requires_permission and permission_override is not True:
            return ToolOutput(
                success=False,
                error=f"Permission required for tool: {tool_name}. "
                       f"Set permission_override=True to authorize.",
            )

        try:
            result = await asyncio.wait_for(
                tool.execute(inp),
                timeout=inp.metadata.get("timeout", 30),
            )
            return result
        except asyncio.TimeoutError:
            return ToolOutput(success=False, error=f"Tool '{tool_name}' timed out")
        except Exception as exc:
            return ToolOutput(success=False, error=str(exc))

    async def execute_batch(
        self, tasks: list[tuple[str, dict[str, Any]]]
    ) -> list[ToolOutput]:
        return await asyncio.gather(*[
            self.execute(name, args) for name, args in tasks
        ])

    def get_tool(self, name: str) -> BaseTool | None:
        return self._registry.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return self._registry.list_tools()

    @property
    def available_tools(self) -> list[str]:
        return self._registry.available_tools
