from __future__ import annotations

import asyncio
import shlex
from typing import Any

from tools.base_tool import BaseTool, ToolInput, ToolOutput


class TerminalTool(BaseTool):
    def __init__(self, allowed_commands: list[str] | None = None, blocked_commands: list[str] | None = None) -> None:
        super().__init__()
        self._allowed = allowed_commands or ["ls", "cat", "pwd", "echo", "head", "tail", "wc", "whoami", "date", "uname", "pip", "python"]
        self._blocked = blocked_commands or ["rm", "sudo", "chmod", "dd", "mkfs", "shutdown", "reboot", "kill", "pkill"]

    @property
    def name(self) -> str:
        return "terminal"

    @property
    def description(self) -> str:
        return "Execute shell commands with security restrictions"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
            "required": ["command"],
        }

    @property
    def requires_permission(self) -> bool:
        return True

    async def execute(self, inp: ToolInput) -> ToolOutput:
        command = inp.arguments.get("command", "")
        if not command:
            return ToolOutput(success=False, error="No command provided")

        cmd_parts = shlex.split(command)
        if not cmd_parts:
            return ToolOutput(success=False, error="Empty command")

        base_cmd = cmd_parts[0]
        if any(blocked in base_cmd for blocked in self._blocked):
            return ToolOutput(success=False, error=f"Command blocked: {base_cmd}")

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=30
                )
            except asyncio.TimeoutError:
                proc.kill()
                return ToolOutput(success=False, error="Command timed out after 30s")

            stdout_str = stdout.decode("utf-8", errors="replace") if stdout else ""
            stderr_str = stderr.decode("utf-8", errors="replace") if stderr else ""

            return ToolOutput(
                success=proc.returncode == 0,
                data={
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "return_code": proc.returncode,
                },
                error=stderr_str if proc.returncode != 0 else "",
            )
        except Exception as exc:
            return ToolOutput(success=False, error=str(exc))
