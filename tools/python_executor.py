from __future__ import annotations

import io
import sys
import textwrap
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

from tools.base_tool import BaseTool, ToolInput, ToolOutput


class PythonExecutorTool(BaseTool):
    @property
    def name(self) -> str:
        return "python_executor"

    @property
    def description(self) -> str:
        return "Execute Python code in a restricted sandbox"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds",
                },
            },
            "required": ["code"],
        }

    @property
    def requires_permission(self) -> bool:
        return True

    async def execute(self, inp: ToolInput) -> ToolOutput:
        code = inp.arguments.get("code", "")
        if not code:
            return ToolOutput(success=False, error="No code provided")

        if any(blocked in code for blocked in ["__import__", "exec(", "eval(", "open(", "__builtins__"]):
            return ToolOutput(success=False, error="Blocked operation detected")

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            compiled = compile(textwrap.dedent(code), "<sandbox>", "exec")
            restricted_globals: dict[str, Any] = {
                "__builtins__": {
                    "abs": abs, "all": all, "any": any, "bool": bool,
                    "chr": chr, "dict": dict, "divmod": divmod, "enumerate": enumerate,
                    "filter": filter, "float": float, "format": format, "frozenset": frozenset,
                    "hash": hash, "hex": hex, "id": id, "int": int, "isinstance": isinstance,
                    "issubclass": issubclass, "iter": iter, "len": len, "list": list,
                    "map": map, "max": max, "min": min, "next": next, "object": object,
                    "oct": oct, "ord": ord, "pow": pow, "print": print, "range": range,
                    "repr": repr, "reversed": reversed, "round": round, "set": set,
                    "slice": slice, "sorted": sorted, "str": str, "sum": sum,
                    "tuple": tuple, "type": type, "zip": zip,
                },
            }

            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(compiled, restricted_globals)

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            return ToolOutput(
                success=not stderr,
                data={
                    "stdout": stdout,
                    "stderr": stderr,
                    "return_value": restricted_globals.get("result", None),
                },
            )
        except Exception:
            return ToolOutput(
                success=False,
                error=traceback.format_exc(),
            )
