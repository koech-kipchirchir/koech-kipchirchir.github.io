from __future__ import annotations

import ast
import logging
import math
import operator
import time
from typing import Any, Optional

from agents.base_agent import AgentConfig, AgentResult, BaseAgent

logger = logging.getLogger("aios.agent.math")


class MathAgent(BaseAgent):
    def __init__(self, config: AgentConfig | None = None) -> None:
        super().__init__(config or AgentConfig(
            name="math",
            system_prompt=(
                "You are a mathematical computation agent. Evaluate expressions, "
                "solve equations, perform symbolic reasoning, and explain mathematical concepts."
            ),
        ))

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        start = time.perf_counter()
        try:
            expression, explanation = self._parse_task(task)
            result = self._safe_eval(expression) if expression else None
            output = self._format_output(task, expression, result, explanation)
            duration = (time.perf_counter() - start) * 1000
            return AgentResult(
                success=True, output=output, agent_name=self.name,
                duration_ms=duration,
                metadata={"expression": expression, "result": result},
            )
        except Exception as exc:
            duration = (time.perf_counter() - start) * 1000
            return AgentResult(
                success=False, output="", agent_name=self.name,
                duration_ms=duration, error=str(exc),
            )

    def _safe_eval(self, expression: str) -> float:
        tree = ast.parse(expression, mode="eval")
        return self._eval_node(tree.body)

    _OPERATORS: dict[type, Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value) if isinstance(node.value, (int, float)) else float(node.value)
        if isinstance(node, ast.UnaryOp):
            return self._OPERATORS[type(node.op)](self._eval_node(node.operand))
        if isinstance(node, ast.BinOp):
            return self._OPERATORS[type(node.op)](self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    def _eval_call(self, node: ast.Call) -> float:
        safe_funcs: dict[str, Any] = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow, "sqrt": math.sqrt,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "ceil": math.ceil, "floor": math.floor, "pi": lambda: math.pi,
            "e": lambda: math.e,
        }
        func_name = node.func.id if isinstance(node.func, ast.Name) else ""
        func = safe_funcs.get(func_name)
        if func is None:
            raise ValueError(f"Function not allowed: {func_name}")
        args = [self._eval_node(a) for a in node.args]
        return func(*args)

    @staticmethod
    def _parse_task(task: str) -> tuple[str, str]:
        tl = task.lower()
        if "=" in task and "?" not in task:
            parts = task.split("=")
            if len(parts) == 2:
                expr = parts[0].strip()
                expected = parts[1].strip()
                return expr, f"Verify: {expr} = {expected}"
        if "sqrt" in tl or "square root" in tl:
            import re
            nums = re.findall(r"\d+\.?\d*", task)
            if nums:
                return f"sqrt({nums[0]})", f"Calculate square root of {nums[0]}"
        if any(op in task for op in ["+", "-", "*", "/", "**", "%"]):
            return task.strip(), f"Evaluate expression: {task}"
        return task, "General mathematical question"

    @staticmethod
    def _format_output(task: str, expr: str | None, result: float | None, explanation: str) -> str:
        lines = [f"## Math Result\n", f"**Task:** {task}\n"]
        if expr:
            lines.append(f"**Expression:** `{expr}`\n")
        if result is not None:
            lines.append(f"**Result:** `{result}`\n")
        lines.append(f"\n_{explanation}_\n")
        return "".join(lines)
