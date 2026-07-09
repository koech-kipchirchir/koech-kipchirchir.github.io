from __future__ import annotations

import ast
import math
import operator
from typing import Any

from tools.base_tool import BaseTool, ToolInput, ToolOutput

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

_SAFE_FUNCS: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "pow": pow, "sqrt": math.sqrt,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "log": math.log, "log10": math.log10, "exp": math.exp,
    "ceil": math.ceil, "floor": math.floor,
    "pi": lambda: math.pi, "e": lambda: math.e,
}


class CalculatorTool(BaseTool):
    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return "Evaluate mathematical expressions safely"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to evaluate",
                }
            },
            "required": ["expression"],
        }

    async def execute(self, inp: ToolInput) -> ToolOutput:
        expression = inp.arguments.get("expression", "")
        if not expression:
            return ToolOutput(success=False, error="No expression provided")

        try:
            tree = ast.parse(expression.strip(), mode="eval")
            result = self._eval_node(tree.body)
            return ToolOutput(success=True, data={"expression": expression, "result": result})
        except Exception as exc:
            return ToolOutput(success=False, error=f"Evaluation error: {exc}")

    def _eval_node(self, node: ast.AST) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value) if isinstance(node.value, (int, float)) else float(node.value)  # type: ignore[arg-type]
        if isinstance(node, ast.UnaryOp):
            return _OPERATORS[type(node.op)](self._eval_node(node.operand))
        if isinstance(node, ast.BinOp):
            return _OPERATORS[type(node.op)](self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.Call):
            return self._eval_call(node)
        raise ValueError(f"Unsupported: {ast.dump(node)}")

    def _eval_call(self, node: ast.Call) -> float:
        func_name = node.func.id if isinstance(node.func, ast.Name) else ""
        func = _SAFE_FUNCS.get(func_name)
        if func is None:
            raise ValueError(f"Unknown function: {func_name}")
        args = [self._eval_node(a) for a in node.args]
        return func(*args)
