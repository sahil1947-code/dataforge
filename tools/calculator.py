"""
tools/calculator.py
-------------------
Local arithmetic tool — no network, no credentials, permission level 0.

Uses Python's ast.literal_eval + a safe expression evaluator so we never
call eval() on raw user input.  Only basic arithmetic operators are allowed.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from typing import Any, Optional

from .registry import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_SAFE_OPS = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Pow:  operator.pow,
    ast.Mod:  operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES = {
    "sqrt": math.sqrt,
    "abs":  abs,
    "round": round,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator")
        return op(_safe_eval(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_NAMES:
            fn = _SAFE_NAMES[node.func.id]
            args = [_safe_eval(a) for a in node.args]
            return fn(*args)
        raise ValueError(f"Function not allowed: {ast.dump(node.func)}")
    if isinstance(node, ast.Name) and node.id in _SAFE_NAMES:
        val = _SAFE_NAMES[node.id]
        if callable(val):
            raise ValueError(f"{node.id} requires arguments")
        return float(val)
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Evaluates arithmetic expressions locally. No network required."
    parameters = {
        "expression": {
            "type": "string",
            "description": "A mathematical expression, e.g. '42 * 12' or 'sqrt(144)'",
        }
    }
    permission_level = 0
    network_required = False
    api_key_required = False
    cancellable = True
    timeout_seconds = 2.0

    async def execute(
        self,
        args: dict[str, Any],
        generation: int,
        profile_id: Optional[str] = None,
    ) -> ToolResult:
        expression = str(args.get("expression", "")).strip()
        if not expression:
            return ToolResult(
                success=False,
                output=None,
                tool_name=self.name,
                generation=generation,
                error="No expression provided.",
            )
        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            # Format cleanly: integer if no fractional part
            formatted = int(result) if result == int(result) else round(result, 10)
            return ToolResult(
                success=True,
                output={"expression": expression, "result": formatted},
                tool_name=self.name,
                generation=generation,
            )
        except Exception as exc:
            return ToolResult(
                success=False,
                output=None,
                tool_name=self.name,
                generation=generation,
                error=f"Could not evaluate '{expression}': {exc}",
            )
