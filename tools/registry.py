"""
tools/registry.py
-----------------
Tool registry and base interface for RIME.

Every tool registered here exposes a standard interface so the Intent Router
and Response Manager can treat all tools uniformly — regardless of whether
they are local or network-dependent.

Architecture notes (TRD §3.7, Implementation Plan Phase 5):
  • Standard interface: name, description, parameters, permissions,
    network_required, cancellable, timeout_seconds, execute()
  • Permission levels (routing/permissions.py):
      0 — read local       (calculator, time, memory lookup)
      1 — create local     (task creation, memory write)
      2 — modify/delete    (task delete, memory delete)
      3 — external access  (weather, transport)
      4 — external action  (future — reserved)
  • All tools are registered at startup; the registry is the single
    source of truth for GET /api/v1/tools.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default tool timeout — tool calls exceeding this are marked timed_out
DEFAULT_TIMEOUT_SECONDS: float = 15.0


@dataclass
class ToolResult:
    success: bool
    output: Any                           # structured result forwarded to the agent
    tool_name: str
    generation: int
    latency_ms: float = 0.0
    from_cache: bool = False
    error: Optional[str] = None
    network_used: bool = False


class BaseTool(ABC):
    """
    Abstract base class for all RIME tools.

    Subclasses must implement `execute()`.  Everything else (timeout
    wrapping, generation tagging, logging) is handled by the registry.
    """

    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}   # JSON Schema–style parameter spec
    permission_level: int = 0
    network_required: bool = False
    api_key_required: bool = False
    cancellable: bool = True
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    enabled: bool = True

    @abstractmethod
    async def execute(
        self,
        args: dict[str, Any],
        generation: int,
        profile_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute the tool with the given arguments.
        Must return a ToolResult — never raise an exception to the caller.
        """


class ToolRegistry:
    """
    Central registry for all RIME tools.

    Usage:
        registry = ToolRegistry()
        registry.register(CalculatorTool())
        result = await registry.run("calculator", {"expression": "42*12"}, gen=5)
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.debug("Tool registered: %s (network=%s)", tool.name, tool.network_required)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_all(self) -> list[dict[str, Any]]:
        """Serialise all registered tools for GET /api/v1/tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "network_required": t.network_required,
                "api_key_required": t.api_key_required,
                "cancellable": t.cancellable,
                "permission_level": t.permission_level,
                "enabled": t.enabled,
                "timeout_seconds": t.timeout_seconds,
            }
            for t in self._tools.values()
        ]

    def set_enabled(self, name: str, enabled: bool) -> bool:
        tool = self._tools.get(name)
        if tool is None:
            return False
        tool.enabled = enabled
        return True

    async def run(
        self,
        name: str,
        args: dict[str, Any],
        generation: int,
        profile_id: Optional[str] = None,
    ) -> ToolResult:
        """
        Execute the named tool with timeout wrapping and structured logging.
        Returns a ToolResult — never raises.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                tool_name=name,
                generation=generation,
                error=f"Tool '{name}' is not registered.",
            )
        if not tool.enabled:
            return ToolResult(
                success=False,
                output=None,
                tool_name=name,
                generation=generation,
                error=f"Tool '{name}' is disabled.",
            )

        t_start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                tool.execute(args, generation, profile_id),
                timeout=tool.timeout_seconds,
            )
            result.latency_ms = (time.monotonic() - t_start) * 1000
            logger.info(
                "Tool '%s' completed in %.1f ms (gen=%d, success=%s, cached=%s)",
                name,
                result.latency_ms,
                generation,
                result.success,
                result.from_cache,
            )
            return result
        except asyncio.TimeoutError:
            latency_ms = (time.monotonic() - t_start) * 1000
            logger.warning(
                "Tool '%s' timed out after %.1f ms (gen=%d).",
                name, latency_ms, generation,
            )
            return ToolResult(
                success=False,
                output=None,
                tool_name=name,
                generation=generation,
                latency_ms=latency_ms,
                error=f"Tool '{name}' timed out after {tool.timeout_seconds}s.",
            )
        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.monotonic() - t_start) * 1000
            logger.error("Tool '%s' error: %s", name, exc)
            return ToolResult(
                success=False,
                output=None,
                tool_name=name,
                generation=generation,
                latency_ms=latency_ms,
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# Singleton registry built at startup — tools register themselves here.
# ---------------------------------------------------------------------------
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
