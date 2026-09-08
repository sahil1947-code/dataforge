"""
routing/local.py
----------------
Local resolution router — handles all tool calls that do not require
a network connection.

Responsible for:
  • Executing local tools (calculator, time, tasks, memory) via the registry
  • Checking network availability before escalating to external
  • Returning a clear NETWORK_UNAVAILABLE signal when live data is needed
    but the network is offline (never fabricates a result)
"""

from __future__ import annotations

import asyncio
import logging
import socket
from typing import Any, Optional

from tools.registry import get_registry, ToolResult

logger = logging.getLogger(__name__)

# Local tools that can always run without network
_LOCAL_TOOL_NAMES: frozenset[str] = frozenset(
    ["calculator", "time", "tasks", "memory"]
)


def _is_network_available(timeout: float = 1.5) -> bool:
    """
    Non-blocking network availability check.
    Attempts a TCP connection to a well-known host.
    Returns True if the internet appears reachable.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
            ("8.8.8.8", 53)
        )
        return True
    except OSError:
        return False


class LocalRouter:
    """
    Executes local tool calls and reports network availability for live-data
    escalation decisions.
    """

    def __init__(self) -> None:
        self._registry = get_registry()
        self._network_cache: Optional[tuple[bool, float]] = None

    async def execute_local(
        self,
        tool_name: str,
        args: dict[str, Any],
        generation: int,
        profile_id: Optional[str] = None,
    ) -> ToolResult:
        """Run a local tool.  Returns ToolResult (never raises)."""
        if tool_name not in _LOCAL_TOOL_NAMES:
            return ToolResult(
                success=False,
                output=None,
                tool_name=tool_name,
                generation=generation,
                error=f"'{tool_name}' is not a local tool.",
            )
        return await self._registry.run(tool_name, args, generation, profile_id)

    async def is_network_available(self) -> bool:
        """
        Check network availability with a short-lived cache (5 s) to avoid
        hammering the socket check on every request.
        """
        import time
        if self._network_cache is not None:
            available, ts = self._network_cache
            if time.monotonic() - ts < 5.0:
                return available

        loop = asyncio.get_running_loop()
        available = await loop.run_in_executor(None, _is_network_available)
        self._network_cache = (available, time.monotonic())
        return available

    def is_local_tool(self, tool_name: str) -> bool:
        return tool_name in _LOCAL_TOOL_NAMES
