"""
tools/transport.py
------------------
Live transit / transport status tool.

Used for the "fixed-delay stress test" in the benchmark harness — this tool
supports an artificial delay injection for testing interruption correctness
(Implementation Plan Phase 2, benchmarks/interruption/).

Permission level 3 (external access).  Cached for 5 minutes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from .registry import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS: int = 300


class TransportTool(BaseTool):
    name = "transport"
    description = "Fetches live train / transit status for a route or station."
    parameters = {
        "route": {
            "type": "string",
            "description": "Route name, train number, or station, e.g. 'Mumbai-Pune express'",
        },
        "type": {
            "type": "string",
            "description": "'status' or 'schedule'. Default: 'status'",
            "default": "status",
        },
        "_test_delay_seconds": {
            "type": "number",
            "description": "Internal: inject artificial latency for stress testing.",
            "default": 0,
        },
    }
    permission_level = 3
    network_required = True
    api_key_required = True
    cancellable = True
    timeout_seconds = 12.0

    def __init__(self) -> None:
        self._cache: dict[str, tuple[dict, float]] = {}

    async def execute(
        self,
        args: dict[str, Any],
        generation: int,
        profile_id: Optional[str] = None,
    ) -> ToolResult:
        route = args.get("route", "").strip()
        query_type = args.get("type", "status")
        test_delay = float(args.get("_test_delay_seconds", 0))

        if not route:
            return ToolResult(
                success=False,
                output=None,
                tool_name=self.name,
                generation=generation,
                error="No route provided.",
            )

        # Artificial delay for stress tests — simulates a slow network call
        if test_delay > 0:
            logger.info(
                "TransportTool: injecting %.1fs delay for stress test (gen=%d).",
                test_delay,
                generation,
            )
            await asyncio.sleep(test_delay)

        cache_key = f"{route}:{query_type}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return ToolResult(
                success=True,
                output=cached,
                tool_name=self.name,
                generation=generation,
                from_cache=True,
                network_used=False,
            )

        try:
            from routing.external import ExternalServiceManager
            result = await ExternalServiceManager.request(
                service="transport",
                operation=query_type,
                args={"route": route},
                profile_id=profile_id,
                generation=generation,
            )
            if result.get("error"):
                return ToolResult(
                    success=False,
                    output=None,
                    tool_name=self.name,
                    generation=generation,
                    error=result["error"],
                    network_used=True,
                )
            self._set_cache(cache_key, result)
            return ToolResult(
                success=True,
                output=result,
                tool_name=self.name,
                generation=generation,
                network_used=True,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                success=False,
                output=None,
                tool_name=self.name,
                generation=generation,
                error=str(exc),
                network_used=True,
            )

    def _get_cache(self, key: str) -> Optional[dict]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry[1]) < _CACHE_TTL_SECONDS:
            return entry[0]
        return None

    def _set_cache(self, key: str, data: dict) -> None:
        self._cache[key] = (data, time.time())
