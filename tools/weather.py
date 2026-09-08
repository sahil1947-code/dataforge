"""
tools/weather.py
----------------
Live weather tool — routes through ExternalServiceManager.

Permission level 3 (external access).  API key required.
Results are cached for 10 minutes to reduce network calls and support
the "cached" label in the Local/Live/Cached indicator (UI/UX Spec §8.1).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from .registry import BaseTool, ToolResult

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS: int = 600   # 10 minutes


class WeatherTool(BaseTool):
    name = "weather"
    description = "Fetches current or forecast weather for a location."
    parameters = {
        "location": {
            "type": "string",
            "description": "City name or coordinates, e.g. 'Mumbai' or '19.07,72.87'",
        },
        "type": {
            "type": "string",
            "description": "'current' or 'forecast'. Default: 'current'",
            "default": "current",
        },
    }
    permission_level = 3
    network_required = True
    api_key_required = True
    cancellable = True
    timeout_seconds = 8.0

    def __init__(self) -> None:
        self._cache: dict[str, tuple[dict, float]] = {}  # key → (data, timestamp)

    async def execute(
        self,
        args: dict[str, Any],
        generation: int,
        profile_id: Optional[str] = None,
    ) -> ToolResult:
        location = args.get("location", "").strip()
        query_type = args.get("type", "current")

        if not location:
            return ToolResult(
                success=False,
                output=None,
                tool_name=self.name,
                generation=generation,
                error="No location provided.",
            )

        cache_key = f"{location}:{query_type}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            logger.info("Weather cache hit for '%s'", location)
            return ToolResult(
                success=True,
                output=cached,
                tool_name=self.name,
                generation=generation,
                from_cache=True,
                network_used=False,
            )

        # Delegate the actual HTTP call to ExternalServiceManager
        # (imported here to avoid circular imports at module load time)
        try:
            from routing.external import ExternalServiceManager
            result = await ExternalServiceManager.request(
                service="weather",
                operation=query_type,
                args={"location": location},
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
            logger.error("Weather tool error: %s", exc)
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
