"""
tools/time_tool.py
------------------
Local date/time tool — returns current date, time, day, timezone info.
No network, no credentials, permission level 0.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Optional
import zoneinfo

from .registry import BaseTool, ToolResult


class TimeTool(BaseTool):
    name = "time"
    description = "Returns the current local date and time. No network required."
    parameters = {
        "timezone": {
            "type": "string",
            "description": "IANA timezone name, e.g. 'Asia/Kolkata'. Defaults to UTC.",
            "default": "UTC",
        },
        "format": {
            "type": "string",
            "description": "One of: 'full' | 'date' | 'time' | 'day'. Default: 'full'.",
            "default": "full",
        },
    }
    permission_level = 0
    network_required = False
    api_key_required = False
    cancellable = True
    timeout_seconds = 1.0

    async def execute(
        self,
        args: dict[str, Any],
        generation: int,
        profile_id: Optional[str] = None,
    ) -> ToolResult:
        tz_name = args.get("timezone", "UTC") or "UTC"
        fmt = args.get("format", "full") or "full"

        try:
            tz = zoneinfo.ZoneInfo(tz_name)
        except Exception:
            tz = timezone.utc
            tz_name = "UTC"

        now = datetime.now(tz)
        output = {
            "iso":      now.strftime("%Y-%m-%dT%H:%M:%S"),
            "date":     now.strftime("%A, %d %B %Y"),
            "time":     now.strftime("%I:%M %p"),
            "day":      now.strftime("%A"),
            "timezone": tz_name,
        }

        if fmt == "date":
            spoken = output["date"]
        elif fmt == "time":
            spoken = output["time"]
        elif fmt == "day":
            spoken = output["day"]
        else:
            spoken = f"{output['date']}, {output['time']} ({tz_name})"

        output["spoken"] = spoken
        return ToolResult(
            success=True,
            output=output,
            tool_name=self.name,
            generation=generation,
        )
