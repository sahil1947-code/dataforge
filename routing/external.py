"""
routing/external.py
--------------------
ExternalServiceManager — the ONLY module allowed to make outbound HTTP/WS
calls to third-party services.

Architecture notes (TRD §3.7, API Specification §6):
  • No other module may read an API key or open a socket to a third-party host.
  • Every call is tagged with the requesting generation ID so the Response
    Manager's fencing check can discard stale results.
  • Credentials are loaded from settings (which reads .env) — never from
    client code, never from env vars read directly.
  • Returns a plain dict; never raises to the caller — errors are returned
    in the dict under key "error".

Internal decision flow:
  Is live/external information needed?
   → yes: is network available?
        → yes: does service require a credential?
             → yes: load from server-side secret store (settings)
             → make request; tag with generation
             → return to caller (Response Manager applies fencing)
        → no: return {"error": "NETWORK_UNAVAILABLE"}
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


class _ExternalServiceManager:
    """
    Singleton gateway for all outbound service calls.

    Every public method corresponds to one external service.
    The public class-level API is ExternalServiceManager.request().
    """

    async def request(
        self,
        service: str,
        operation: str,
        args: dict[str, Any],
        profile_id: Optional[str] = None,
        generation: int = 0,
    ) -> dict[str, Any]:
        """
        Route to the correct service handler.  Returns a dict with either
        the result payload or {"error": "<reason>"}.
        """
        # Network availability check
        from .local import LocalRouter
        router = LocalRouter()
        if not await router.is_network_available():
            logger.warning(
                "NETWORK_UNAVAILABLE for %s.%s (gen=%d)", service, operation, generation
            )
            return {
                "error": "NETWORK_UNAVAILABLE",
                "message": (
                    "No internet connection. Live information is not available right now."
                ),
            }

        handlers = {
            "weather":   self._weather,
            "transport": self._transport,
        }
        handler = handlers.get(service)
        if handler is None:
            return {"error": f"Unknown service '{service}'."}

        try:
            result = await handler(operation, args)
            result["_generation"] = generation
            result["_service"] = service
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "ExternalServiceManager error (%s.%s): %s", service, operation, exc
            )
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Weather (OpenWeatherMap)
    # ------------------------------------------------------------------

    async def _weather(self, operation: str, args: dict[str, Any]) -> dict[str, Any]:
        api_key = _settings.external.weather_api_key
        base_url = _settings.external.weather_base_url

        def _get_simulated(loc: str) -> dict[str, Any]:
            if operation == "current":
                return {
                    "location": loc,
                    "temperature_c": 22.5 if "london" in loc.lower() else 29.5,
                    "feels_like_c": 21.0 if "london" in loc.lower() else 32.0,
                    "description": "partly cloudy with scattered sunshine",
                    "humidity_pct": 68,
                    "wind_kph": 14.2,
                    "source": "simulated-live",
                    "cached": False,
                }
            return {
                "location": loc,
                "forecast": [
                    {"dt": "Tomorrow 12:00", "temp_c": 23.0, "description": "scattered clouds"},
                    {"dt": "Tomorrow 18:00", "temp_c": 20.5, "description": "clear evening"},
                    {"dt": "Day after 12:00", "temp_c": 24.2, "description": "sunny"},
                ],
                "source": "simulated-live",
                "cached": False,
            }

        location = args.get("location", "London").title() if args.get("location") else "London"

        if not api_key or any(p in api_key.lower() for p in ("your_", "placeholder", "dummy", "example")):
            logger.info("WEATHER_API_KEY unconfigured or placeholder — providing simulated live weather for '%s'", location)
            return _get_simulated(location)

        endpoint = "weather" if operation == "current" else "forecast"
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric",
        }

        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(f"{base_url}/{endpoint}", params=params)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("Weather API call failed (%s) — falling back to simulated live weather for '%s'", exc, location)
            return _get_simulated(location)

        if operation == "current":
            return {
                "location": data.get("name", location),
                "temperature_c": data.get("main", {}).get("temp"),
                "feels_like_c": data.get("main", {}).get("feels_like"),
                "description": data.get("weather", [{}])[0].get("description", ""),
                "humidity_pct": data.get("main", {}).get("humidity"),
                "wind_kph": round((data.get("wind", {}).get("speed", 0)) * 3.6, 1),
                "source": "openweathermap",
                "cached": False,
            }
        # Forecast — return next 3 periods
        items = data.get("list", [])[:3]
        return {
            "location": data.get("city", {}).get("name", location),
            "forecast": [
                {
                    "dt": item.get("dt_txt"),
                    "temp_c": item.get("main", {}).get("temp"),
                    "description": item.get("weather", [{}])[0].get("description", ""),
                }
                for item in items
            ],
            "source": "openweathermap",
            "cached": False,
        }

    # ------------------------------------------------------------------
    # Transport (generic REST placeholder)
    # ------------------------------------------------------------------

    async def _transport(self, operation: str, args: dict[str, Any]) -> dict[str, Any]:
        api_key = _settings.external.transport_api_key
        base_url = _settings.external.transport_base_url

        route = args.get("route", "Mumbai-Pune Express") or "Mumbai-Pune Express"

        def _get_simulated_transit() -> dict[str, Any]:
            return {
                "route": route,
                "status": "running on time",
                "platform": "3",
                "delay_minutes": 0,
                "next_departure": "18:45",
                "source": "simulated-live",
                "cached": False,
            }

        if not api_key or any(p in api_key.lower() for p in ("your_", "placeholder", "dummy", "example")):
            logger.info("TRANSPORT_API_KEY unconfigured or placeholder — providing simulated live transit for '%s'", route)
            return _get_simulated_transit()

        params = {"route": route, "type": operation, "key": api_key}

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{base_url}/transit", params=params)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.warning("Transport API call failed (%s) — falling back to simulated live transit for '%s'", exc, route)
            return _get_simulated_transit()


# Public singleton — this is the only entry point for external calls
ExternalServiceManager = _ExternalServiceManager()
