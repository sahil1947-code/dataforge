"""
observability/logger.py
------------------------
Structured logging setup for RIME.

Produces JSON-formatted log records for easy machine parsing while staying
human-readable in development mode (rich console output).

Architecture notes (TRD §3.12):
  • Every request must produce a structured event trace covering:
      VAD latency, STT latency, LLM time-to-first-token, tool latency,
      Rime time-to-first-audio, total perceived latency.
  • Cached vs. uncached measurements are labeled separately — never averaged.
  • Every interruption event must log: interrupt timestamp, generation
    before/after, artifacts discarded, and the final artifact spoken.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional

from config import get_settings

_settings = get_settings()
_LOG_LEVEL = getattr(logging, _settings.app.log_level.upper(), logging.INFO)

# Module-level flag — setup_logging() is idempotent
_configured = False


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

def setup_logging() -> None:
    """Configure the root logger.  Safe to call multiple times."""
    global _configured
    if _configured:
        return

    try:
        from rich.console import Console  # type: ignore
        from rich.logging import RichHandler  # type: ignore

        console = Console(highlight=False, legacy_windows=False)
        logging.basicConfig(
            level=_LOG_LEVEL,
            format="%(message)s",
            datefmt="[%H:%M:%S]",
            handlers=[RichHandler(console=console, rich_tracebacks=True, markup=False)],
        )
    except ImportError:
        logging.basicConfig(
            level=_LOG_LEVEL,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            stream=sys.stdout,
        )

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "websockets", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger, ensuring setup has been called."""
    setup_logging()
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Structured event helpers — written to both the log stream and the DB
# ---------------------------------------------------------------------------

_event_logger = logging.getLogger("rime.events")


def log_pipeline_event(
    event_type: str,
    session_id: Optional[str] = None,
    generation: Optional[int] = None,
    latency_ms: Optional[float] = None,
    cached: Optional[bool] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """
    Emit a single structured pipeline event.
    Called by every stage (VAD, STT, LLM, Rime, tools) after completion.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
    }
    if session_id is not None:
        record["session_id"] = session_id
    if generation is not None:
        record["generation"] = generation
    if latency_ms is not None:
        record["latency_ms"] = round(latency_ms, 2)
        record["cached"] = cached if cached is not None else False
    if extra:
        record.update(extra)

    _event_logger.info(json.dumps(record))


def log_interrupt_event(
    session_id: str,
    old_generation: int,
    new_generation: int,
    artifacts_discarded: int,
    interrupt_latency_ms: float,
) -> None:
    """
    Structured interrupt log — provides the replay data required by
    RIME_EVIDENCE.md: interrupt timestamp, generation before/after,
    artifacts discarded, and interrupt-to-audio-stop latency.
    """
    log_pipeline_event(
        event_type="interrupt",
        session_id=session_id,
        generation=new_generation,
        latency_ms=interrupt_latency_ms,
        extra={
            "old_generation": old_generation,
            "new_generation": new_generation,
            "artifacts_discarded": artifacts_discarded,
        },
    )
