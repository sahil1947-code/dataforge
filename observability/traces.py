"""
observability/traces.py
------------------------
Per-request pipeline trace collector and event timeline builder.

Every request that passes through the pipeline produces a PipelineTrace
capturing the latency of each stage.  These traces power:
  • The Developer Dashboard (UI/UX Spec §7.2)
  • The GET /api/v1/sessions/{id}/timeline replay endpoint
  • The latency benchmark table in RIME_EVIDENCE.md

Architecture notes (TRD §3.12):
  • Cached vs. uncached measurements are labeled separately — never averaged.
  • Every interruption is a first-class event in the trace timeline.
  • Traces are persisted to the events table for durability and replay.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from database import execute, fetch_all, rows_to_dicts, encode_json


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class TraceEvent:
    """A single timestamped event in a pipeline trace."""
    name: str                    # e.g. "vad_complete", "stt_complete"
    timestamp: float             # monotonic time for duration calculation
    wall_time: str               # ISO-8601 wall clock time for display
    latency_ms: Optional[float] = None
    cached: bool = False
    generation: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTrace:
    """
    Full trace for one user request from speech onset to Rime audio start.

    Fields map directly to the Developer Dashboard layout (UI/UX Spec §7.2)
    and the latency table in TRD §5.
    """
    trace_id: str
    conversation_id: str
    generation: int
    profile_id: Optional[str]
    started_at: float = field(default_factory=time.monotonic)
    events: list[TraceEvent] = field(default_factory=list)

    # Individual stage latencies (ms) — populated as each stage completes
    vad_ms: Optional[float] = None
    stt_ms: Optional[float] = None
    speaker_match_ms: Optional[float] = None
    llm_first_token_ms: Optional[float] = None
    tool_ms: Optional[float] = None
    tool_cached: bool = False
    rime_first_audio_ms: Optional[float] = None
    total_perceived_ms: Optional[float] = None
    interrupted: bool = False

    def add_event(
        self,
        name: str,
        latency_ms: Optional[float] = None,
        cached: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                name=name,
                timestamp=time.monotonic(),
                wall_time=_utcnow(),
                latency_ms=latency_ms,
                cached=cached,
                generation=self.generation,
                metadata=metadata or {},
            )
        )

    def mark_complete(self) -> None:
        """Finalise the trace — compute total perceived latency."""
        self.total_perceived_ms = (time.monotonic() - self.started_at) * 1000
        self.add_event("trace_complete", latency_ms=self.total_perceived_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "generation": self.generation,
            "profile_id": self.profile_id,
            "vad_ms": self.vad_ms,
            "stt_ms": self.stt_ms,
            "speaker_match_ms": self.speaker_match_ms,
            "llm_first_token_ms": self.llm_first_token_ms,
            "tool_ms": self.tool_ms,
            "tool_cached": self.tool_cached,
            "rime_first_audio_ms": self.rime_first_audio_ms,
            "total_perceived_ms": self.total_perceived_ms,
            "interrupted": self.interrupted,
            "events": [
                {
                    "name": e.name,
                    "wall_time": e.wall_time,
                    "latency_ms": e.latency_ms,
                    "cached": e.cached,
                    "generation": e.generation,
                    **e.metadata,
                }
                for e in self.events
            ],
        }


class TraceCollector:
    """
    Creates and stores PipelineTraces for active requests.

    One TraceCollector per application instance (singleton-like, owned by
    the app startup code).  Traces are kept in memory for the current window
    and persisted to the events table for replay.
    """

    def __init__(self, max_in_memory: int = 500) -> None:
        self._traces: dict[str, PipelineTrace] = {}
        self._completed: list[PipelineTrace] = []
        self._max = max_in_memory

    def start(
        self,
        conversation_id: str,
        generation: int,
        profile_id: Optional[str] = None,
    ) -> PipelineTrace:
        trace = PipelineTrace(
            trace_id=f"trace_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            generation=generation,
            profile_id=profile_id,
        )
        trace.add_event("trace_started")
        self._traces[trace.trace_id] = trace
        return trace

    async def finish(self, trace: PipelineTrace) -> None:
        """Finalise the trace, persist it, and move it to the completed list."""
        trace.mark_complete()
        self._traces.pop(trace.trace_id, None)
        self._completed.append(trace)
        if len(self._completed) > self._max:
            self._completed.pop(0)
        await self._persist(trace)

    def get_active(self, trace_id: str) -> Optional[PipelineTrace]:
        return self._traces.get(trace_id)

    def get_completed(
        self,
        conversation_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        traces = self._completed
        if conversation_id:
            traces = [t for t in traces if t.conversation_id == conversation_id]
        return [t.to_dict() for t in traces[-limit:]]

    async def get_session_timeline(
        self, conversation_id: str
    ) -> list[dict[str, Any]]:
        """
        Return the full event timeline for a session — used by the
        GET /api/v1/sessions/{id}/timeline replay endpoint.
        """
        rows = await fetch_all(
            """
            SELECT metadata FROM events
             WHERE event_type = 'pipeline_trace'
               AND json_extract(metadata, '$.conversation_id') = ?
             ORDER BY timestamp ASC
            """,
            (conversation_id,),
        )
        import json
        result = []
        for row in rows:
            try:
                result.append(json.loads(row["metadata"]))
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # Aggregated metrics (for GET /api/v1/metrics/latency)
    # ------------------------------------------------------------------

    def compute_latency_summary(self) -> dict[str, Any]:
        """
        Compute mean latencies for the most recent completed traces,
        split by cached vs. uncached — as required by TRD §3.12.
        """
        if not self._completed:
            return {"window": "last_500", "uncached": {}, "cached": {}}

        uncached: list[PipelineTrace] = []
        cached: list[PipelineTrace] = []

        for t in self._completed[-200:]:
            if t.tool_cached:
                cached.append(t)
            else:
                uncached.append(t)

        def avg(values: list[Optional[float]]) -> Optional[float]:
            vals = [v for v in values if v is not None]
            return round(sum(vals) / len(vals), 1) if vals else None

        def summarise(traces: list[PipelineTrace]) -> dict[str, Any]:
            return {
                "vad_ms":              avg([t.vad_ms for t in traces]),
                "stt_ms":              avg([t.stt_ms for t in traces]),
                "speaker_match_ms":    avg([t.speaker_match_ms for t in traces]),
                "llm_first_token_ms":  avg([t.llm_first_token_ms for t in traces]),
                "tool_ms":             avg([t.tool_ms for t in traces]),
                "rime_first_audio_ms": avg([t.rime_first_audio_ms for t in traces]),
                "total_ms":            avg([t.total_perceived_ms for t in traces]),
                "sample_count":        len(traces),
            }

        return {
            "window": "last_200",
            "uncached": summarise(uncached),
            "cached": summarise(cached),
        }

    def compute_interruption_summary(self) -> dict[str, Any]:
        interrupted = [t for t in self._completed if t.interrupted]
        total = len(self._completed)
        n_interrupted = len(interrupted)
        success_rate = (
            round((n_interrupted / total) * 100, 1) if total else None
        )
        return {
            "total_requests": total,
            "interrupted_count": n_interrupted,
            "interruption_rate_pct": success_rate,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist(self, trace: PipelineTrace) -> None:
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        try:
            await execute(
                """
                INSERT INTO events
                    (event_id, profile_id, event_type, timestamp, metadata)
                VALUES (?, ?, 'pipeline_trace', ?, ?)
                """,
                (
                    event_id,
                    trace.profile_id,
                    _utcnow(),
                    encode_json(trace.to_dict()),
                ),
            )
        except Exception:  # noqa: BLE001
            pass  # Trace persistence failure must never affect the user
