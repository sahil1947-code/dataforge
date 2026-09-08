"""
observability/dashboard.py
---------------------------
Developer / Performance Dashboard data provider.

Serves the data behind UI/UX Spec §7 (Developer Dashboard) and the
GET /api/v1/metrics/* endpoints.

Surfaces:
  • Per-stage latency (VAD, STT, speaker match, LLM, tool, Rime, total)
  • Cached vs. uncached split (TRD §3.12 — never averaged together)
  • Interruption correctness rate (PRD §9 metric — target ≥ 95%)
  • Stale-result rejection rate
  • Session event timeline for replay
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from database import fetch_all, rows_to_dicts
from .traces import TraceCollector


@dataclass
class InterruptionStats:
    total_trials: int
    correct: int             # stale artifact never surfaced
    incorrect: int           # stale artifact reached the user (must be 0)
    correctness_rate_pct: float
    mean_interrupt_latency_ms: Optional[float]
    stale_artifacts_rejected: int


class MetricsDashboard:
    """
    Aggregates and exposes performance metrics for the Developer Dashboard
    and the /api/v1/metrics/* REST endpoints.
    """

    def __init__(self, trace_collector: TraceCollector) -> None:
        self._traces = trace_collector
        # Interrupt events are accumulated here by the ResponseManager callback
        self._interrupt_events: list[dict[str, Any]] = []
        self._stale_rejected: int = 0
        self._stale_surfaced: int = 0  # must always be 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_interrupt(
        self,
        session_id: str,
        old_gen: int,
        new_gen: int,
        artifacts_discarded: int,
        latency_ms: float,
    ) -> None:
        """Called by the Response Manager interrupt handler."""
        self._interrupt_events.append({
            "session_id": session_id,
            "old_gen": old_gen,
            "new_gen": new_gen,
            "artifacts_discarded": artifacts_discarded,
            "latency_ms": latency_ms,
            "timestamp": time.monotonic(),
        })
        self._stale_rejected += artifacts_discarded

    def record_stale_surfaced(self) -> None:
        """
        Called if — and only if — a stale artifact reaches the user.
        In a correctly functioning system this is never called.
        Tracked for the test harness to assert correctness_rate == 100%.
        """
        self._stale_surfaced += 1

    def get_latency_metrics(self) -> dict[str, Any]:
        return self._traces.compute_latency_summary()

    def get_interruption_metrics(self) -> InterruptionStats:
        total = len(self._interrupt_events)
        if total == 0:
            return InterruptionStats(
                total_trials=0,
                correct=0,
                incorrect=0,
                correctness_rate_pct=0.0,
                mean_interrupt_latency_ms=None,
                stale_artifacts_rejected=0,
            )

        incorrect = self._stale_surfaced
        correct = total - incorrect
        rate = round((correct / total) * 100, 2)
        mean_lat = round(
            sum(e["latency_ms"] for e in self._interrupt_events) / total, 1
        )
        return InterruptionStats(
            total_trials=total,
            correct=correct,
            incorrect=incorrect,
            correctness_rate_pct=rate,
            mean_interrupt_latency_ms=mean_lat,
            stale_artifacts_rejected=self._stale_rejected,
        )

    async def get_session_timeline(
        self, conversation_id: str
    ) -> list[dict[str, Any]]:
        """
        Full event-by-event timeline for session replay (App Flow §15,
        API Spec §4.8).  Returns events in chronological order.
        """
        return await self._traces.get_session_timeline(conversation_id)

    def get_session_summary(self, conversation_id: str) -> dict[str, Any]:
        """Return the most recent completed trace for a conversation."""
        traces = self._traces.get_completed(conversation_id=conversation_id, limit=1)
        return traces[0] if traces else {}

    def format_for_dashboard(self) -> dict[str, Any]:
        """
        Single dict consumed by the Developer Dashboard UI (UI/UX Spec §7).
        """
        latency = self.get_latency_metrics()
        interruption = self.get_interruption_metrics()
        return {
            "latency": latency,
            "interruption": {
                "total_trials": interruption.total_trials,
                "correct": interruption.correct,
                "correctness_rate_pct": interruption.correctness_rate_pct,
                "mean_interrupt_to_silence_ms": interruption.mean_interrupt_latency_ms,
                "stale_artifacts_rejected": interruption.stale_artifacts_rejected,
                "stale_artifacts_surfaced": self._stale_surfaced,
            },
        }
