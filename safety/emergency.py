"""
safety/emergency.py
--------------------
Emergency detection classifier for RIME.

Architecture notes (TRD §3.4):
  • Secondary classifier — never triggers autonomous external action.
  • High confidence (≥ 0.85 by default) → emits a confirmable local prompt only.
  • Every trigger is written to emergency_events regardless of confidence.
  • Uses both keyword signals and optional acoustic features (speaking rate,
    pitch variance) when audio features are supplied.
  • The UI modal (UI/UX Spec §8.3) is the ONLY surface allowed to appear
    unsolicited — announced via Rime, never silently.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from config import get_settings
from database import execute, encode_json

logger = logging.getLogger(__name__)
_settings = get_settings()

# Keyword patterns that contribute to the distress score
_HIGH_CONFIDENCE_KEYWORDS = re.compile(
    r"\b(?:help me|i need help|emergency|call (?:the )?police|"
    r"i(?:'m| am) (?:in danger|hurt|injured|dying)|"
    r"somebody help|please help|can(?:'t| not) breathe|"
    r"chest pain|heart attack|i feel unsafe)\b",
    re.I,
)

_MODERATE_KEYWORDS = re.compile(
    r"\b(?:help|scared|afraid|danger|hurt|pain|emergency|"
    r"unsafe|threatening|attack|robbery|fire|accident|crash)\b",
    re.I,
)


@dataclass
class EmergencyEvent:
    event_id: str
    profile_id: Optional[str]
    trigger_type: str           # "keyword" | "acoustic" | "keyword+acoustic"
    confidence: float
    action_taken: str           # "none" | "confirmed_activated" | "cancelled"
    timestamp: str


class EmergencyDetector:
    """
    Classifies transcribed text (and optional audio features) for distress signals.

    Ownership: emergency_events table (write).
    The ONLY class in the system that writes to emergency_events.
    """

    def __init__(self) -> None:
        self._threshold = _settings.safety.emergency_confidence_threshold
        self._enabled = _settings.safety.enable_emergency_detection

    async def classify(
        self,
        transcript: str,
        profile_id: Optional[str] = None,
        audio_features: Optional[dict] = None,
    ) -> Optional[EmergencyEvent]:
        """
        Analyse `transcript` for distress signals.

        Returns an EmergencyEvent if the confidence is above the logging
        threshold (0.3), or None if clearly not distress.
        The caller is responsible for surfacing the UI modal when
        event.confidence >= self._threshold.
        """
        if not self._enabled:
            return None

        keyword_score = self._keyword_score(transcript)
        acoustic_score = self._acoustic_score(audio_features) if audio_features else 0.0
        trigger_type = self._trigger_type(keyword_score, acoustic_score)
        confidence = self._combined_score(keyword_score, acoustic_score)

        # Only log events with some evidence — avoids flooding the audit table
        if confidence < 0.20:
            return None

        event = EmergencyEvent(
            event_id=f"emg_{uuid.uuid4().hex[:12]}",
            profile_id=profile_id,
            trigger_type=trigger_type,
            confidence=round(confidence, 4),
            action_taken="none",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        # Every trigger is logged — even below the action threshold (TRD §3.4)
        await self._persist(event)
        logger.info(
            "Emergency signal: confidence=%.2f type=%s (threshold=%.2f)",
            confidence,
            trigger_type,
            self._threshold,
        )
        return event

    async def update_action(self, event_id: str, action: str) -> None:
        """
        Called by the confirmation endpoint when the user confirms or cancels.
        action must be 'confirmed_activated' or 'cancelled'.
        """
        valid = {"confirmed_activated", "cancelled"}
        if action not in valid:
            raise ValueError(f"Invalid action '{action}'. Must be one of {valid}.")
        await execute(
            "UPDATE emergency_events SET action_taken = ? WHERE event_id = ?",
            (action, event_id),
        )
        logger.info("Emergency event %s: action_taken = %s", event_id, action)

    @property
    def threshold(self) -> float:
        return self._threshold

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _keyword_score(self, text: str) -> float:
        if _HIGH_CONFIDENCE_KEYWORDS.search(text):
            return 0.80
        count = len(_MODERATE_KEYWORDS.findall(text))
        return min(0.60, count * 0.15)

    def _acoustic_score(self, features: dict) -> float:
        """
        Derive a distress score from acoustic features.
        Expected keys: speaking_rate_wpm, pitch_variance, energy_mean.
        All are optional; missing keys contribute 0.
        """
        score = 0.0
        # Fast speaking rate (> 180 wpm) is associated with distress
        wpm = features.get("speaking_rate_wpm", 0)
        if wpm > 200:
            score += 0.20
        elif wpm > 180:
            score += 0.10
        # High pitch variance
        pitch_var = features.get("pitch_variance", 0)
        if pitch_var > 1.5:
            score += 0.15
        elif pitch_var > 1.0:
            score += 0.08
        return min(0.40, score)

    def _combined_score(self, kw: float, acoustic: float) -> float:
        # Keyword signal is weighted 70%, acoustic 30%
        return round(kw * 0.70 + acoustic * 0.30, 4)

    def _trigger_type(self, kw: float, acoustic: float) -> str:
        if kw > 0 and acoustic > 0:
            return "keyword+acoustic"
        if kw > 0:
            return "keyword"
        return "acoustic"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    async def _persist(self, event: EmergencyEvent) -> None:
        await execute(
            """
            INSERT INTO emergency_events
                (event_id, profile_id, timestamp, trigger_type, confidence, action_taken)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.event_id,
                event.profile_id,
                event.timestamp,
                event.trigger_type,
                event.confidence,
                event.action_taken,
            ),
        )
