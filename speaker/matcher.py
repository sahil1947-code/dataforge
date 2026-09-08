"""
speaker/matcher.py
------------------
Nearest-neighbour speaker matching against stored embeddings.

Implements the three-tier confidence policy from TRD §3.3:
  • similarity ≥ 0.80  →  CONFIDENT match  →  load profile silently
  • 0.60 ≤ sim < 0.80  →  TENTATIVE match  →  ask a confirming question
  • sim < 0.60          →  UNKNOWN          →  create new anonymous profile

Performance target: ≤ 200 ms for up to 50 enrolled speakers (TRD §3.3).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from config import get_settings
from database import fetch_all, rows_to_dicts
from .embedding import SpeakerEmbedder, cosine_similarity

logger = logging.getLogger(__name__)
_settings = get_settings()


class MatchConfidence(str, Enum):
    CONFIDENT = "confident"    # similarity ≥ CONFIDENT_THRESHOLD
    TENTATIVE = "tentative"    # TENTATIVE_THRESHOLD ≤ similarity < CONFIDENT_THRESHOLD
    UNKNOWN = "unknown"        # similarity < TENTATIVE_THRESHOLD


@dataclass
class MatchResult:
    confidence: MatchConfidence
    profile_id: Optional[str]          # None when UNKNOWN
    similarity: float                  # best cosine similarity found (0.0 if no profiles)
    matched_embedding_id: Optional[str]
    latency_ms: float


class SpeakerMatcher:
    """
    Compares a candidate embedding against every stored voice_embedding row
    and returns the best match according to the three-tier policy.

    Owned tables: voice_embeddings (read only here)
    """

    def __init__(self, embedder: SpeakerEmbedder) -> None:
        self._embedder = embedder
        self._confident_threshold = _settings.speaker.confident_threshold
        self._tentative_threshold = _settings.speaker.tentative_threshold

    async def match(self, audio_float32: np.ndarray) -> MatchResult:
        """
        Embed the supplied audio and find the closest stored profile.
        Returns a MatchResult in ≤ 200 ms for up to 50 enrolled speakers.
        """
        t_start = time.monotonic()

        candidate = await self._embedder.embed(audio_float32)
        rows = await fetch_all(
            "SELECT embedding_id, profile_id, embedding_vector FROM voice_embeddings"
        )
        latency_ms = (time.monotonic() - t_start) * 1000

        if not rows:
            return MatchResult(
                confidence=MatchConfidence.UNKNOWN,
                profile_id=None,
                similarity=0.0,
                matched_embedding_id=None,
                latency_ms=latency_ms,
            )

        best_sim = -1.0
        best_profile_id: Optional[str] = None
        best_emb_id: Optional[str] = None

        for row in rows:
            stored = self._embedder.blob_to_vector(row["embedding_vector"])
            sim = cosine_similarity(candidate, stored)
            if sim > best_sim:
                best_sim = sim
                best_profile_id = row["profile_id"]
                best_emb_id = row["embedding_id"]

        latency_ms = (time.monotonic() - t_start) * 1000

        if best_sim >= self._confident_threshold:
            conf = MatchConfidence.CONFIDENT
        elif best_sim >= self._tentative_threshold:
            conf = MatchConfidence.TENTATIVE
        else:
            conf = MatchConfidence.UNKNOWN
            best_profile_id = None
            best_emb_id = None

        logger.debug(
            "Speaker match: confidence=%s similarity=%.3f profile=%s latency=%.1fms",
            conf.value,
            best_sim,
            best_profile_id,
            latency_ms,
        )

        return MatchResult(
            confidence=conf,
            profile_id=best_profile_id,
            similarity=best_sim,
            matched_embedding_id=best_emb_id,
            latency_ms=latency_ms,
        )
