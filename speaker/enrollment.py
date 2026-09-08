"""
speaker/enrollment.py
---------------------
Automatic voice enrollment for new speakers.

When the Speaker Matcher returns no confident match, EnrollmentService
creates an anonymous profile and stores the first voice embedding.

Architecture notes (TRD §3.3, App Flow §3):
  • Silent and automatic — no explicit "sign-up" step
  • Profile ID format: "speaker_<8-char hex hash of first embedding>"
  • Multiple embeddings per profile accumulate over time (one-to-many)
  • Quality score is stored alongside each embedding for future pruning
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from datetime import datetime, timezone

import numpy as np

from database import transaction, fetch_one, execute, encode_json
from .embedding import SpeakerEmbedder

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _profile_id_from_vector(vector: np.ndarray) -> str:
    """Derive a short, deterministic profile ID from the first embedding."""
    digest = hashlib.sha256(vector.tobytes()).hexdigest()[:8]
    return f"speaker_{digest}"


class EnrollmentService:
    """
    Creates new profiles and attaches voice embeddings to existing ones.

    Owned tables: profiles, voice_embeddings  (Backend Schema §4)
    """

    def __init__(self, embedder: SpeakerEmbedder) -> None:
        self._embedder = embedder

    # ------------------------------------------------------------------
    # New speaker
    # ------------------------------------------------------------------

    async def enroll_new(
        self,
        audio_float32: np.ndarray,
        quality_score: float = 0.0,
    ) -> str:
        """
        Compute an embedding from the supplied audio, create an anonymous
        profile, persist the embedding, and return the new profile_id.
        """
        vector = await self._embedder.embed(audio_float32)
        profile_id = _profile_id_from_vector(vector)
        now = _utcnow()

        async with transaction():
            # Upsert profile (in case a parallel enrollment raced us)
            existing = await fetch_one(
                "SELECT profile_id FROM profiles WHERE profile_id = ?",
                (profile_id,),
            )
            if existing is None:
                await execute(
                    """
                    INSERT INTO profiles
                        (profile_id, display_name, created_at, last_seen,
                         status, preferences_json)
                    VALUES (?, NULL, ?, ?, 'active', '{}')
                    """,
                    (profile_id, now, now),
                )
                logger.info("New profile created: %s", profile_id)
            else:
                await execute(
                    "UPDATE profiles SET last_seen = ? WHERE profile_id = ?",
                    (now, profile_id),
                )

            await self._store_embedding(profile_id, vector, quality_score, now)

        return profile_id

    # ------------------------------------------------------------------
    # Add embedding to existing profile
    # ------------------------------------------------------------------

    async def add_embedding(
        self,
        profile_id: str,
        audio_float32: np.ndarray,
        quality_score: float = 0.0,
    ) -> str:
        """
        Compute a new embedding from `audio_float32` and attach it to the
        given profile.  Returns the new embedding_id.
        Used to improve match confidence under different acoustic conditions.
        """
        vector = await self._embedder.embed(audio_float32)
        now = _utcnow()
        embedding_id = await self._store_embedding(
            profile_id, vector, quality_score, now
        )
        logger.info(
            "Added embedding %s to profile %s (quality=%.2f)",
            embedding_id,
            profile_id,
            quality_score,
        )
        return embedding_id

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _store_embedding(
        self,
        profile_id: str,
        vector: np.ndarray,
        quality_score: float,
        now: str,
    ) -> str:
        embedding_id = f"emb_{uuid.uuid4().hex[:12]}"
        blob = self._embedder.vector_to_blob(vector)
        await execute(
            """
            INSERT INTO voice_embeddings
                (embedding_id, profile_id, embedding_vector, quality_score, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (embedding_id, profile_id, blob, quality_score, now),
        )
        return embedding_id
