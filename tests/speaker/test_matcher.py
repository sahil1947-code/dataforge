"""
tests/speaker/test_matcher.py
------------------------------
Tests for the speaker matcher — 3-tier confidence policy and
correct profile selection from stored embeddings.
"""

import asyncio
import struct
import pytest
import numpy as np
from unittest.mock import AsyncMock, patch

from speaker.matcher import SpeakerMatcher, MatchConfidence, MatchResult
from speaker.embedding import SpeakerEmbedder, cosine_similarity, EMBEDDING_DIM


def _make_unit_vector(seed: float) -> np.ndarray:
    """Create a deterministic unit vector for testing."""
    rng = np.random.default_rng(int(seed * 1000))
    v = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def _vec_to_blob(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


@pytest.mark.asyncio
async def test_no_profiles_returns_unknown():
    embedder = SpeakerEmbedder()
    matcher = SpeakerMatcher(embedder)

    audio = np.zeros(16000, dtype=np.float32)

    with patch("speaker.matcher.fetch_all", new=AsyncMock(return_value=[])):
        result = await matcher.match(audio)

    assert result.confidence == MatchConfidence.UNKNOWN
    assert result.profile_id is None
    assert result.similarity == 0.0


@pytest.mark.asyncio
async def test_confident_match():
    embedder = SpeakerEmbedder()
    matcher = SpeakerMatcher(embedder)

    # Make candidate and stored vector nearly identical
    v = _make_unit_vector(1.0)
    candidate = v + np.random.default_rng(0).standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.001
    candidate = candidate / np.linalg.norm(candidate)

    stored_row = {
        "embedding_id": "emb_001",
        "profile_id":   "speaker_abc",
        "embedding_vector": _vec_to_blob(v),
    }

    class _MockRow(dict):
        def __getitem__(self, k): return dict.__getitem__(self, k)

    with patch("speaker.matcher.fetch_all", new=AsyncMock(return_value=[stored_row])):
        with patch.object(embedder, "embed", new=AsyncMock(return_value=candidate)):
            result = await matcher.match(np.zeros(16000, dtype=np.float32))

    assert result.confidence == MatchConfidence.CONFIDENT
    assert result.profile_id == "speaker_abc"
    assert result.similarity >= 0.80


@pytest.mark.asyncio
async def test_unknown_match_below_threshold():
    embedder = SpeakerEmbedder()
    matcher = SpeakerMatcher(embedder)

    # Orthogonal vectors → similarity ≈ 0
    v1 = _make_unit_vector(1.0)
    v2 = _make_unit_vector(2.0)
    # Ensure they're actually dissimilar
    v2 = v2 - np.dot(v2, v1) * v1
    v2 = v2 / np.linalg.norm(v2)

    stored_row = {
        "embedding_id": "emb_002",
        "profile_id": "speaker_xyz",
        "embedding_vector": _vec_to_blob(v1),
    }

    with patch("speaker.matcher.fetch_all", new=AsyncMock(return_value=[stored_row])):
        with patch.object(embedder, "embed", new=AsyncMock(return_value=v2)):
            result = await matcher.match(np.zeros(16000, dtype=np.float32))

    assert result.confidence == MatchConfidence.UNKNOWN
    assert result.profile_id is None


@pytest.mark.asyncio
async def test_best_profile_selected_when_multiple():
    embedder = SpeakerEmbedder()
    matcher = SpeakerMatcher(embedder)

    target = _make_unit_vector(99.0)
    close  = target + np.random.default_rng(5).standard_normal(EMBEDDING_DIM).astype(np.float32) * 0.01
    close  = close / np.linalg.norm(close)
    far    = _make_unit_vector(50.0)

    rows = [
        {"embedding_id": "e1", "profile_id": "profile_far",   "embedding_vector": _vec_to_blob(far)},
        {"embedding_id": "e2", "profile_id": "profile_close", "embedding_vector": _vec_to_blob(close)},
    ]

    with patch("speaker.matcher.fetch_all", new=AsyncMock(return_value=rows)):
        with patch.object(embedder, "embed", new=AsyncMock(return_value=target)):
            result = await matcher.match(np.zeros(16000, dtype=np.float32))

    assert result.profile_id == "profile_close"


def test_cosine_similarity_identical():
    v = _make_unit_vector(7.0)
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    v1 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v2 = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    v1[0] = 1.0
    v2[1] = 1.0
    assert abs(cosine_similarity(v1, v2)) < 1e-5
