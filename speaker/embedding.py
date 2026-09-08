"""
speaker/embedding.py
--------------------
Local speaker embedding model integration.

Wraps SpeechBrain's ECAPA-TDNN (or an equivalent lightweight model) to
produce a fixed-length embedding vector from a raw PCM segment.

Architecture notes (TRD §3.3):
  • No network dependency — model weights stored locally
  • Embedding dimensions: 192 (ECAPA-TDNN small) or 512 (full)
  • Target latency: ≤ 200 ms for a 2-second speech segment
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Dimensionality of the output embedding vector
EMBEDDING_DIM: int = 192
_MODEL_DIR = Path.home() / ".cache" / "rime" / "speaker_model"


class SpeakerEmbedder:
    """
    Produces normalised L2 embedding vectors from a float32 PCM array.

    Usage:
        embedder = SpeakerEmbedder()
        await embedder.load()
        vector = await embedder.embed(pcm_float32)  # shape (EMBEDDING_DIM,)
    """

    def __init__(self) -> None:
        self._model = None
        self._loaded = False

    async def load(self) -> None:
        """Load model weights from local cache (downloads once on first run)."""
        if self._loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_blocking)
        self._loaded = True

    def _load_blocking(self) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(_MODEL_DIR),
                run_opts={"device": "cpu"},
            )
            logger.info("ECAPA-TDNN speaker embedding model loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SpeechBrain unavailable (%s) — using hash-based stub embedder.", exc
            )
            self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(self, audio_float32: np.ndarray) -> np.ndarray:
        """
        Return a normalised float32 embedding vector of shape (EMBEDDING_DIM,).

        The audio array must be at 16 kHz, mono, float32 in [-1, 1].
        """
        t_start = time.monotonic()
        loop = asyncio.get_running_loop()
        vector = await loop.run_in_executor(None, self._embed_blocking, audio_float32)
        latency_ms = (time.monotonic() - t_start) * 1000
        logger.debug("Embedding computed in %.1f ms", latency_ms)
        return vector

    def _embed_blocking(self, audio: np.ndarray) -> np.ndarray:
        if self._model is not None:
            return self._speechbrain_embed(audio)
        return self._stub_embed(audio)

    def _speechbrain_embed(self, audio: np.ndarray) -> np.ndarray:
        try:
            import torch  # type: ignore

            tensor = torch.tensor(audio).unsqueeze(0)  # (1, T)
            with torch.no_grad():
                embedding = self._model.encode_batch(tensor)
            vec = embedding.squeeze().numpy()
            return _l2_normalise(vec[:EMBEDDING_DIM])
        except Exception as exc:  # noqa: BLE001
            logger.warning("SpeechBrain embed error: %s — using stub", exc)
            return self._stub_embed(audio)

    @staticmethod
    def _stub_embed(audio: np.ndarray) -> np.ndarray:
        """
        Deterministic hash-based stub.
        Different audio waveforms produce different (but reproducible) vectors,
        which lets the test suite exercise matching logic without real ML models.
        """
        digest = hashlib.sha256(audio.tobytes()).digest()
        # Unpack 24 floats from the first 96 bytes of the digest, repeated/truncated
        values = []
        seed = digest
        while len(values) < EMBEDDING_DIM:
            seed = hashlib.sha256(seed).digest()
            for i in range(0, len(seed) - 3, 4):
                f = struct.unpack("f", seed[i : i + 4])[0]
                if not (f != f):  # skip NaN
                    values.append(f)
        vec = np.array(values[:EMBEDDING_DIM], dtype=np.float32)
        return _l2_normalise(vec)

    # ------------------------------------------------------------------
    # Serialisation helpers (for database storage)
    # ------------------------------------------------------------------

    @staticmethod
    def vector_to_blob(vector: np.ndarray) -> bytes:
        """Serialise a float32 numpy vector to raw bytes for SQLite BLOB."""
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def blob_to_vector(blob: bytes) -> np.ndarray:
        """Deserialise bytes from SQLite BLOB back to a float32 numpy array."""
        return np.frombuffer(blob, dtype=np.float32).copy()


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two L2-normalised vectors (= their dot product)."""
    return float(np.dot(a, b))
