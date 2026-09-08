"""
audio/vad.py
------------
Voice Activity Detection for RIME.

Wraps the Silero VAD model (or a lightweight fallback) to classify each
incoming AudioFrame as SPEECH, SILENCE, or NOISE.

Architecture notes (TRD §3.2):
  • Adds ≤ 30 ms processing latency per frame
  • No network dependency — model is loaded from local cache
  • Two consecutive SPEECH frames transition the session to VOICE_DETECTED
  • Emits a VADResult per frame; callers subscribe to get_events()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Optional

import numpy as np

from .capture import AudioFrame, SAMPLE_RATE

logger = logging.getLogger(__name__)


class VADLabel(str, Enum):
    SPEECH = "SPEECH"
    SILENCE = "SILENCE"
    NOISE = "NOISE"


@dataclass
class VADResult:
    label: VADLabel
    confidence: float          # 0.0–1.0
    frame_timestamp: float     # matches AudioFrame.timestamp
    processing_latency_ms: float


# Consecutive SPEECH frame threshold before emitting a VOICE_DETECTED signal
_SPEECH_ONSET_FRAMES: int = 2
# Confidence threshold to call a frame SPEECH
_SPEECH_THRESHOLD: float = 0.50


class VoiceActivityDetector:
    """
    Classifies AudioFrames into SPEECH / SILENCE / NOISE.

    Usage:
        vad = VoiceActivityDetector()
        await vad.load()
        async for result in vad.process(frame_stream):
            if result.label == VADLabel.SPEECH:
                ...
    """

    def __init__(self, speech_threshold: float = _SPEECH_THRESHOLD) -> None:
        self._threshold = speech_threshold
        self._model = None
        self._utils = None
        self._loaded = False
        self._consecutive_speech: int = 0
        self._onset_callbacks: list = []
        self._result_queue: asyncio.Queue[VADResult] = asyncio.Queue(maxsize=1000)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Load the Silero VAD model (runs once; cached locally by torch.hub)."""
        if self._loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model_blocking)
        self._loaded = True

    def _load_model_blocking(self) -> None:
        try:
            import torch  # type: ignore

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self._model = model
            self._utils = utils
            logger.info("Silero VAD model loaded.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Silero VAD unavailable (%s) — using energy-based fallback.", exc
            )
            self._model = None

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    async def process_frame(self, frame: AudioFrame) -> VADResult:
        """
        Classify a single AudioFrame.  Returns a VADResult immediately.
        Also pushes the result to the internal queue for subscribers.
        """
        t_start = time.monotonic()
        label, confidence = await asyncio.get_running_loop().run_in_executor(
            None, self._classify, frame.float32
        )
        latency_ms = (time.monotonic() - t_start) * 1000

        result = VADResult(
            label=label,
            confidence=confidence,
            frame_timestamp=frame.timestamp,
            processing_latency_ms=latency_ms,
        )

        # Track speech onset
        if label == VADLabel.SPEECH:
            self._consecutive_speech += 1
        else:
            self._consecutive_speech = 0

        await self._result_queue.put(result)
        return result

    def _classify(self, audio_float32: np.ndarray) -> tuple[VADLabel, float]:
        """Synchronous classification — runs in a thread pool executor."""
        if self._model is not None:
            return self._silero_classify(audio_float32)
        return self._energy_classify(audio_float32)

    def _silero_classify(self, audio: np.ndarray) -> tuple[VADLabel, float]:
        try:
            import torch  # type: ignore

            # Silero VAD strictly requires 512 samples at 16 kHz
            if len(audio) < 512:
                audio_512 = np.pad(audio, (0, 512 - len(audio)))
            elif len(audio) > 512:
                audio_512 = audio[:512]
            else:
                audio_512 = audio

            tensor = torch.from_numpy(audio_512).unsqueeze(0)
            with torch.no_grad():
                confidence = float(self._model(tensor, SAMPLE_RATE).item())
            if confidence >= self._threshold:
                return VADLabel.SPEECH, confidence
            return VADLabel.SILENCE, 1.0 - confidence
        except Exception as exc:  # noqa: BLE001
            logger.debug("Silero classify error: %s — falling back to energy", exc)
            return self._energy_classify(audio)

    @staticmethod
    def _energy_classify(audio: np.ndarray) -> tuple[VADLabel, float]:
        """
        Simple RMS energy heuristic — used when the Silero model is unavailable.
        Threshold tuned for a typical office microphone at 16 kHz.
        """
        rms = float(np.sqrt(np.mean(audio ** 2)))
        # Values below ~0.005 are effectively silence on a 16-bit stream
        if rms > 0.02:
            confidence = min(1.0, rms / 0.1)
            return VADLabel.SPEECH, confidence
        if rms > 0.005:
            return VADLabel.NOISE, 0.5
        return VADLabel.SILENCE, 1.0 - rms / 0.005

    # ------------------------------------------------------------------
    # Subscriber API
    # ------------------------------------------------------------------

    async def get_events(self) -> AsyncIterator[VADResult]:
        """Async generator yielding VADResults as they are produced."""
        while True:
            result = await self._result_queue.get()
            yield result

    @property
    def is_voice_onset(self) -> bool:
        """True once two consecutive SPEECH frames have been seen."""
        return self._consecutive_speech >= _SPEECH_ONSET_FRAMES

    def reset_onset(self) -> None:
        """Called by the Session Manager after an onset event is handled."""
        self._consecutive_speech = 0
