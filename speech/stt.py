"""
speech/stt.py
-------------
Local speech-to-text using a Whisper-family model.

Architecture notes (TRD §3.6):
  • No network dependency — model weights loaded from local cache
  • Default model size: "base" (~74 MB, good latency/accuracy balance)
  • Target latency: ≤ 300 ms on a short utterance (uncached)
  • Per-word timestamps preserved for barge-in alignment
  • Falls back to faster-whisper if available, then to standard whisper
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


@dataclass
class WordTimestamp:
    word: str
    start: float   # seconds from start of utterance
    end: float


@dataclass
class TranscriptResult:
    text: str
    language: str
    words: list[WordTimestamp] = field(default_factory=list)
    latency_ms: float = 0.0
    model_size: str = ""


class SpeechToText:
    """
    Wraps a local Whisper model.  Exposes a single async method:
        result = await stt.transcribe(audio_float32)

    The first call loads the model from disk (or downloads it once to
    ~/.cache/whisper).  Subsequent calls reuse the loaded model.
    """

    def __init__(self, model_size: Optional[str] = None) -> None:
        self._model_size = model_size or _settings.stt.model_size
        self._model = None
        self._backend: str = "none"
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def load(self) -> None:
        if self._loaded:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_model_blocking)
        self._loaded = True

    def _load_model_blocking(self) -> None:
        # Try faster-whisper first (significantly lower latency on CPU)
        try:
            from faster_whisper import WhisperModel  # type: ignore

            self._model = WhisperModel(
                self._model_size,
                device="cpu",
                compute_type="int8",
            )
            self._backend = "faster-whisper"
            logger.info(
                "faster-whisper loaded (size=%s).", self._model_size
            )
            return
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("faster-whisper load error: %s", exc)

        # Fall back to openai-whisper
        try:
            import whisper  # type: ignore

            self._model = whisper.load_model(self._model_size)
            self._backend = "openai-whisper"
            logger.info(
                "openai-whisper loaded (size=%s).", self._model_size
            )
            return
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("openai-whisper load error: %s", exc)

        logger.warning(
            "No Whisper implementation available — STT will return stub transcripts."
        )
        self._backend = "stub"

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------

    async def transcribe(self, audio_float32: np.ndarray) -> TranscriptResult:
        """
        Transcribe a VAD-bounded speech segment.
        `audio_float32` must be mono, 16 kHz, float32 in [-1, 1].
        """
        if not self._loaded:
            await self.load()

        t_start = time.monotonic()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, self._transcribe_blocking, audio_float32
        )
        result.latency_ms = (time.monotonic() - t_start) * 1000
        logger.debug(
            "STT (%s): '%s' in %.1f ms",
            self._backend,
            result.text[:60],
            result.latency_ms,
        )
        return result

    def _transcribe_blocking(self, audio: np.ndarray) -> TranscriptResult:
        if self._backend == "faster-whisper":
            return self._transcribe_faster(audio)
        if self._backend == "openai-whisper":
            return self._transcribe_openai(audio)
        return self._stub_transcribe(audio)

    def _transcribe_faster(self, audio: np.ndarray) -> TranscriptResult:
        segments, info = self._model.transcribe(
            audio,
            word_timestamps=True,
            vad_filter=False,   # we already ran VAD upstream
        )
        words: list[WordTimestamp] = []
        text_parts: list[str] = []
        for seg in segments:
            text_parts.append(seg.text)
            if seg.words:
                for w in seg.words:
                    words.append(WordTimestamp(w.word, w.start, w.end))

        return TranscriptResult(
            text=" ".join(text_parts).strip(),
            language=info.language,
            words=words,
            model_size=self._model_size,
        )

    def _transcribe_openai(self, audio: np.ndarray) -> TranscriptResult:
        result = self._model.transcribe(
            audio,
            word_timestamps=True,
            fp16=False,
        )
        words: list[WordTimestamp] = []
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                words.append(WordTimestamp(w["word"], w["start"], w["end"]))

        return TranscriptResult(
            text=result.get("text", "").strip(),
            language=result.get("language", "en"),
            words=words,
            model_size=self._model_size,
        )

    @staticmethod
    def _stub_transcribe(audio: np.ndarray) -> TranscriptResult:
        """
        Returns a clearly-labeled stub transcript when no model is available.
        Used in CI and test environments.
        """
        return TranscriptResult(
            text="[STT_STUB: no model loaded]",
            language="en",
            words=[],
            model_size="stub",
        )
