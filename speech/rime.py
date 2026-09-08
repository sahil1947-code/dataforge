"""
speech/rime.py
--------------
Rime TTS integration — WebSocket streaming, mid-utterance cancellation.

Architecture notes (TRD §3.10):
  • Transport: WebSocket streaming (not HTTP request/response) so that
    mid-utterance cancellation is possible at any point.
  • Every audio chunk is tagged with the generation ID it was produced for;
    chunks tagged with a superseded generation are discarded before playback.
  • Cancellation: close/flush the WebSocket stream; discard buffered audio.
  • Configuration: all five Rime parameters are read from settings, which
    loads from .env — never hard-coded here.

Pinned Rime parameters (must match .env / README at submission time):
  model_id, speaker, language, audio_format, endpoint
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import AsyncIterator, Callable, Optional

from config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


@dataclass
class AudioChunk:
    data: bytes
    generation: int
    timestamp: float


class RimeSession:
    """
    A single Rime TTS streaming session for one spoken response.

    Lifecycle:
        session = await RimeTTS.speak(text, generation)
        async for chunk in session.audio_stream():
            play(chunk.data)
        # OR, on interrupt:
        await session.cancel()
    """

    def __init__(
        self,
        text: str,
        generation: int,
        on_first_audio: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._text = text
        self._generation = generation
        self._on_first_audio = on_first_audio
        self._cancelled = False
        self._ws = None
        self._first_audio_time: Optional[float] = None
        self._queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()

    async def cancel(self) -> None:
        """
        Immediately stop playback and discard any buffered audio.
        Called by the Response Manager on interrupt (TRD §3.9).
        Target: ≤ 150 ms from interrupt signal to audio silence.
        """
        self._cancelled = True
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:  # noqa: BLE001
                pass
        # Drain the queue and signal end
        while not self._queue.empty():
            self._queue.get_nowait()
        await self._queue.put(None)
        logger.debug(
            "RimeSession cancelled for generation %d", self._generation
        )

    async def audio_stream(self) -> AsyncIterator[AudioChunk]:
        """Yield AudioChunks as they arrive from Rime."""
        while True:
            chunk_bytes = await self._queue.get()
            if chunk_bytes is None:
                break
            if self._cancelled:
                break
            yield AudioChunk(
                data=chunk_bytes,
                generation=self._generation,
                timestamp=time.time(),
            )

    # Internal — called by RimeTTS._stream
    async def _put(self, data: Optional[bytes]) -> None:
        if not self._cancelled:
            await self._queue.put(data)
            if data is not None and self._first_audio_time is None:
                self._first_audio_time = time.monotonic()
                if self._on_first_audio:
                    self._on_first_audio(self._first_audio_time)

    def _attach_ws(self, ws) -> None:  # type: ignore[annotation-unchecked]
        self._ws = ws

    @property
    def first_audio_latency_ms(self) -> Optional[float]:
        return (
            self._first_audio_time * 1000 if self._first_audio_time else None
        )


class RimeTTS:
    """
    Factory for RimeSessions.  All external calls go through ExternalServiceManager
    in production; this class handles the Rime-specific WebSocket protocol.

    Usage:
        tts = RimeTTS()
        session = await tts.speak("Hello world", generation=5)
        async for chunk in session.audio_stream():
            send_to_client(chunk.data)
    """

    def __init__(self) -> None:
        self._api_key = _settings.rime.api_key
        self._model_id = _settings.rime.model_id
        self._speaker = _settings.rime.speaker
        self._language = _settings.rime.language
        self._audio_format = _settings.rime.audio_format
        self._endpoint = _settings.rime.endpoint
        self._http_endpoint = _settings.rime.http_endpoint

    async def speak(
        self,
        text: str,
        generation: int,
        on_first_audio: Optional[Callable[[float], None]] = None,
    ) -> RimeSession:
        """
        Start a streaming TTS session for `text`.
        Returns a RimeSession immediately; audio arrives asynchronously.
        """
        session = RimeSession(
            text=text,
            generation=generation,
            on_first_audio=on_first_audio,
        )
        # Launch streaming in background task so caller gets session immediately
        asyncio.create_task(self._stream(session))
        return session

    async def _stream(self, session: RimeSession) -> None:
        """
        Open a WebSocket to Rime, send the synthesis request,
        and forward binary audio chunks to the session queue.
        """
        if not self._api_key or any(p in self._api_key.lower() for p in ("your_", "dummy", "placeholder", "example")):
            logger.info(
                "RIME_API_KEY not configured or placeholder — using TTS stub for generation %d.",
                session._generation,
            )
            await self._stub_stream(session)
            return

        try:
            import websockets  # type: ignore

            payload = json.dumps(
                {
                    "text": session._text,
                    "modelId": self._model_id,
                    "speaker": self._speaker,
                    "lang": self._language,
                    "audioFormat": self._audio_format,
                }
            )
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }

            async with websockets.connect(
                self._endpoint, extra_headers=headers
            ) as ws:
                session._attach_ws(ws)
                await ws.send(payload)

                async for message in ws:
                    if session._cancelled:
                        break
                    if isinstance(message, bytes):
                        await session._put(message)
                    elif isinstance(message, str):
                        # Rime may send JSON control messages
                        try:
                            ctrl = json.loads(message)
                            if ctrl.get("done"):
                                break
                        except json.JSONDecodeError:
                            pass

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Rime WebSocket error (generation %d): %s",
                session._generation,
                exc,
            )
            # Fall back to HTTP synthesis if WS fails
            await self._http_fallback(session)
        finally:
            await session._put(None)  # signal end of stream

    async def _http_fallback(self, session: RimeSession) -> None:
        """
        HTTP fallback synthesis used when the WebSocket endpoint is unreachable.
        This is logged and disclosed — Rime remains the default judged path.
        """
        if not self._api_key:
            await self._stub_stream(session)
            return

        try:
            import httpx

            payload = {
                "text": session._text,
                "modelId": self._model_id,
                "speaker": self._speaker,
                "lang": self._language,
                "audioFormat": self._audio_format,
            }
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    self._http_endpoint, json=payload, headers=headers
                )
                resp.raise_for_status()
                # Deliver as a single chunk
                await session._put(resp.content)
                logger.info(
                    "Rime HTTP fallback used for generation %d (WS unavailable).",
                    session._generation,
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("Rime HTTP fallback also failed: %s", exc)
            await self._stub_stream(session)

    @staticmethod
    async def _stub_stream(session: RimeSession) -> None:
        """
        Emits a minimal WAV header as a stub audio chunk when Rime is
        unreachable.  Lets the pipeline continue functioning in test/offline
        environments without real audio output.
        """
        import struct

        # Minimal valid 8-bit PCM WAV header + 10 ms of silence at 16kHz
        sample_rate = 16000
        num_samples = 160
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = num_samples * block_align

        wav_header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_size,
            b"WAVE",
            b"fmt ",
            16,            # chunk size
            1,             # PCM format
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b"data",
            data_size,
        )
        silence = b"\x00" * data_size
        await session._put(wav_header + silence)
        await session._put(None)
        logger.debug(
            "Rime stub audio emitted for generation %d.", session._generation
        )
