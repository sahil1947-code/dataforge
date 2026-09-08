"""
audio/capture.py
----------------
Microphone capture pipeline for RIME.

Reads raw PCM frames from the system microphone and pushes them into an
in-memory ring buffer.  The buffer has a fixed maximum duration and is
overwritten continuously — audio is never written to disk.

Architecture notes (TRD §3.1):
  • Frame size: 20 ms at 16 kHz = 320 samples
  • Ring buffer: configurable max duration (default 30 s)
  • Thread-safe: put_nowait into asyncio.Queue consumed by downstream tasks
  • On device disconnect: emits a DEVICE_DISCONNECTED event; does not crash
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE: int = 16_000          # Hz — matches Whisper and Silero VAD expectation
FRAME_DURATION_MS: int = 20        # milliseconds per PCM frame
SAMPLES_PER_FRAME: int = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 320
MAX_BUFFER_SECONDS: int = 30       # rolling in-memory ring buffer duration
MAX_BUFFER_FRAMES: int = int(MAX_BUFFER_SECONDS * 1000 / FRAME_DURATION_MS)


@dataclass
class AudioFrame:
    """A single 20 ms PCM frame of audio."""

    data: np.ndarray          # int16, shape (SAMPLES_PER_FRAME,)
    timestamp: float          # wall-clock time of capture (seconds since epoch)
    sample_rate: int = SAMPLE_RATE
    duration_ms: int = FRAME_DURATION_MS

    @property
    def float32(self) -> np.ndarray:
        """Normalised float32 version used by VAD and embedding models."""
        return self.data.astype(np.float32) / 32768.0


class RingBuffer:
    """
    Fixed-capacity deque of AudioFrames, overwritten oldest-first.
    Access is NOT thread-safe on its own; callers must hold the lock.
    """

    def __init__(self, max_frames: int = MAX_BUFFER_FRAMES) -> None:
        self._buf: collections.deque[AudioFrame] = collections.deque(
            maxlen=max_frames
        )
        self._lock = threading.Lock()

    def push(self, frame: AudioFrame) -> None:
        with self._lock:
            self._buf.append(frame)

    def snapshot(self, last_n_frames: Optional[int] = None) -> list[AudioFrame]:
        """Return a copy of the last N frames (or all if N is None)."""
        with self._lock:
            frames = list(self._buf)
        if last_n_frames is not None:
            return frames[-last_n_frames:]
        return frames

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class AudioCapture:
    """
    Streams microphone audio into a RingBuffer and an asyncio.Queue for
    downstream consumers (VAD, speaker embedding, STT).

    Usage:
        capture = AudioCapture()
        await capture.start()
        async for frame in capture.frames():
            ...
        await capture.stop()
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_duration_ms: int = FRAME_DURATION_MS,
        on_device_disconnect: Optional[Callable[[], None]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.samples_per_frame = int(sample_rate * frame_duration_ms / 1000)
        self._ring = RingBuffer()
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=500)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._on_device_disconnect = on_device_disconnect
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin capturing audio from the default microphone."""
        if self._running:
            return
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(
            target=self._capture_thread, daemon=True, name="rime-audio-capture"
        )
        self._thread.start()
        logger.info(
            "AudioCapture started — %d Hz, %d ms frames",
            self.sample_rate,
            self.frame_duration_ms,
        )

    async def stop(self) -> None:
        """Stop capturing and drain the queue."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._ring.clear()
        logger.info("AudioCapture stopped.")

    async def frames(self):
        """Async generator yielding AudioFrames as they arrive."""
        while self._running:
            try:
                frame = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                yield frame
            except asyncio.TimeoutError:
                continue

    def get_recent_audio(
        self, duration_ms: int = 2000
    ) -> np.ndarray:
        """
        Return a contiguous float32 PCM array covering the last `duration_ms`
        milliseconds from the ring buffer — used by the speaker embedder.
        """
        n_frames = max(1, int(duration_ms / self.frame_duration_ms))
        frames = self._ring.snapshot(last_n_frames=n_frames)
        if not frames:
            return np.zeros(self.samples_per_frame, dtype=np.float32)
        return np.concatenate([f.float32 for f in frames])

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_thread(self) -> None:
        """Blocking audio capture loop — runs in a background thread."""
        try:
            import pyaudio  # type: ignore

            pa = pyaudio.PyAudio()
            stream = pa.open(
                rate=self.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.samples_per_frame,
            )

            while self._running:
                try:
                    raw = stream.read(
                        self.samples_per_frame, exception_on_overflow=False
                    )
                    pcm = np.frombuffer(raw, dtype=np.int16)
                    frame = AudioFrame(data=pcm, timestamp=time.time())
                    self._ring.push(frame)
                    if self._loop and not self._queue.full():
                        self._loop.call_soon_threadsafe(
                            self._queue.put_nowait, frame
                        )
                except OSError as exc:
                    logger.warning("Audio read error: %s", exc)
                    break

            stream.stop_stream()
            stream.close()
            pa.terminate()

        except ImportError:
            logger.warning(
                "PyAudio not installed — running AudioCapture in simulation mode."
            )
            self._simulate_silence()
        except Exception as exc:  # noqa: BLE001
            logger.error("Audio device error: %s", exc)
            if self._on_device_disconnect:
                self._on_device_disconnect()

    def _simulate_silence(self) -> None:
        """
        Fallback used when PyAudio is unavailable (CI / headless environments).
        Emits silent frames at the correct cadence so the pipeline stays alive.
        """
        frame_interval = self.frame_duration_ms / 1000.0
        while self._running:
            silence = np.zeros(self.samples_per_frame, dtype=np.int16)
            frame = AudioFrame(data=silence, timestamp=time.time())
            self._ring.push(frame)
            if self._loop and not self._queue.full():
                self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)
            time.sleep(frame_interval)

    def inject_frame(self, frame: AudioFrame) -> None:
        """
        Inject a synthetic frame directly — used by the test harness and
        benchmarks to feed pre-recorded audio without a live microphone.
        """
        self._ring.push(frame)
        if self._loop and not self._queue.full():
            self._loop.call_soon_threadsafe(self._queue.put_nowait, frame)
