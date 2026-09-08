"""
api/ws_handler.py
-----------------
WebSocket session handler for RIME.

Handles the full realtime WebSocket lifecycle per API Specification §3:
  • Receives binary audio frames and control messages from the client
  • Runs VAD on every frame
  • Triggers speaker identification on speech onset
  • Delegates full pipeline turns to VoicePipeline
  • Forwards all server→client events (state transitions, transcripts,
    routing decisions, tool events, Rime audio) back to the client
"""

from __future__ import annotations

import asyncio
import json
import logging
import numpy as np
import time
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from audio.capture import AudioFrame, SAMPLE_RATE, SAMPLES_PER_FRAME
from audio.vad import VoiceActivityDetector, VADLabel
from conversation.session import Session, SessionManager
from conversation.state_machine import SessionState
from speaker.embedding import SpeakerEmbedder
from speaker.matcher import SpeakerMatcher, MatchConfidence
from speaker.enrollment import EnrollmentService
from speaker.profiles import ProfileService
from safety.privacy import PrivacyManager
from .pipeline import VoicePipeline

logger = logging.getLogger(__name__)

# Consecutive SPEECH frames needed before we treat this as a real utterance
_SPEECH_ONSET_FRAMES = 2
# Minimum frames of silence after speech before we consider the turn complete
_SILENCE_END_FRAMES = 15   # ~300 ms at 20 ms/frame


class WSSessionHandler:
    """
    Per-connection WebSocket handler.

    Instantiated once per client connection.  The handler owns:
      • the VAD instance (fresh per connection)
      • the speech buffer (accumulates frames for STT)
      • the Session object (created via SessionManager)
    """

    def __init__(
        self,
        ws: WebSocket,
        pipeline: VoicePipeline,
        session_manager: SessionManager,
        embedder: SpeakerEmbedder,
        stt,
    ) -> None:
        self._ws             = ws
        self._pipeline       = pipeline
        self._session_mgr    = session_manager
        self._embedder       = embedder
        self._stt            = stt
        self._vad            = VoiceActivityDetector()
        self._matcher        = SpeakerMatcher(embedder)
        self._enrollment     = EnrollmentService(embedder)
        self._profile_svc    = ProfileService()
        self._privacy        = PrivacyManager()

        self._session: Session | None = None
        self._speech_frames: list[AudioFrame] = []
        self._consecutive_speech: int = 0
        self._consecutive_silence: int = 0
        self._in_utterance: bool = False
        self._turn_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        await self._ws.accept()
        await self._vad.load()

        # Create session (speaker is unknown until first speech segment)
        self._session = await self._session_mgr.create()
        await self._send({"type": "session.created",
                          "conversation_id": self._session.conversation_id})

        try:
            while True:
                message = await self._ws.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if "bytes" in message and message["bytes"]:
                    await self._handle_frame(message["bytes"])
                elif "text" in message and message["text"]:
                    await self._handle_control(message["text"].encode("utf-8"))
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected: %s",
                        self._session.conversation_id if self._session else "?")
        except Exception as exc:
            logger.error("WebSocket error: %s", exc)
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # Frame dispatcher
    # ------------------------------------------------------------------

    async def _handle_frame(self, raw: bytes) -> None:
        # Control messages arrive as JSON text frames sent as bytes
        if raw[:1] == b"{":
            await self._handle_control(raw)
            return

        # Otherwise treat as raw PCM audio (16-bit mono)
        if len(raw) < 2:
            return
        pcm = np.frombuffer(raw, dtype=np.int16)
        # Process every frame segment (SAMPLES_PER_FRAME = 320 samples / 20ms)
        step = SAMPLES_PER_FRAME
        for i in range(0, len(pcm), step):
            chunk = pcm[i : i + step]
            if len(chunk) < step:
                chunk = np.pad(chunk, (0, step - len(chunk)))
            frame = AudioFrame(data=chunk, timestamp=time.time())
            logger.debug('Received PCM frame length %d', len(pcm))
            vad_result = await self._vad.process_frame(frame)
            await self._handle_vad(frame, vad_result)

    async def _handle_control(self, raw: bytes) -> None:
        try:
            msg = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        msg_type = msg.get("type", "")

        if msg_type == "control.interrupt":
            if self._session and self._session.state_machine.is_interruptible():
                await self._pipeline.handle_interrupt(self._session, self._send)

        elif msg_type in ("control.text_input", "text_input", "user.utterance"):
            text = msg.get("text", "").strip()
            if text and self._session:
                if not self._session.profile_id:
                    default_prof = "speaker_primary"
                    await self._session_mgr.attach_profile(self._session, default_prof)
                    await self._load_speaker_context(default_prof)
                    await self._send({
                        "type": "speaker.identified",
                        "profile_id": default_prof,
                        "confidence": 1.0,
                    })

                target_gen = self._session.current_generation + 1
                await self._send({
                    "type": "transcript.final",
                    "text": text,
                    "generation": target_gen,
                })
                if self._turn_task and not self._turn_task.done():
                    self._turn_task.cancel()
                self._turn_task = asyncio.create_task(
                    self._pipeline.handle_turn(self._session, text, self._send)
                )

        elif msg_type == "control.privacy_mode":
            if self._session:
                enabled = msg.get("enabled", False)
                if enabled:
                    await self._privacy.enable_privacy_mode(
                        self._session.profile_id, self._session
                    )
                else:
                    await self._privacy.disable_privacy_mode(
                        self._session.profile_id, self._session
                    )
                await self._send({"type": "privacy.updated",
                                  "privacy_mode": enabled})

        elif msg_type == "control.temp_session":
            if self._session:
                enabled = msg.get("enabled", False)
                await self._privacy.enable_temp_session(self._session) \
                    if enabled else \
                    await self._privacy.disable_temp_session(self._session)
                await self._send({"type": "session.mode_updated",
                                  "session_mode": "temporary" if enabled else "normal"})

    # ------------------------------------------------------------------
    # VAD logic
    # ------------------------------------------------------------------

    async def _handle_vad(self, frame: AudioFrame, result) -> None:
        if not self._session:
            return

        is_speech = result.label == VADLabel.SPEECH

        if is_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0
            self._speech_frames.append(frame)

            # Detect barge-in while Rime is speaking
            if self._session.state_machine.is_interruptible() and \
                    self._consecutive_speech >= _SPEECH_ONSET_FRAMES:
                await self._pipeline.handle_interrupt(self._session, self._send)
                self._in_utterance = True

            elif not self._in_utterance and \
                    self._consecutive_speech >= _SPEECH_ONSET_FRAMES:
                self._in_utterance = True
                await self._send({"type": "vad.speech_started"})

        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0

            if self._in_utterance and \
                    self._consecutive_silence >= _SILENCE_END_FRAMES:
                # End of utterance — process the accumulated speech
                self._in_utterance = False
                await self._process_utterance()
                self._speech_frames = []

    # ------------------------------------------------------------------
    # Utterance processing: speaker ID → STT → pipeline turn
    # ------------------------------------------------------------------

    async def _process_utterance(self) -> None:
        if not self._speech_frames or not self._session:
            return

        audio = np.concatenate([f.float32 for f in self._speech_frames])

        # --- Speaker identification ---
        if self._privacy.should_profile(self._session):
            match = await self._matcher.match(audio)

            if match.confidence == MatchConfidence.CONFIDENT:
                if self._session.profile_id != match.profile_id:
                    await self._session_mgr.attach_profile(
                        self._session, match.profile_id
                    )
                    await self._load_speaker_context(match.profile_id)
                await self._send({
                    "type": "speaker.identified",
                    "profile_id": match.profile_id,
                    "confidence": match.similarity,
                })

            elif match.confidence == MatchConfidence.TENTATIVE:
                await self._send({
                    "type": "speaker.tentative",
                    "profile_id": match.profile_id,
                    "confidence": match.similarity,
                })

            else:
                # New speaker — enroll silently
                new_profile_id = await self._enrollment.enroll_new(audio)
                await self._session_mgr.attach_profile(self._session, new_profile_id)
                await self._send({
                    "type": "speaker.unknown",
                    "temp_profile_id": new_profile_id,
                })
        else:
            await self._send({"type": "speaker.privacy_mode_active"})

        # --- STT ---
        t_stt = time.monotonic()
        transcript_result = await self._stt.transcribe(audio)
        stt_ms = (time.monotonic() - t_stt) * 1000

        if not transcript_result.text.strip() or \
                transcript_result.text.startswith("[STT_STUB"):
            return

        await self._send({
            "type": "transcript.final",
            "text": transcript_result.text,
            "generation": self._session.current_generation + 1,
        })

        # --- Pipeline turn ---
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()

        self._turn_task = asyncio.create_task(
            self._pipeline.handle_turn(
                self._session, transcript_result.text, self._send
            )
        )

    # ------------------------------------------------------------------
    # Context loader
    # ------------------------------------------------------------------

    async def _load_speaker_context(self, profile_id: str) -> None:
        profile = await self._profile_svc.get(profile_id)
        if profile:
            self._session.context["display_name"] = (
                profile.get("display_name") or profile_id
            )
            self._session.context["preferences"] = profile.get("preferences", {})
        await self._profile_svc.touch(profile_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send(self, msg: dict) -> None:
        try:
            await self._ws.send_text(json.dumps(msg))
        except Exception:
            pass  # Client disconnected; do not crash the handler

    async def _cleanup(self) -> None:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
        if self._session:
            await self._session_mgr.close(self._session)
