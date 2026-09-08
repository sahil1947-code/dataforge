"""
api/pipeline.py
---------------
The live voice pipeline — wires every module together into one request loop.

Called by the WebSocket handler for each incoming audio frame.  This is the
single place where VAD → Speaker ID → STT → Intent Router → Tool/Agent →
Response Manager → Rime TTS are chained in the correct order.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Callable, Coroutine, Optional

from audio.vad import VADLabel, VADResult
from conversation.session import Session
from conversation.state_machine import SessionState
from conversation.interruption import Artifact, ArtifactType
from routing.intent import IntentRouter, RouteType
from routing.local import LocalRouter
from tools.registry import get_registry
from agent.agent import ReasoningAgent
from speech.rime import RimeTTS
from speech.stt import SpeechToText
from memory.memories import MemoryService
from safety.emergency import EmergencyDetector
from safety.privacy import PrivacyManager
from observability.traces import TraceCollector, PipelineTrace
from observability.logger import log_pipeline_event, log_interrupt_event
from database import fetch_all, rows_to_dicts

logger = logging.getLogger(__name__)

# Type alias for the WebSocket send callback
SendCallback = Callable[[dict], Coroutine]


class VoicePipeline:
    """
    Stateless pipeline processor.  One instance is shared across all sessions;
    per-session state lives entirely in the Session object.
    """

    def __init__(
        self,
        stt: SpeechToText,
        rime: RimeTTS,
        agent: ReasoningAgent,
        trace_collector: TraceCollector,
    ) -> None:
        self._stt     = stt
        self._rime    = rime
        self._agent   = agent
        self._traces  = trace_collector
        self._router  = IntentRouter()
        self._local   = LocalRouter()
        self._memory  = MemoryService()
        self._safety  = EmergencyDetector()
        self._privacy = PrivacyManager()

    # ------------------------------------------------------------------
    # Barge-in / interrupt (called the moment VAD fires during SPEAKING)
    # ------------------------------------------------------------------

    async def handle_interrupt(
        self,
        session: Session,
        send: SendCallback,
    ) -> None:
        t0 = time.monotonic()
        old_gen = session.current_generation

        new_gen = await session.response_manager.on_interrupt(
            current_rime_session=session.active_rime_session
        )
        session.active_rime_session = None

        latency_ms = (time.monotonic() - t0) * 1000
        log_interrupt_event(
            session_id=session.conversation_id,
            old_generation=old_gen,
            new_generation=new_gen,
            artifacts_discarded=session.response_manager.discarded_count,
            interrupt_latency_ms=latency_ms,
        )

        await send({
            "type": "state.transition",
            "from": SessionState.SPEAKING.value,
            "to": SessionState.INTERRUPTED.value,
            "generation": new_gen,
        })
        await send({
            "type": "rime.stopped",
            "generation": old_gen,
            "reason": "interrupted",
        })

    # ------------------------------------------------------------------
    # Full turn: transcript → routing → tool/agent → Rime
    # ------------------------------------------------------------------

    async def handle_turn(
        self,
        session: Session,
        transcript: str,
        send: SendCallback,
    ) -> None:
        generation = await session.response_manager.on_new_turn()
        trace = self._traces.start(
            session.conversation_id, generation, session.profile_id
        )

        try:
            await self._run_turn(session, transcript, generation, send, trace)
        except Exception as exc:
            logger.error("Pipeline turn error (gen=%d): %s", generation, exc)
            await send({"type": "error", "code": "PIPELINE_ERROR", "message": str(exc)})
        finally:
            await self._traces.finish(trace)

    async def _run_turn(
        self,
        session: Session,
        transcript: str,
        generation: int,
        send: SendCallback,
        trace: PipelineTrace,
    ) -> None:
        # --- State: THINKING ---
        await session.state_machine.transition(SessionState.THINKING, generation)
        await send({"type": "state.transition", "from": "LISTENING",
                    "to": "THINKING", "generation": generation})

        # --- Emergency check ---
        emergency = await self._safety.classify(transcript, session.profile_id)
        if emergency and emergency.confidence >= self._safety.threshold:
            await send({
                "type": "emergency.detected",
                "event_id": emergency.event_id,
                "confidence": emergency.confidence,
                "trigger_type": emergency.trigger_type,
            })
            # Do not continue with normal routing — wait for confirmation
            await session.state_machine.transition(SessionState.LISTENING, generation)
            return

        # --- Intent routing ---
        decision = await self._router.route(
            transcript, generation, session.profile_id
        )
        await send({
            "type": "routing.decision",
            "route": decision.route.value,
            "tool": decision.tool_name,
            "reason": decision.reason,
            "generation": generation,
        })

        # --- Tool execution (if routed to a tool) ---
        tool_result = None
        if decision.route in (RouteType.LOCAL_TOOL, RouteType.LIVE_DATA):
            # State: TOOL_RUNNING
            await session.state_machine.transition(SessionState.TOOL_RUNNING, generation)
            tool_call_id = f"tc_{uuid.uuid4().hex[:10]}"

            await send({
                "type": "tool.started",
                "tool_call_id": tool_call_id,
                "tool_name": decision.tool_name,
                "generation": generation,
            })
            await session.response_manager.register_tool_call(
                tool_call_id, decision.tool_name,
                json.dumps(decision.args), generation
            )

            t_tool = time.monotonic()
            raw_result = await get_registry().run(
                decision.tool_name, decision.args,
                generation, session.profile_id
            )
            trace.tool_ms = (time.monotonic() - t_tool) * 1000
            trace.tool_cached = raw_result.from_cache

            # Fencing check — discard if generation was superseded during tool call
            is_current = await session.response_manager.complete_tool_call(
                tool_call_id, generation, raw_result.success
            )
            stale = not is_current

            await send({
                "type": "tool.completed",
                "tool_call_id": tool_call_id,
                "generation": generation,
                "stale": stale,
                "success": raw_result.success and not stale,
            })
            log_pipeline_event("tool_complete", session.conversation_id,
                               generation, trace.tool_ms, raw_result.from_cache)

            if stale:
                logger.info("Tool result discarded (stale) gen=%d", generation)
                return  # Session is already on a newer generation

            tool_result = {
                "tool_name": decision.tool_name,
                "success": raw_result.success,
                "output": raw_result.output,
                "error": raw_result.error,
                "network_used": raw_result.network_used,
                "from_cache": raw_result.from_cache,
            }

        # --- Memory recall context injection ---
        memory_context: list[dict] = []
        if session.profile_id:
            memory_context = await self._memory.get_context_for_session(
                session.profile_id, limit=10
            )

        # --- Conversation history ---
        history = await self._load_history(session.conversation_id)

        # --- Agent / LLM reasoning ---
        speaker_name = session.context.get("display_name", "User")

        # State: SPEAKING
        await session.state_machine.transition(SessionState.SPEAKING, generation)

        t_rime_start: list[float] = []

        def on_first_audio(t: float) -> None:
            trace.rime_first_audio_ms = (t - trace.started_at) * 1000
            t_rime_start.append(t)

        rime_session = await self._rime.speak("", generation, on_first_audio)
        session.active_rime_session = rime_session

        await send({"type": "rime.started", "generation": generation})

        full_response_parts: list[str] = []
        seq = session.next_sequence()

        async for sentence in self._agent.stream(
            transcript, memory_context, history,
            generation, tool_result, speaker_name
        ):
            # Fence check on every sentence before it reaches Rime
            artifact = Artifact(
                payload=sentence,
                generation=generation,
                artifact_type=ArtifactType.LLM_TOKEN,
            )
            if not await session.response_manager.allow(artifact):
                logger.debug("Stale sentence discarded gen=%d", generation)
                break

            # Accumulate for full transcript
            full_response_parts.append(sentence)

            # Stream sentence text to UI immediately
            await send({
                "type": "transcript.assistant",
                "text": sentence,
                "generation": generation,
                "partial": True,
            })

            # Send new Rime session for this sentence
            rime_sentence_session = await self._rime.speak(
                sentence, generation, on_first_audio if not t_rime_start else None
            )
            session.active_rime_session = rime_sentence_session

            async for chunk in rime_sentence_session.audio_stream():
                await send({
                    "type": "rime.audio_chunk",
                    "generation": chunk.generation,
                    "size": len(chunk.data),
                })
        full_response = " ".join(full_response_parts)

        # Emit spoken response text for the companion UI transcript view
        if full_response:
            await send({
                "type": "transcript.assistant",
                "text": full_response,
                "generation": generation,
            })
            await session.response_manager.persist_spoken_message(
                role="assistant",
                text=full_response,
                generation=generation,
                sequence_number=seq,
            )
            # Persist user message too
            await session.response_manager.persist_spoken_message(
                role="user",
                text=transcript,
                generation=generation,
                sequence_number=seq - 1 if seq > 1 else 0,
            )

        await send({
            "type": "rime.stopped",
            "generation": generation,
            "reason": "completed",
        })
        await session.state_machine.transition(SessionState.LISTENING, generation)
        session.active_rime_session = None

        trace.total_perceived_ms = (time.monotonic() - trace.started_at) * 1000
        log_pipeline_event("turn_complete", session.conversation_id,
                           generation, trace.total_perceived_ms)

    # ------------------------------------------------------------------
    # Helper: load recent conversation history from DB
    # ------------------------------------------------------------------

    async def _load_history(self, conversation_id: str, limit: int = 10) -> list[dict]:
        rows = await fetch_all(
            """
            SELECT role, text FROM messages
             WHERE conversation_id = ? AND status = 'spoken'
             ORDER BY sequence_number DESC LIMIT ?
            """,
            (conversation_id, limit),
        )
        return list(reversed(rows_to_dicts(rows)))
