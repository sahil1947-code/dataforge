"""
conversation/interruption.py
-----------------------------
Response Manager — the correctness-critical component (TRD §3.9).

This is the single most important module in the system.

Responsibilities:
  1. Maintain current_generation per session via GenerationCounter.
  2. On detected user speech during SPEAKING or TOOL_RUNNING:
       a. Signal Rime to halt playback (interrupt fence).
       b. Increment current_generation.
       c. Mark all in-flight tool calls for the old generation as 'stale'.
  3. On any artifact arrival (tool result, LLM completion, audio chunk):
       check its generation tag before allowing it to reach Rime or
       before writing it to `messages`.
  4. Persist ONLY artifacts belonging to the generation the user
     actually heard, so conversation history reflects reality.

The fencing check is O(1) and executed on every artifact — no exceptions.

This module owns the `messages` and `tool_calls` status fields
(Backend Schema §4).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from database import execute, fetch_all, transaction
from .generations import GenerationCounter, TaggedArtifact, check_artifact
from .state_machine import SessionState, StateMachine

logger = logging.getLogger(__name__)


class ArtifactType(str, Enum):
    LLM_TOKEN    = "llm_token"
    TOOL_RESULT  = "tool_result"
    AUDIO_CHUNK  = "audio_chunk"
    MESSAGE      = "message"


@dataclass
class Artifact:
    payload: Any
    generation: int
    artifact_type: ArtifactType
    tool_call_id: Optional[str] = None


@dataclass
class InterruptEvent:
    session_id: str
    old_generation: int
    new_generation: int
    timestamp: float = field(default_factory=time.monotonic)
    artifacts_discarded: int = 0


InterruptCallback = Callable[[InterruptEvent], Coroutine]


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ResponseManager:
    """
    The authoritative gate between pipeline components and the user.

    Every artifact that could reach the user (TTS, persisted message, tool
    result written to the conversation) passes through allow() before going
    further.  If allow() returns False, the caller must discard the artifact.

    Usage:
        rm = ResponseManager(state_machine, generation_counter)

        # On new user turn:
        new_gen = await rm.on_new_turn()

        # On artifact arrival:
        if await rm.allow(artifact):
            await rime.speak(artifact.payload, artifact.generation)
        else:
            log_stale_discard(artifact)
    """

    def __init__(
        self,
        state_machine: StateMachine,
        generation_counter: GenerationCounter,
        conversation_id: str,
        rime_cancel_callback: Optional[Callable[[], Coroutine]] = None,
    ) -> None:
        self._sm = state_machine
        self._gen = generation_counter
        self._conversation_id = conversation_id
        self._rime_cancel = rime_cancel_callback
        self._interrupt_callbacks: list[InterruptCallback] = []
        self._active_rime_session = None
        self._discarded_count: int = 0
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Core fencing check — O(1)
    # ------------------------------------------------------------------

    async def allow(self, artifact: Artifact) -> bool:
        """
        The fencing gate.  Returns True if the artifact is for the current
        generation and should reach the user.  Returns False (stale) otherwise.

        This is the direct implementation of:
            if artifact.generation != session.current_generation:
                discard(artifact)
        """
        is_fresh = self._gen.is_current(artifact.generation)
        if not is_fresh:
            self._discarded_count += 1
            await self._mark_artifact_stale(artifact)
        return is_fresh

    # ------------------------------------------------------------------
    # Interrupt handling
    # ------------------------------------------------------------------

    async def on_interrupt(self, current_rime_session=None) -> int:
        """
        Called by the VAD/pipeline when user speech is detected during
        SPEAKING or TOOL_RUNNING.

        Actions (TRD §3.9):
          a. Signal Rime to halt playback within ≤ 150 ms.
          b. Increment current_generation.
          c. Mark in-flight tool calls for the old generation as 'stale'.

        Returns the new generation number.
        """
        async with self._lock:
            old_gen = self._gen.current
            t_interrupt = time.monotonic()

            # a. Cancel Rime immediately
            if current_rime_session is not None:
                await current_rime_session.cancel()
            elif self._rime_cancel is not None:
                await self._rime_cancel()

            # b. Increment generation
            new_gen = await self._gen.increment()

            # c. Mark stale tool calls in DB
            stale_count = await self._mark_stale_tool_calls(
                self._conversation_id, old_gen
            )

            # Transition state machine
            await self._sm.interrupt(generation=new_gen)

            interrupt_ms = (time.monotonic() - t_interrupt) * 1000
            logger.info(
                "INTERRUPT: gen %d → %d | stale tool calls: %d | latency: %.1f ms",
                old_gen,
                new_gen,
                stale_count,
                interrupt_ms,
            )

            event = InterruptEvent(
                session_id=self._conversation_id,
                old_generation=old_gen,
                new_generation=new_gen,
                artifacts_discarded=stale_count,
            )
            for cb in self._interrupt_callbacks:
                asyncio.create_task(cb(event))

            return new_gen

    async def on_new_turn(self) -> int:
        """
        Called at the start of a new user request (not necessarily an interrupt).
        Increments the generation counter so old pending results are fenced out.
        """
        return await self._gen.increment()

    # ------------------------------------------------------------------
    # Message persistence (only spoken = current generation)
    # ------------------------------------------------------------------

    async def persist_spoken_message(
        self,
        role: str,
        text: str,
        generation: int,
        sequence_number: int,
    ) -> Optional[str]:
        """
        Write a message to the DB only if it belongs to the current generation.
        Returns the message_id on success, None if stale.
        """
        if not self._gen.is_current(generation):
            logger.debug(
                "Stale message not persisted: gen=%d current=%d text='%.40s'",
                generation,
                self._gen.current,
                text,
            )
            return None

        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        await execute(
            """
            INSERT INTO messages
                (message_id, conversation_id, role, text, timestamp,
                 sequence_number, generation, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'spoken')
            """,
            (
                message_id,
                self._conversation_id,
                role,
                text,
                _utcnow(),
                sequence_number,
                generation,
            ),
        )
        return message_id

    async def register_tool_call(
        self, tool_call_id: str, tool_name: str, arguments: str, generation: int
    ) -> None:
        """Record a new tool invocation in the DB."""
        await execute(
            """
            INSERT INTO tool_calls
                (tool_call_id, conversation_id, tool_name, arguments,
                 status, generation, started_at)
            VALUES (?, ?, ?, ?, 'started', ?, ?)
            """,
            (
                tool_call_id,
                self._conversation_id,
                tool_name,
                arguments,
                generation,
                _utcnow(),
            ),
        )

    async def complete_tool_call(
        self, tool_call_id: str, generation: int, success: bool = True
    ) -> bool:
        """
        Mark a tool call as completed or stale depending on current generation.
        Returns True if it was current (result should be used), False if stale.
        """
        is_current = self._gen.is_current(generation)
        status = "completed" if (is_current and success) else "stale"
        await execute(
            """
            UPDATE tool_calls
               SET status = ?, completed_at = ?
             WHERE tool_call_id = ?
            """,
            (status, _utcnow(), tool_call_id),
        )
        return is_current

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_interrupt_event(self, callback: InterruptCallback) -> None:
        self._interrupt_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def current_generation(self) -> int:
        return self._gen.current

    @property
    def discarded_count(self) -> int:
        return self._discarded_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _mark_stale_tool_calls(
        self, conversation_id: str, old_generation: int
    ) -> int:
        """
        Mark all in-flight tool calls from `old_generation` as 'stale'.
        Returns the number of rows updated.
        """
        rows = await fetch_all(
            """
            SELECT tool_call_id FROM tool_calls
             WHERE conversation_id = ?
               AND generation = ?
               AND status = 'started'
            """,
            (conversation_id, old_generation),
        )
        if not rows:
            return 0

        for row in rows:
            await execute(
                """
                UPDATE tool_calls
                   SET status = 'stale', completed_at = ?
                 WHERE tool_call_id = ?
                """,
                (_utcnow(), row["tool_call_id"]),
            )
        return len(rows)

    async def _mark_artifact_stale(self, artifact: Artifact) -> None:
        """Update DB for stale tool results; log for stale LLM/audio artifacts."""
        if artifact.artifact_type == ArtifactType.TOOL_RESULT and artifact.tool_call_id:
            await execute(
                """
                UPDATE tool_calls
                   SET status = 'stale', completed_at = ?
                 WHERE tool_call_id = ? AND status = 'started'
                """,
                (_utcnow(), artifact.tool_call_id),
            )
        elif artifact.artifact_type == ArtifactType.MESSAGE:
            # Stale messages get written with status stale_discarded for audit trail
            message_id = f"msg_{uuid.uuid4().hex[:12]}"
            try:
                await execute(
                    """
                    INSERT INTO messages
                        (message_id, conversation_id, role, text, timestamp,
                         sequence_number, generation, status)
                    VALUES (?, ?, 'assistant', ?, ?, -1, ?, 'stale_discarded')
                    """,
                    (
                        message_id,
                        self._conversation_id,
                        str(artifact.payload)[:2000],
                        _utcnow(),
                        artifact.generation,
                    ),
                )
            except Exception:  # noqa: BLE001
                pass  # DB write failure must not break the interrupt path
