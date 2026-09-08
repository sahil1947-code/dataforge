"""
conversation/session.py
------------------------
Session object and SessionManager for RIME.

A Session binds together:
  • a profile_id (who is speaking)
  • a conversation_id (this interaction's DB row)
  • a StateMachine (the current pipeline state)
  • a GenerationCounter + ResponseManager (interruption correctness)
  • the active Rime TTS session (for cancellation)
  • privacy / session-mode flags

SessionManager creates, retrieves, and tears down sessions.
Owned tables: conversations  (Backend Schema §4)
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from database import execute, fetch_one, row_to_dict
from .state_machine import SessionState, StateMachine
from .generations import GenerationCounter
from .interruption import ResponseManager

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_conversation_id() -> str:
    return f"conv_{uuid.uuid4().hex[:16]}"


@dataclass
class Session:
    """
    Runtime state for an active conversation.  One Session per connected
    WebSocket client.  The Session is the single source of truth for the
    conversation's pipeline state and correctness guarantees.
    """

    conversation_id: str
    profile_id: Optional[str]        # None if speaker is unidentified
    session_mode: str                 # "normal" | "temporary"
    state_machine: StateMachine
    generation_counter: GenerationCounter
    response_manager: ResponseManager
    active_rime_session: Any = None   # speech.rime.RimeSession | None
    context: dict[str, Any] = field(default_factory=dict)
    sequence_number: int = 0          # increments with each new message
    privacy_mode: bool = False
    started_at: str = field(default_factory=_utcnow)

    def next_sequence(self) -> int:
        self.sequence_number += 1
        return self.sequence_number

    @property
    def current_generation(self) -> int:
        return self.generation_counter.current

    @property
    def state(self) -> SessionState:
        return self.state_machine.state

    def is_temporary(self) -> bool:
        return self.session_mode == "temporary"


class SessionManager:
    """
    Creates and manages Session objects.

    One session per active WebSocket connection.  Sessions are kept in
    memory in `_sessions`; the DB row in `conversations` is written on
    creation and updated on close.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def create(
        self,
        profile_id: Optional[str] = None,
        session_mode: str = "normal",
    ) -> Session:
        """
        Create a new Session and persist a conversations row.
        Returns the Session immediately — it starts in SESSION_CREATED state.
        """
        conversation_id = _new_conversation_id()
        now = _utcnow()

        # Persist conversation row
        await execute(
            """
            INSERT INTO conversations
                (conversation_id, profile_id, started_at, session_mode)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, profile_id, now, session_mode),
        )

        sm = StateMachine(initial=SessionState.SESSION_CREATED)
        gen = GenerationCounter(initial=0)
        rm = ResponseManager(
            state_machine=sm,
            generation_counter=gen,
            conversation_id=conversation_id,
        )

        session = Session(
            conversation_id=conversation_id,
            profile_id=profile_id,
            session_mode=session_mode,
            state_machine=sm,
            generation_counter=gen,
            response_manager=rm,
            started_at=now,
        )

        async with self._lock:
            self._sessions[conversation_id] = session

        await sm.transition(SessionState.LISTENING, generation=0)
        logger.info(
            "Session created: %s (profile=%s, mode=%s)",
            conversation_id,
            profile_id,
            session_mode,
        )
        return session

    async def get(self, conversation_id: str) -> Optional[Session]:
        async with self._lock:
            return self._sessions.get(conversation_id)

    async def attach_profile(self, session: Session, profile_id: str) -> None:
        """Bind a recognised profile to an already-running session."""
        session.profile_id = profile_id
        now = _utcnow()
        await execute(
            """
            INSERT OR IGNORE INTO profiles
                (profile_id, display_name, created_at, last_seen, status, preferences_json)
            VALUES (?, ?, ?, ?, 'active', '{}')
            """,
            (profile_id, profile_id, now, now),
        )
        await execute(
            "UPDATE conversations SET profile_id = ? WHERE conversation_id = ?",
            (profile_id, session.conversation_id),
        )
        logger.info(
            "Profile %s attached to session %s", profile_id, session.conversation_id
        )

    async def close(self, session: Session) -> None:
        """
        End the session: update the DB row, clean up temporary data if needed.
        """
        ended_at = _utcnow()
        await execute(
            "UPDATE conversations SET ended_at = ? WHERE conversation_id = ?",
            (ended_at, session.conversation_id),
        )

        if session.is_temporary():
            await self._delete_temporary_session(session)

        async with self._lock:
            self._sessions.pop(session.conversation_id, None)

        logger.info(
            "Session closed: %s (temporary=%s)",
            session.conversation_id,
            session.is_temporary(),
        )

    async def set_privacy_mode(self, session: Session, enabled: bool) -> None:
        session.privacy_mode = enabled
        logger.info(
            "Privacy mode %s for session %s",
            "enabled" if enabled else "disabled",
            session.conversation_id,
        )

    async def set_temp_mode(self, session: Session, enabled: bool) -> None:
        mode = "temporary" if enabled else "normal"
        session.session_mode = mode
        await execute(
            "UPDATE conversations SET session_mode = ? WHERE conversation_id = ?",
            (mode, session.conversation_id),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _delete_temporary_session(self, session: Session) -> None:
        """Delete all data for a temporary session (App Flow §9)."""
        cid = session.conversation_id
        await execute(
            "DELETE FROM tool_calls WHERE conversation_id = ?", (cid,)
        )
        await execute(
            "DELETE FROM messages WHERE conversation_id = ?", (cid,)
        )
        await execute(
            "DELETE FROM conversations WHERE conversation_id = ?", (cid,)
        )
        logger.info(
            "Temporary session %s data purged from DB.", cid
        )

    @property
    def active_count(self) -> int:
        return len(self._sessions)
