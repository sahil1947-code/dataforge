"""
safety/privacy.py
-----------------
Privacy mode and temporary session management.

Implements PRD US-09 and US-10 (App Flow §8, §9):
  • Privacy mode: disables speaker profiling, persistent memory writes,
    and audio retention for the current session.
  • Temporary session: session data is destroyed at session end (handled
    by conversation/session.py SessionManager._delete_temporary_session).

Privacy state is tracked per-session in the Session object AND in the
profiles table (profiles.status = 'privacy_mode') for persistence across
reconnections.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from database import execute, fetch_one
from speaker.profiles import ProfileService

logger = logging.getLogger(__name__)


@dataclass
class PrivacyStatus:
    background_listening: bool
    voice_identification: bool
    persistent_memory: bool
    cloud_requests: bool
    audio_storage: bool
    session_mode: str        # "normal" | "temporary"
    network_blocked: bool


class PrivacyManager:
    """
    Manages privacy mode toggles for a session and optionally persists them
    to the profile row.

    Every toggle change can be triggered equally by voice or UI — both paths
    call the same methods here (UI/UX Spec §6.4).
    """

    def __init__(self) -> None:
        self._profile_service = ProfileService()

    async def enable_privacy_mode(
        self,
        profile_id: Optional[str],
        session,
    ) -> PrivacyStatus:
        """
        Enable privacy mode for the current session.
        Effects (App Flow §8):
          - Voice profiling disabled (no new embeddings stored)
          - Persistent memory writes blocked
          - Audio retention beyond the rolling buffer blocked
        """
        session.privacy_mode = True
        if profile_id:
            await self._profile_service.set_status(profile_id, "privacy_mode")
        logger.info("Privacy mode ENABLED (profile=%s)", profile_id)
        return self.get_status(session)

    async def disable_privacy_mode(
        self,
        profile_id: Optional[str],
        session,
    ) -> PrivacyStatus:
        """Exit privacy mode and restore normal operation."""
        session.privacy_mode = False
        if profile_id:
            await self._profile_service.set_status(profile_id, "active")
        logger.info("Privacy mode DISABLED (profile=%s)", profile_id)
        return self.get_status(session)

    async def enable_temp_session(self, session) -> None:
        """Mark the current session as temporary (App Flow §9)."""
        from conversation.session import SessionManager
        manager = SessionManager()
        await manager.set_temp_mode(session, enabled=True)
        logger.info(
            "Temporary session enabled for conversation %s", session.conversation_id
        )

    async def disable_temp_session(self, session) -> None:
        from conversation.session import SessionManager
        manager = SessionManager()
        await manager.set_temp_mode(session, enabled=False)

    def get_status(self, session) -> PrivacyStatus:
        """Return the current privacy status for a session."""
        is_privacy = session.privacy_mode
        is_temp = session.is_temporary()
        return PrivacyStatus(
            background_listening=True,           # always on — VAD must run
            voice_identification=not is_privacy,
            persistent_memory=not is_privacy,
            cloud_requests=True,                 # controlled per-tool, not globally here
            audio_storage=False,                 # audio never stored by default (TRD §3.1)
            session_mode=session.session_mode,
            network_blocked=False,               # set to True if user explicitly blocks cloud
        )

    def should_profile(self, session) -> bool:
        """Returns False when speaker profiling must be suppressed (privacy mode)."""
        return not session.privacy_mode

    def should_persist_memory(self, session) -> bool:
        """Returns False when memory writes must be suppressed (privacy mode)."""
        return not session.privacy_mode
