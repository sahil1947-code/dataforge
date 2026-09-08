"""
safety/confirmations.py
------------------------
Confirmation service for destructive and high-permission operations.

Architecture notes (TRD §3.4, UI/UX Spec §4.3, App Flow §7):
  • All operations at permission level ≥ 2 require explicit user confirmation
    before execution.
  • Confirmations are tracked in-memory per session — they are not persisted
    (a crashed session means the confirmation must be re-requested).
  • A pending confirmation has a TTL of 60 seconds; after that it expires and
    the user must confirm again.
  • The "why did you do that?" transparency feature also uses this service
    to surface routing decisions (App Flow §14).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)

CONFIRMATION_TTL_SECONDS: float = 60.0


class ConfirmationOutcome(str, Enum):
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"
    EXPIRED   = "expired"


@dataclass
class ConfirmationRequest:
    confirmation_id: str
    operation: str
    description: str                     # human-readable, spoken by Rime
    args: dict[str, Any]
    profile_id: Optional[str]
    created_at: float = field(default_factory=time.monotonic)
    outcome: Optional[ConfirmationOutcome] = None
    _future: Optional[asyncio.Future] = field(default=None, repr=False)

    def is_expired(self) -> bool:
        return time.monotonic() - self.created_at > CONFIRMATION_TTL_SECONDS


class ConfirmationService:
    """
    Manages pending confirmations for a single session.

    Usage:
        svc = ConfirmationService()
        req = await svc.request("forget_profile", "Delete all your history?", args, pid)
        outcome = await svc.wait(req.confirmation_id, timeout=60)
        if outcome == ConfirmationOutcome.CONFIRMED:
            ...execute...
    """

    def __init__(self) -> None:
        self._pending: dict[str, ConfirmationRequest] = {}

    async def request(
        self,
        operation: str,
        description: str,
        args: dict[str, Any],
        profile_id: Optional[str] = None,
    ) -> ConfirmationRequest:
        """
        Create a pending confirmation and return the request object.
        The caller should speak `description` via Rime and then call wait().
        """
        cid = f"conf_{uuid.uuid4().hex[:10]}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        req = ConfirmationRequest(
            confirmation_id=cid,
            operation=operation,
            description=description,
            args=args,
            profile_id=profile_id,
            _future=future,
        )
        self._pending[cid] = req
        logger.info(
            "Confirmation requested: id=%s op='%s' profile=%s",
            cid, operation, profile_id,
        )
        return req

    async def wait(
        self,
        confirmation_id: str,
        timeout: float = CONFIRMATION_TTL_SECONDS,
    ) -> ConfirmationOutcome:
        """
        Await the user's response to a confirmation.
        Returns the outcome (CONFIRMED | CANCELLED | EXPIRED).
        """
        req = self._pending.get(confirmation_id)
        if req is None:
            return ConfirmationOutcome.EXPIRED
        if req.is_expired():
            self._expire(req)
            return ConfirmationOutcome.EXPIRED
        try:
            outcome = await asyncio.wait_for(req._future, timeout=timeout)
            req.outcome = outcome
            return outcome
        except asyncio.TimeoutError:
            self._expire(req)
            return ConfirmationOutcome.EXPIRED
        finally:
            self._pending.pop(confirmation_id, None)

    def resolve(
        self,
        confirmation_id: str,
        confirmed: bool,
    ) -> bool:
        """
        Called when the user speaks "yes" / "no" or taps the UI button.
        Returns True if the confirmation was still pending and has been resolved.
        """
        req = self._pending.get(confirmation_id)
        if req is None or req._future is None or req._future.done():
            return False
        outcome = ConfirmationOutcome.CONFIRMED if confirmed else ConfirmationOutcome.CANCELLED
        req.outcome = outcome
        req._future.set_result(outcome)
        logger.info(
            "Confirmation %s resolved: %s", confirmation_id, outcome.value
        )
        return True

    def get_pending(self, confirmation_id: str) -> Optional[ConfirmationRequest]:
        req = self._pending.get(confirmation_id)
        if req and req.is_expired():
            self._expire(req)
            return None
        return req

    def list_pending(self) -> list[ConfirmationRequest]:
        # Expire stale ones first
        stale = [cid for cid, r in self._pending.items() if r.is_expired()]
        for cid in stale:
            self._expire(self._pending[cid])
        return list(self._pending.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _expire(self, req: ConfirmationRequest) -> None:
        if req._future and not req._future.done():
            req._future.set_result(ConfirmationOutcome.EXPIRED)
        req.outcome = ConfirmationOutcome.EXPIRED
        self._pending.pop(req.confirmation_id, None)
        logger.info("Confirmation %s expired.", req.confirmation_id)


# ---------------------------------------------------------------------------
# Confirmation prompt templates spoken by Rime
# ---------------------------------------------------------------------------

CONFIRMATION_PROMPTS: dict[str, str] = {
    "forget_profile": (
        "This will permanently delete everything I know about you — "
        "all memories, tasks, and conversation history. "
        "There is no way to undo this. "
        "Please say yes to confirm, or no to cancel."
    ),
    "delete_memory_category": (
        "This will delete all memories in that category. "
        "Say yes to confirm or no to cancel."
    ),
    "delete_task": (
        "This will cancel the task permanently. "
        "Say yes to confirm or no to cancel."
    ),
    "emergency_activate": (
        "I detected a possible distress signal. "
        "Say yes to activate emergency mode, or no if you are safe."
    ),
}


def get_confirmation_prompt(operation: str, fallback: str = "") -> str:
    return CONFIRMATION_PROMPTS.get(operation, fallback or f"Confirm operation: {operation}?")
