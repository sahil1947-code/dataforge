"""
conversation/state_machine.py
------------------------------
Session state machine for RIME.

States (TRD §3.5):
    SESSION_CREATED → LISTENING → THINKING → TOOL_RUNNING → SPEAKING → LISTENING
                                       ↑
                             (interrupt from any state)
                                       ↓
                                 INTERRUPTED → LISTENING

Key requirement: accepting an interrupt signal from ANY state must complete
within one processing tick, target ≤ 50 ms.
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    SESSION_CREATED = "SESSION_CREATED"
    LISTENING       = "LISTENING"
    THINKING        = "THINKING"
    TOOL_RUNNING    = "TOOL_RUNNING"
    SPEAKING        = "SPEAKING"
    INTERRUPTED     = "INTERRUPTED"
    IDLE            = "IDLE"            # device disconnected / paused
    PROFILE_SELECTION = "PROFILE_SELECTION"  # awaiting speaker confirmation


# Valid transitions — every state can be interrupted
_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.SESSION_CREATED: {
        SessionState.LISTENING,
        SessionState.INTERRUPTED,
        SessionState.IDLE,
    },
    SessionState.LISTENING: {
        SessionState.THINKING,
        SessionState.INTERRUPTED,
        SessionState.IDLE,
        SessionState.PROFILE_SELECTION,
    },
    SessionState.THINKING: {
        SessionState.TOOL_RUNNING,
        SessionState.SPEAKING,
        SessionState.INTERRUPTED,
        SessionState.IDLE,
    },
    SessionState.TOOL_RUNNING: {
        SessionState.SPEAKING,
        SessionState.THINKING,
        SessionState.INTERRUPTED,
        SessionState.IDLE,
    },
    SessionState.SPEAKING: {
        SessionState.LISTENING,
        SessionState.INTERRUPTED,
        SessionState.IDLE,
    },
    SessionState.INTERRUPTED: {
        SessionState.LISTENING,
        SessionState.IDLE,
    },
    SessionState.IDLE: {
        SessionState.LISTENING,
        SessionState.SESSION_CREATED,
    },
    SessionState.PROFILE_SELECTION: {
        SessionState.LISTENING,
        SessionState.INTERRUPTED,
        SessionState.IDLE,
    },
}


TransitionCallback = Callable[[SessionState, SessionState, int], None]


class StateMachine:
    """
    Thread-safe state machine for a single RIME session.

    Transition callbacks are called synchronously within transition().
    Async callbacks can be scheduled via asyncio.create_task() from
    the callback if needed.
    """

    def __init__(self, initial: SessionState = SessionState.SESSION_CREATED) -> None:
        self._state = initial
        self._lock = asyncio.Lock()
        self._callbacks: list[TransitionCallback] = []
        self._history: list[tuple[SessionState, SessionState, float]] = []

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def state(self) -> SessionState:
        return self._state

    def is_interruptible(self) -> bool:
        """True for all states where a barge-in signal should trigger an interrupt."""
        return self._state in {
            SessionState.SPEAKING,
            SessionState.TOOL_RUNNING,
            SessionState.THINKING,
        }

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    async def transition(
        self,
        new_state: SessionState,
        generation: int = 0,
        force: bool = False,
    ) -> bool:
        """
        Transition to `new_state`.  Returns True on success, False if the
        transition is not allowed (and `force` is False).

        The transition is O(1) and completes within a single asyncio tick
        (≤ 50 ms target, TRD §3.5).
        """
        async with self._lock:
            allowed = _TRANSITIONS.get(self._state, set())
            if new_state not in allowed and not force:
                logger.warning(
                    "Invalid state transition %s → %s (generation %d) — ignored.",
                    self._state.value,
                    new_state.value,
                    generation,
                )
                return False

            old_state = self._state
            self._state = new_state
            self._history.append((old_state, new_state, time.monotonic()))

            logger.info(
                "State: %s → %s  [gen=%d]",
                old_state.value,
                new_state.value,
                generation,
            )
            for cb in self._callbacks:
                try:
                    cb(old_state, new_state, generation)
                except Exception as exc:  # noqa: BLE001
                    logger.error("State transition callback error: %s", exc)

        return True

    async def interrupt(self, generation: int) -> bool:
        """
        Fast path for barge-in — transitions to INTERRUPTED from any
        interruptible state.  Target: ≤ 50 ms.
        """
        return await self.transition(
            SessionState.INTERRUPTED, generation=generation, force=True
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_transition(self, callback: TransitionCallback) -> None:
        """Register a callback invoked on every state change."""
        self._callbacks.append(callback)

    def remove_callback(self, callback: TransitionCallback) -> None:
        self._callbacks.remove(callback)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def recent_history(self, n: int = 10) -> list[dict]:
        return [
            {"from": h[0].value, "to": h[1].value, "at": h[2]}
            for h in self._history[-n:]
        ]
