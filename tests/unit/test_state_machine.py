"""
tests/unit/test_state_machine.py
---------------------------------
Unit tests for the session state machine.
Verifies all valid transitions, interrupt from any interruptible state,
and invalid transition rejection.
"""

import asyncio
import time
import pytest
from conversation.state_machine import StateMachine, SessionState


@pytest.mark.asyncio
async def test_initial_state():
    sm = StateMachine()
    assert sm.state == SessionState.SESSION_CREATED


@pytest.mark.asyncio
async def test_created_to_listening():
    sm = StateMachine()
    ok = await sm.transition(SessionState.LISTENING)
    assert ok is True
    assert sm.state == SessionState.LISTENING


@pytest.mark.asyncio
async def test_listening_to_thinking():
    sm = StateMachine()
    await sm.transition(SessionState.LISTENING)
    ok = await sm.transition(SessionState.THINKING)
    assert ok is True
    assert sm.state == SessionState.THINKING


@pytest.mark.asyncio
async def test_thinking_to_tool_running():
    sm = StateMachine()
    await sm.transition(SessionState.LISTENING)
    await sm.transition(SessionState.THINKING)
    ok = await sm.transition(SessionState.TOOL_RUNNING)
    assert ok
    assert sm.state == SessionState.TOOL_RUNNING


@pytest.mark.asyncio
async def test_speaking_to_listening():
    sm = StateMachine()
    await sm.transition(SessionState.LISTENING)
    await sm.transition(SessionState.THINKING)
    await sm.transition(SessionState.SPEAKING)
    ok = await sm.transition(SessionState.LISTENING)
    assert ok
    assert sm.state == SessionState.LISTENING


@pytest.mark.asyncio
async def test_interrupt_from_speaking():
    sm = StateMachine()
    await sm.transition(SessionState.LISTENING)
    await sm.transition(SessionState.THINKING)
    await sm.transition(SessionState.SPEAKING)
    ok = await sm.interrupt(generation=2)
    assert ok is True
    assert sm.state == SessionState.INTERRUPTED


@pytest.mark.asyncio
async def test_interrupt_from_tool_running():
    sm = StateMachine()
    await sm.transition(SessionState.LISTENING)
    await sm.transition(SessionState.THINKING)
    await sm.transition(SessionState.TOOL_RUNNING)
    ok = await sm.interrupt(generation=2)
    assert ok is True
    assert sm.state == SessionState.INTERRUPTED


@pytest.mark.asyncio
async def test_interrupted_to_listening():
    sm = StateMachine()
    await sm.transition(SessionState.LISTENING)
    await sm.transition(SessionState.THINKING)
    await sm.interrupt(generation=2)
    ok = await sm.transition(SessionState.LISTENING, generation=2)
    assert ok
    assert sm.state == SessionState.LISTENING


@pytest.mark.asyncio
async def test_invalid_transition_rejected():
    sm = StateMachine()
    # Cannot go from SESSION_CREATED directly to SPEAKING
    ok = await sm.transition(SessionState.SPEAKING)
    assert ok is False
    assert sm.state == SessionState.SESSION_CREATED


@pytest.mark.asyncio
async def test_is_interruptible():
    sm = StateMachine()
    await sm.transition(SessionState.LISTENING)
    assert sm.is_interruptible() is False

    await sm.transition(SessionState.THINKING)
    assert sm.is_interruptible() is True

    await sm.transition(SessionState.TOOL_RUNNING)
    assert sm.is_interruptible() is True

    await sm.transition(SessionState.SPEAKING)
    assert sm.is_interruptible() is True


@pytest.mark.asyncio
async def test_interrupt_latency_under_50ms():
    """Interrupt must complete in ≤ 50 ms (TRD §3.5)."""
    sm = StateMachine()
    await sm.transition(SessionState.LISTENING)
    await sm.transition(SessionState.THINKING)
    await sm.transition(SessionState.SPEAKING)

    t0 = time.monotonic()
    await sm.interrupt(generation=99)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms < 50, f"Interrupt took {elapsed_ms:.1f} ms (target < 50 ms)"


@pytest.mark.asyncio
async def test_transition_callback_fires():
    sm = StateMachine()
    fired: list[tuple] = []

    def cb(frm, to, gen):
        fired.append((frm, to, gen))

    sm.on_transition(cb)
    await sm.transition(SessionState.LISTENING, generation=1)
    assert len(fired) == 1
    assert fired[0] == (SessionState.SESSION_CREATED, SessionState.LISTENING, 1)


@pytest.mark.asyncio
async def test_history_recorded():
    sm = StateMachine()
    await sm.transition(SessionState.LISTENING)
    await sm.transition(SessionState.THINKING)
    history = sm.recent_history(n=5)
    assert len(history) == 2
    assert history[-1]["to"] == SessionState.THINKING.value
