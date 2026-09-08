"""
tests/unit/test_response_manager.py
-------------------------------------
Unit tests for the Response Manager — the correctness-critical component.

These tests are the highest-priority tests in the entire codebase.
Every generation-ID fencing scenario is covered explicitly.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from conversation.state_machine import StateMachine, SessionState
from conversation.generations import GenerationCounter
from conversation.interruption import ResponseManager, Artifact, ArtifactType


def _make_rm(conversation_id: str = "test_conv_001") -> ResponseManager:
    sm  = StateMachine()
    gen = GenerationCounter(initial=0)
    rm  = ResponseManager(
        state_machine=sm,
        generation_counter=gen,
        conversation_id=conversation_id,
    )
    return rm


# ── Fencing: allow() ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_allow_current_generation():
    rm = _make_rm()
    await rm.on_new_turn()  # gen → 1
    art = Artifact(payload="Hello", generation=1,
                   artifact_type=ArtifactType.LLM_TOKEN)
    assert await rm.allow(art) is True


@pytest.mark.asyncio
async def test_deny_stale_generation():
    rm = _make_rm()
    await rm.on_new_turn()  # gen → 1
    await rm.on_new_turn()  # gen → 2
    stale = Artifact(payload="old result", generation=1,
                     artifact_type=ArtifactType.TOOL_RESULT,
                     tool_call_id="tc_old")
    assert await rm.allow(stale) is False


@pytest.mark.asyncio
async def test_allow_increments_discarded_count_on_stale():
    rm = _make_rm()
    await rm.on_new_turn()  # gen → 1
    await rm.on_new_turn()  # gen → 2
    stale = Artifact(payload="old", generation=1,
                     artifact_type=ArtifactType.LLM_TOKEN)
    await rm.allow(stale)
    assert rm.discarded_count == 1


@pytest.mark.asyncio
async def test_stale_never_reaches_user_across_multiple_artifacts():
    """Simulate 5 stale artifacts arriving after interrupt. All must be denied."""
    rm = _make_rm()
    await rm.on_new_turn()  # gen 1
    await rm.on_new_turn()  # gen 2 (interrupt)

    for _ in range(5):
        art = Artifact(payload="stale chunk", generation=1,
                       artifact_type=ArtifactType.AUDIO_CHUNK)
        result = await rm.allow(art)
        assert result is False

    assert rm.discarded_count == 5


@pytest.mark.asyncio
async def test_fresh_artifact_allowed_after_interrupt():
    rm = _make_rm()
    await rm.on_new_turn()   # gen 1
    await rm.on_interrupt()  # gen 2

    fresh = Artifact(payload="new answer", generation=2,
                     artifact_type=ArtifactType.LLM_TOKEN)
    assert await rm.allow(fresh) is True


# ── Interrupt: on_interrupt() ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_interrupt_increments_generation():
    rm = _make_rm()
    await rm.on_new_turn()  # gen 1
    old = rm.current_generation
    new_gen = await rm.on_interrupt()
    assert new_gen == old + 1
    assert rm.current_generation == new_gen


@pytest.mark.asyncio
async def test_interrupt_cancels_rime_session():
    rm = _make_rm()
    await rm.on_new_turn()

    mock_session = AsyncMock()
    mock_session.cancel = AsyncMock()

    await rm.on_interrupt(current_rime_session=mock_session)
    mock_session.cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_multiple_interrupts_keep_incrementing():
    rm = _make_rm()
    await rm.on_new_turn()     # gen 1
    g2 = await rm.on_interrupt()   # gen 2
    g3 = await rm.on_interrupt()   # gen 3
    assert g2 == 2
    assert g3 == 3


# ── on_new_turn() ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_on_new_turn_increments():
    rm = _make_rm()
    assert rm.current_generation == 0
    g = await rm.on_new_turn()
    assert g == 1
    assert rm.current_generation == 1


# ── persist_spoken_message() ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_persist_spoken_message_current_gen(tmp_path, monkeypatch):
    """Message must be persisted when generation is current."""
    rm = _make_rm()
    await rm.on_new_turn()  # gen 1

    persisted_calls: list = []

    async def mock_execute(query, params=()):
        if "INSERT INTO messages" in query:
            persisted_calls.append(params)
        return 0

    monkeypatch.setattr("conversation.interruption.execute", mock_execute)

    msg_id = await rm.persist_spoken_message(
        role="assistant", text="Hello!", generation=1, sequence_number=1
    )
    assert msg_id is not None
    assert len(persisted_calls) == 1


@pytest.mark.asyncio
async def test_persist_spoken_message_stale_gen(monkeypatch):
    """Stale message must NOT be persisted."""
    rm = _make_rm()
    await rm.on_new_turn()  # gen 1
    await rm.on_new_turn()  # gen 2

    persisted_calls: list = []

    async def mock_execute(query, params=()):
        if "INSERT INTO messages" in query:
            persisted_calls.append(params)
        return 0

    monkeypatch.setattr("conversation.interruption.execute", mock_execute)

    msg_id = await rm.persist_spoken_message(
        role="assistant", text="Stale!", generation=1, sequence_number=1
    )
    assert msg_id is None
    assert len(persisted_calls) == 0


# ── Tool call lifecycle ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_complete_tool_call_current_returns_true(monkeypatch):
    rm = _make_rm()
    await rm.on_new_turn()  # gen 1

    async def mock_execute(q, p=()):
        return 0
    monkeypatch.setattr("conversation.interruption.execute", mock_execute)
    monkeypatch.setattr("conversation.interruption.fetch_all",
                        AsyncMock(return_value=[]))

    await rm.register_tool_call("tc_1", "calculator", "{}", 1)
    is_current = await rm.complete_tool_call("tc_1", generation=1, success=True)
    assert is_current is True


@pytest.mark.asyncio
async def test_complete_tool_call_stale_returns_false(monkeypatch):
    rm = _make_rm()
    await rm.on_new_turn()  # gen 1
    await rm.on_new_turn()  # gen 2

    async def mock_execute(q, p=()):
        return 0
    monkeypatch.setattr("conversation.interruption.execute", mock_execute)
    monkeypatch.setattr("conversation.interruption.fetch_all",
                        AsyncMock(return_value=[]))

    await rm.register_tool_call("tc_1", "transport", "{}", 1)
    is_current = await rm.complete_tool_call("tc_1", generation=1, success=True)
    assert is_current is False
