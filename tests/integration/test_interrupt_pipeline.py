"""
tests/integration/test_interrupt_pipeline.py
---------------------------------------------
Integration test: full pipeline interrupt during a slow tool call.

This is the wire-level expression of the generation-ID fencing claim.
It exercises the same code path as the stress test but with assertions
on the exact sequence of events rather than bulk statistics.
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch

from conversation.state_machine import StateMachine, SessionState
from conversation.generations import GenerationCounter
from conversation.interruption import ResponseManager, Artifact, ArtifactType


# ── Helpers ───────────────────────────────────────────────────────────

class _StubRime:
    def __init__(self):
        self.cancelled = False
        self.spoken_texts: list[str] = []

    async def cancel(self):
        self.cancelled = True

    async def speak(self, text: str, generation: int):
        self.cancelled = False
        self.spoken_texts.append((text, generation))


def _make_session() -> tuple[StateMachine, GenerationCounter, ResponseManager]:
    sm  = StateMachine()
    gen = GenerationCounter(initial=0)
    rm  = ResponseManager(sm, gen, conversation_id="int_test_conv")
    return sm, gen, rm


# ── Core interrupt scenario ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_stale_tool_result_never_reaches_user():
    """
    Scenario:
      gen 1 → slow tool call starts
      INTERRUPT → gen 2
      slow tool result arrives (gen 1) → must be discarded
      gen 2 → fresh calculator result → must be allowed
    """
    sm, gen_counter, rm = _make_session()
    rime = _StubRime()

    # Patch DB calls so no real SQLite is needed
    with patch("conversation.interruption.execute", new=AsyncMock(return_value=0)), \
         patch("conversation.interruption.fetch_all", new=AsyncMock(return_value=[])):

        await sm.transition(SessionState.LISTENING)
        await sm.transition(SessionState.THINKING)

        gen1 = await rm.on_new_turn()  # 1
        assert gen1 == 1

        await sm.transition(SessionState.TOOL_RUNNING, generation=gen1)

        # --- Interrupt ---
        gen2 = await rm.on_interrupt(current_rime_session=rime)
        assert gen2 == 2
        assert rime.cancelled is True

        # --- Stale tool result (gen 1) arrives ---
        stale = Artifact(
            payload={"trains": []},
            generation=gen1,
            artifact_type=ArtifactType.TOOL_RESULT,
            tool_call_id="tc_slow",
        )
        allowed_stale = await rm.allow(stale)
        assert allowed_stale is False, "Stale tool result must not be allowed through"

        # --- Fresh result (gen 2) arrives ---
        fresh = Artifact(
            payload="The answer is 10.",
            generation=gen2,
            artifact_type=ArtifactType.LLM_TOKEN,
        )
        allowed_fresh = await rm.allow(fresh)
        assert allowed_fresh is True, "Fresh result must be allowed through"

        # --- Speak only the fresh result ---
        if allowed_fresh:
            await rime.speak(fresh.payload, gen2)

    assert "The answer is 10." in [t for t, _ in rime.spoken_texts]
    # Stale payload must never appear in spoken texts
    spoken_payloads = [t for t, _ in rime.spoken_texts]
    assert all("trains" not in p for p in spoken_payloads)


@pytest.mark.asyncio
async def test_multiple_stale_artifacts_all_discarded():
    sm, _, rm = _make_session()

    with patch("conversation.interruption.execute", new=AsyncMock(return_value=0)), \
         patch("conversation.interruption.fetch_all", new=AsyncMock(return_value=[])):

        gen1 = await rm.on_new_turn()
        await rm.on_interrupt()   # gen → 2

        # 10 stale artifacts from gen 1
        for i in range(10):
            art = Artifact(
                payload=f"stale chunk {i}",
                generation=gen1,
                artifact_type=ArtifactType.AUDIO_CHUNK,
            )
            assert await rm.allow(art) is False

    assert rm.discarded_count == 10


@pytest.mark.asyncio
async def test_correct_generation_after_two_interrupts():
    sm, _, rm = _make_session()

    with patch("conversation.interruption.execute", new=AsyncMock(return_value=0)), \
         patch("conversation.interruption.fetch_all", new=AsyncMock(return_value=[])):

        await rm.on_new_turn()      # gen 1
        await rm.on_interrupt()     # gen 2
        await rm.on_new_turn()      # gen 3
        g4 = await rm.on_interrupt()  # gen 4

        assert g4 == 4

        art = Artifact("final", generation=4, artifact_type=ArtifactType.LLM_TOKEN)
        assert await rm.allow(art) is True

        for old_gen in (1, 2, 3):
            stale = Artifact("old", generation=old_gen,
                             artifact_type=ArtifactType.LLM_TOKEN)
            assert await rm.allow(stale) is False


@pytest.mark.asyncio
async def test_state_machine_reaches_listening_after_interrupt():
    sm, _, rm = _make_session()

    with patch("conversation.interruption.execute", new=AsyncMock(return_value=0)), \
         patch("conversation.interruption.fetch_all", new=AsyncMock(return_value=[])):

        await sm.transition(SessionState.LISTENING)
        await sm.transition(SessionState.THINKING)
        await sm.transition(SessionState.SPEAKING)

        await rm.on_interrupt()  # → INTERRUPTED

        assert sm.state == SessionState.INTERRUPTED

        await sm.transition(SessionState.LISTENING, generation=2)
        assert sm.state == SessionState.LISTENING
