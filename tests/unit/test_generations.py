"""
tests/unit/test_generations.py
-------------------------------
Unit tests for the generation counter and artifact tagging logic.
These are the highest-coverage tests in the codebase — correctness of
generation-ID fencing depends on these primitives.
"""

import asyncio
import pytest
from conversation.generations import (
    GenerationCounter,
    TaggedArtifact,
    tag_artifact,
    check_artifact,
)


@pytest.mark.asyncio
async def test_counter_starts_at_initial():
    c = GenerationCounter(initial=0)
    assert c.current == 0


@pytest.mark.asyncio
async def test_counter_increments():
    c = GenerationCounter(initial=0)
    new = await c.increment()
    assert new == 1
    assert c.current == 1


@pytest.mark.asyncio
async def test_counter_increments_multiple_times():
    c = GenerationCounter(initial=0)
    for i in range(1, 6):
        val = await c.increment()
        assert val == i
    assert c.current == 5


@pytest.mark.asyncio
async def test_is_current_true():
    c = GenerationCounter(initial=3)
    assert c.is_current(3) is True


@pytest.mark.asyncio
async def test_is_current_false_past():
    c = GenerationCounter(initial=3)
    assert c.is_current(2) is False


@pytest.mark.asyncio
async def test_is_current_false_future():
    c = GenerationCounter(initial=3)
    assert c.is_current(4) is False


@pytest.mark.asyncio
async def test_tag_artifact():
    art = tag_artifact(payload="hello", generation=5, artifact_type="llm_token")
    assert art.payload == "hello"
    assert art.generation == 5
    assert art.artifact_type == "llm_token"


@pytest.mark.asyncio
async def test_check_artifact_current():
    c = GenerationCounter(initial=5)
    art = tag_artifact("hello", generation=5, artifact_type="llm_token")
    assert check_artifact(art, c) is True


@pytest.mark.asyncio
async def test_check_artifact_stale():
    c = GenerationCounter(initial=5)
    art = tag_artifact("stale", generation=4, artifact_type="tool_result")
    assert check_artifact(art, c) is False


@pytest.mark.asyncio
async def test_concurrent_increments_are_safe():
    """Counter must be thread-safe under concurrent asyncio tasks."""
    c = GenerationCounter(initial=0)

    async def bump():
        return await c.increment()

    results = await asyncio.gather(*[bump() for _ in range(20)])
    # All 20 increments must produce unique values 1..20
    assert sorted(results) == list(range(1, 21))
    assert c.current == 20
