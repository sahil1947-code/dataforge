"""
conversation/generations.py
----------------------------
Per-session generation counter and artifact tagging.

This module is the direct implementation of TRD §3.9 "Generation IDs":

    Every user request increments a per-session `generation` counter.
    Every downstream artifact (LLM output, tool result, Rime audio chunk)
    is tagged with the generation it was produced for.

    if artifact.generation != session.current_generation:
        discard(artifact)

The counter is intentionally simple and O(1) — correctness over cleverness.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class GenerationCounter:
    """
    Per-session monotonically increasing generation counter.

    Thread-safe via asyncio.Lock.  Increment is O(1).
    """

    def __init__(self, initial: int = 0) -> None:
        self._value = initial
        self._lock = asyncio.Lock()

    @property
    def current(self) -> int:
        return self._value

    async def increment(self) -> int:
        """
        Increment the counter and return the new generation.
        Called by the Response Manager every time a user interrupts
        or starts a new request.
        """
        async with self._lock:
            self._value += 1
            logger.debug("Generation incremented to %d", self._value)
            return self._value

    def is_current(self, generation: int) -> bool:
        """O(1) check — used by the Response Manager fencing logic."""
        return generation == self._value


@dataclass
class TaggedArtifact(Generic[T]):
    """
    Wraps any downstream artifact (LLM token, tool result, audio chunk)
    with the generation it was produced for.
    """

    payload: T
    generation: int
    artifact_type: str    # "llm_token" | "tool_result" | "audio_chunk" | "message"


def tag_artifact(payload: T, generation: int, artifact_type: str) -> TaggedArtifact[T]:
    """Convenience factory — tags a payload with the current generation."""
    return TaggedArtifact(
        payload=payload,
        generation=generation,
        artifact_type=artifact_type,
    )


def check_artifact(artifact: TaggedArtifact, counter: GenerationCounter) -> bool:
    """
    Returns True if the artifact belongs to the current generation and
    should be allowed through the Response Manager fence.

    Returns False (stale) if it was produced for a superseded generation.
    """
    is_fresh = counter.is_current(artifact.generation)
    if not is_fresh:
        logger.debug(
            "Stale artifact discarded: type=%s artifact_gen=%d current_gen=%d",
            artifact.artifact_type,
            artifact.generation,
            counter.current,
        )
    return is_fresh
