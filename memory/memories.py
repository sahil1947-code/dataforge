"""
memory/memories.py
------------------
Persistent memory store for RIME — read, write, expire, and retrieve.

Memory levels (TRD §3.11):
  0 — Temporary   : discarded immediately after processing
  1 — Session     : discarded at session end
  2 — Short-term  : rolling window (default 30 days), enforced by expires_at
  3 — Persistent  : retained until explicit "forget" command

Owned table: memories  (Backend Schema §4)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from database import (
    execute,
    fetch_all,
    fetch_one,
    row_to_dict,
    rows_to_dicts,
    transaction,
)

logger = logging.getLogger(__name__)

# Default TTL for short-term memories (level 2)
_LEVEL2_TTL_DAYS: int = 30


class MemoryLevel:
    TEMPORARY   = 0
    SESSION     = 1
    SHORT_TERM  = 2
    PERSISTENT  = 3


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _expires_at(level: int, ttl_days: int = _LEVEL2_TTL_DAYS) -> Optional[str]:
    """Return an ISO-8601 expiry string for levels 1 and 2; None for level 3."""
    if level == MemoryLevel.SESSION:
        # Expire in 24 hours — cleanup.py handles actual deletion at session end
        dt = datetime.now(timezone.utc) + timedelta(hours=24)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if level == MemoryLevel.SHORT_TERM:
        dt = datetime.now(timezone.utc) + timedelta(days=ttl_days)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return None  # Level 3 — no expiry


class MemoryService:
    """
    CRUD for the `memories` table.

    Retrieval is used by the Agent to inject relevant memories into the
    reasoning context.  The "memory approval" pattern (Implementation
    Plan §11) means level-3 memories are only written after explicit
    user confirmation or an explicit "remember this" command.
    """

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def add(
        self,
        profile_id: str,
        category: str,
        content: str,
        memory_level: int = MemoryLevel.PERSISTENT,
        importance: int = 3,
    ) -> str:
        """
        Persist a new memory.  Returns the memory_id.
        Level-3 memories have no expiry; levels 0–2 carry an expires_at.
        """
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        now = _utcnow()
        expires = _expires_at(memory_level)

        await execute(
            """
            INSERT INTO memories
                (memory_id, profile_id, category, content, importance,
                 memory_level, created_at, updated_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                profile_id,
                category,
                content,
                importance,
                memory_level,
                now,
                now,
                expires,
            ),
        )
        logger.info(
            "Memory added: %s (profile=%s, level=%d, category=%s)",
            memory_id,
            profile_id,
            memory_level,
            category,
        )
        return memory_id

    async def update(
        self,
        memory_id: str,
        profile_id: str,
        content: Optional[str] = None,
        importance: Optional[int] = None,
    ) -> bool:
        row = await fetch_one(
            "SELECT * FROM memories WHERE memory_id = ? AND profile_id = ?",
            (memory_id, profile_id),
        )
        if row is None:
            return False
        d = dict(row)
        new_content   = content    if content    is not None else d["content"]
        new_importance = importance if importance is not None else d["importance"]
        await execute(
            """
            UPDATE memories
               SET content = ?, importance = ?, updated_at = ?
             WHERE memory_id = ? AND profile_id = ?
            """,
            (new_content, new_importance, _utcnow(), memory_id, profile_id),
        )
        return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, memory_id: str, profile_id: str) -> Optional[dict[str, Any]]:
        row = await fetch_one(
            "SELECT * FROM memories WHERE memory_id = ? AND profile_id = ?",
            (memory_id, profile_id),
        )
        return row_to_dict(row)

    async def list_for_profile(
        self,
        profile_id: str,
        category: Optional[str] = None,
        level: Optional[int] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Return non-expired memories for a profile, optionally filtered
        by category and memory level — ordered by importance descending.
        """
        query = """
            SELECT * FROM memories
             WHERE profile_id = ?
               AND (expires_at IS NULL OR expires_at > ?)
        """
        params: list[Any] = [profile_id, _utcnow()]

        if category is not None:
            query += " AND category = ?"
            params.append(category)
        if level is not None:
            query += " AND memory_level = ?"
            params.append(level)

        query += " ORDER BY importance DESC, updated_at DESC LIMIT ?"
        params.append(limit)

        rows = await fetch_all(query, tuple(params))
        return rows_to_dicts(rows)

    async def get_context_for_session(
        self, profile_id: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        """
        Return the most relevant persistent memories for injection into
        the reasoning context when a session starts.
        Only level-3 (persistent) memories are injected into context — levels
        0–2 are transient and not fed back to the LLM from the DB.
        """
        return await self.list_for_profile(
            profile_id=profile_id,
            level=MemoryLevel.PERSISTENT,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Delete (granular — "forget me" cascade is in CleanupService)
    # ------------------------------------------------------------------

    async def delete(self, memory_id: str, profile_id: str) -> bool:
        row = await fetch_one(
            "SELECT memory_id FROM memories WHERE memory_id = ? AND profile_id = ?",
            (memory_id, profile_id),
        )
        if row is None:
            return False
        await execute(
            "DELETE FROM memories WHERE memory_id = ? AND profile_id = ?",
            (memory_id, profile_id),
        )
        logger.info("Memory %s deleted (profile=%s)", memory_id, profile_id)
        return True

    async def delete_category(self, profile_id: str, category: str) -> int:
        """Bulk delete all memories in a category for a profile."""
        rows = await fetch_all(
            "SELECT memory_id FROM memories WHERE profile_id = ? AND category = ?",
            (profile_id, category),
        )
        count = len(rows)
        await execute(
            "DELETE FROM memories WHERE profile_id = ? AND category = ?",
            (profile_id, category),
        )
        logger.info(
            "Deleted %d memories in category '%s' for profile %s",
            count,
            category,
            profile_id,
        )
        return count

    async def expire_session_memories(self, profile_id: str) -> int:
        """Delete level-0 and level-1 memories at session end."""
        rows = await fetch_all(
            """
            SELECT memory_id FROM memories
             WHERE profile_id = ? AND memory_level IN (0, 1)
            """,
            (profile_id,),
        )
        count = len(rows)
        if count:
            await execute(
                "DELETE FROM memories WHERE profile_id = ? AND memory_level IN (0, 1)",
                (profile_id,),
            )
            logger.info(
                "Expired %d session/temporary memories for profile %s",
                count,
                profile_id,
            )
        return count

    async def detect_candidate_memory(
        self, text: str
    ) -> Optional[tuple[str, str]]:
        """
        Lightweight heuristic to detect if the user's utterance contains
        an explicit memory instruction ("remember that…", "I prefer…", etc.).
        Returns (category, content) if detected, None otherwise.

        A production implementation would use the LLM for this; here we
        provide a rule-based fallback that covers the most common patterns.
        """
        text_lower = text.lower().strip()
        triggers = [
            ("remember that", "conversation"),
            ("i prefer", "personal"),
            ("my favourite", "personal"),
            ("my favorite", "personal"),
            ("i always", "personal"),
            ("i usually", "personal"),
            ("my name is", "personal"),
            ("call me", "personal"),
            ("i work on", "work"),
            ("my deadline", "work"),
            ("i travel", "travel"),
            ("i live in", "personal"),
        ]
        for trigger, category in triggers:
            if trigger in text_lower:
                # Extract the content after the trigger phrase
                idx = text_lower.index(trigger)
                content = text[idx:].strip()
                return category, content
        return None
