"""
memory/cleanup.py
-----------------
The "Forget Me" protocol and scheduled cleanup for RIME.

Implements:
  1. Full cascading deletion for a profile (PRD US-08, Backend Schema §7).
  2. Granular forgetting (single memory, single conversation, category).
  3. Scheduled rolling-window expiry for level-2 memories and old events.

The "forget me" transaction executes as a single atomic SQLite transaction
so a partial failure cannot leave orphaned data (TRD §6, Backend Schema §7).

Owned tables: ALL profile-scoped tables (owns delete access).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from database import execute, fetch_all, fetch_one, transaction, get_connection

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class CleanupService:
    """
    Handles all destructive data operations for profile management.

    Every public method on this class corresponds directly to a documented
    deletion operation in Backend Schema §7 and §8.
    """

    # ------------------------------------------------------------------
    # Full "Forget Me" cascade (PRD US-08)
    # ------------------------------------------------------------------

    async def forget_profile(self, profile_id: str) -> dict[str, Any]:
        """
        Delete everything linked to `profile_id` in a single atomic transaction.

        Deletion order respects FK dependencies (children before parents):
          tool_calls → messages → conversations → tasks → memories →
          events → emergency_events → voice_embeddings → profiles

        Returns a summary dict with row counts per table for the API response.
        """
        conn = await get_connection()
        counts: dict[str, int] = {}

        # If profile doesn't exist, return zero counts cleanly (no FK violation on events table)
        cursor = await conn.execute(
            "SELECT 1 FROM profiles WHERE profile_id = ?", (profile_id,)
        )
        if not await cursor.fetchone():
            return {
                "tool_calls": 0,
                "messages": 0,
                "conversations": 0,
                "tasks": 0,
                "memories": 0,
                "events": 0,
                "emergency_events": 0,
                "voice_embeddings": 0,
                "profiles": 0,
            }

        try:
            await conn.execute("BEGIN")

            # Log deletion event BEFORE the data is gone (for audit)
            await self._log_deletion_event(conn, profile_id)

            # 1. tool_calls (via conversations)
            r = await conn.execute(
                """
                DELETE FROM tool_calls
                 WHERE conversation_id IN (
                    SELECT conversation_id FROM conversations
                     WHERE profile_id = ?
                 )
                """,
                (profile_id,),
            )
            counts["tool_calls"] = r.rowcount

            # 2. messages (via conversations)
            r = await conn.execute(
                """
                DELETE FROM messages
                 WHERE conversation_id IN (
                    SELECT conversation_id FROM conversations
                     WHERE profile_id = ?
                 )
                """,
                (profile_id,),
            )
            counts["messages"] = r.rowcount

            # 3. conversations
            r = await conn.execute(
                "DELETE FROM conversations WHERE profile_id = ?", (profile_id,)
            )
            counts["conversations"] = r.rowcount

            # 4. tasks
            r = await conn.execute(
                "DELETE FROM tasks WHERE profile_id = ?", (profile_id,)
            )
            counts["tasks"] = r.rowcount

            # 5. memories
            r = await conn.execute(
                "DELETE FROM memories WHERE profile_id = ?", (profile_id,)
            )
            counts["memories"] = r.rowcount

            # 6. events
            r = await conn.execute(
                "DELETE FROM events WHERE profile_id = ?", (profile_id,)
            )
            counts["events"] = r.rowcount

            # 7. emergency_events
            r = await conn.execute(
                "DELETE FROM emergency_events WHERE profile_id = ?", (profile_id,)
            )
            counts["emergency_events"] = r.rowcount

            # 8. voice_embeddings
            r = await conn.execute(
                "DELETE FROM voice_embeddings WHERE profile_id = ?", (profile_id,)
            )
            counts["voice_embeddings"] = r.rowcount

            # 9. profiles (the row itself)
            r = await conn.execute(
                "DELETE FROM profiles WHERE profile_id = ?", (profile_id,)
            )
            counts["profiles"] = r.rowcount

            await conn.execute("COMMIT")

        except Exception:
            await conn.execute("ROLLBACK")
            logger.error("forget_profile transaction rolled back for %s", profile_id)
            raise

        logger.info(
            "Profile %s fully deleted. Row counts: %s", profile_id, counts
        )
        return counts

    # ------------------------------------------------------------------
    # Granular forgetting (Backend Schema §8)
    # ------------------------------------------------------------------

    async def forget_memory(self, memory_id: str, profile_id: str) -> bool:
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
        logger.info("Memory %s deleted for profile %s", memory_id, profile_id)
        return True

    async def forget_memory_category(self, profile_id: str, category: str) -> int:
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
            "Deleted %d memories (category=%s) for profile %s",
            count, category, profile_id,
        )
        return count

    async def forget_conversation(self, conversation_id: str, profile_id: str) -> bool:
        """Delete a single conversation and all its messages and tool calls."""
        row = await fetch_one(
            "SELECT conversation_id FROM conversations WHERE conversation_id = ? AND profile_id = ?",
            (conversation_id, profile_id),
        )
        if row is None:
            return False

        conn = await get_connection()
        await conn.execute("BEGIN")
        try:
            await conn.execute(
                "DELETE FROM tool_calls WHERE conversation_id = ?",
                (conversation_id,),
            )
            await conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            )
            await conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            await conn.execute("COMMIT")
        except Exception:
            await conn.execute("ROLLBACK")
            raise

        logger.info("Conversation %s deleted for profile %s", conversation_id, profile_id)
        return True

    # ------------------------------------------------------------------
    # Scheduled rolling-window expiry
    # ------------------------------------------------------------------

    async def expire_short_term_memories(self) -> int:
        """
        Delete all memory rows whose expires_at has passed.
        Run by the task scheduler on a daily cadence.
        """
        now = _utcnow()
        rows = await fetch_all(
            "SELECT memory_id FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
            (now,),
        )
        count = len(rows)
        if count:
            await execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            logger.info("Expired %d short-term/session memories.", count)
        return count

    async def expire_old_events(self, retention_days: int = 90) -> int:
        """
        Delete general events older than `retention_days` days.
        Emergency events are NOT touched here — they have their own policy.
        """
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)
        cutoff_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = await fetch_all(
            "SELECT event_id FROM events WHERE timestamp < ?", (cutoff_str,)
        )
        count = len(rows)
        if count:
            await execute(
                "DELETE FROM events WHERE timestamp < ?", (cutoff_str,)
            )
            logger.info("Expired %d old audit events (older than %d days).", count, retention_days)
        return count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _log_deletion_event(self, conn, profile_id: str) -> None:
        """
        Write a deletion-initiated event BEFORE the transaction deletes data.
        Only the fact of deletion is logged (no content) as required by the
        "verifiable, complete deletion" requirement in the TRD.
        """
        import uuid, json
        event_id = f"evt_{uuid.uuid4().hex[:12]}"
        await conn.execute(
            """
            INSERT INTO events (event_id, profile_id, event_type, timestamp, metadata)
            VALUES (?, ?, 'profile_deletion_initiated', ?, ?)
            """,
            (
                event_id,
                profile_id,
                _utcnow(),
                json.dumps({"initiated_by": "forget_me_command"}),
            ),
        )
