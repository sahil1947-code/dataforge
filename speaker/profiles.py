"""
speaker/profiles.py
-------------------
Profile management: rename, list, switch, delete, update preferences.

Handles the REST-facing operations for /api/v1/profiles/*.
Owned table: profiles  (reads voice_embeddings for counts).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from database import (
    fetch_one,
    fetch_all,
    execute,
    transaction,
    row_to_dict,
    rows_to_dicts,
    encode_json,
    decode_json,
)

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ProfileService:
    """
    CRUD and profile-lifecycle operations.

    All destructive operations (delete) are handled by memory/cleanup.py
    which owns the cascading-deletion transaction.  This module only
    handles non-destructive mutations and reads.
    """

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get(self, profile_id: str) -> Optional[dict[str, Any]]:
        row = await fetch_one(
            "SELECT * FROM profiles WHERE profile_id = ?", (profile_id,)
        )
        if row is None:
            return None
        d = row_to_dict(row)
        d["preferences"] = decode_json(d.pop("preferences_json", "{}"))
        return d

    async def list_all(self) -> list[dict[str, Any]]:
        rows = await fetch_all(
            "SELECT * FROM profiles WHERE status != 'disabled' ORDER BY last_seen DESC"
        )
        result = []
        for row in rows:
            d = dict(row)
            d["preferences"] = decode_json(d.pop("preferences_json", "{}"))
            result.append(d)
        return result

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def update(
        self,
        profile_id: str,
        display_name: Optional[str] = None,
        preferences: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        existing = await self.get(profile_id)
        if existing is None:
            return None

        new_name = display_name if display_name is not None else existing.get("display_name")
        merged_prefs = {**existing.get("preferences", {}), **(preferences or {})}
        now = _utcnow()

        await execute(
            """
            UPDATE profiles
               SET display_name = ?,
                   preferences_json = ?,
                   last_seen = ?
             WHERE profile_id = ?
            """,
            (new_name, encode_json(merged_prefs), now, profile_id),
        )
        logger.info("Profile %s updated (name=%s)", profile_id, new_name)
        return await self.get(profile_id)

    async def set_status(self, profile_id: str, status: str) -> None:
        """Set profile status: 'active' | 'privacy_mode' | 'disabled'."""
        valid = {"active", "privacy_mode", "disabled"}
        if status not in valid:
            raise ValueError(f"Invalid status '{status}'. Must be one of {valid}.")
        await execute(
            "UPDATE profiles SET status = ?, last_seen = ? WHERE profile_id = ?",
            (status, _utcnow(), profile_id),
        )
        logger.info("Profile %s status → %s", profile_id, status)

    async def touch(self, profile_id: str) -> None:
        """Update last_seen to now (called on every session start for a known profile)."""
        await execute(
            "UPDATE profiles SET last_seen = ? WHERE profile_id = ?",
            (_utcnow(), profile_id),
        )

    async def get_preferences(self, profile_id: str) -> dict[str, Any]:
        row = await fetch_one(
            "SELECT preferences_json FROM profiles WHERE profile_id = ?",
            (profile_id,),
        )
        if row is None:
            return {}
        return decode_json(row["preferences_json"])

    async def set_preference(
        self, profile_id: str, key: str, value: Any
    ) -> None:
        prefs = await self.get_preferences(profile_id)
        prefs[key] = value
        await execute(
            "UPDATE profiles SET preferences_json = ? WHERE profile_id = ?",
            (encode_json(prefs), profile_id),
        )
