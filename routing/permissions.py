"""
routing/permissions.py
-----------------------
Permission level definitions and enforcement for RIME tools.

Levels (TRD §3.7, Implementation Plan Phase 5):
  0 — read local        : calculator, time lookup, memory read
  1 — create local      : task creation, memory write
  2 — modify/delete     : task delete, memory delete, profile rename
  3 — external access   : weather, transport (read-only external)
  4 — external action   : reserved (Tier 4 — future)

Operations at level ≥ 2 require an explicit user confirmation step
(see safety/confirmations.py) before execution.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional


class PermissionLevel(IntEnum):
    READ_LOCAL      = 0
    CREATE_LOCAL    = 1
    MODIFY_DELETE   = 2
    EXTERNAL_ACCESS = 3
    EXTERNAL_ACTION = 4


# Human-readable descriptions used for the "why did you do that?" feature
PERMISSION_DESCRIPTIONS: dict[int, str] = {
    0: "Read local data — no network, no changes",
    1: "Create local data — no network",
    2: "Modify or delete local data",
    3: "External service access — network request made",
    4: "External service action — real-world effect",
}


def check_permission(
    required_level: int,
    granted_level: int,
    operation: str = "",
) -> tuple[bool, str]:
    """
    Returns (allowed: bool, reason: str).

    Operations requiring level ≥ 2 always require explicit confirmation
    and should not be auto-executed — the caller must call
    safety/confirmations.py first.
    """
    if granted_level >= required_level:
        return True, "Permission granted."
    reason = (
        f"Operation '{operation}' requires permission level "
        f"{required_level} ({PERMISSION_DESCRIPTIONS.get(required_level, '?')}), "
        f"but the current context only grants level {granted_level}."
    )
    return False, reason


def requires_confirmation(permission_level: int) -> bool:
    """True for destructive or external operations (level ≥ 2)."""
    return permission_level >= PermissionLevel.MODIFY_DELETE
