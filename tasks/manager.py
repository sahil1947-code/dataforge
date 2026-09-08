"""
tasks/manager.py
----------------
Task CRUD and state machine for RIME.

Task lifecycle (TRD §3.11, Backend Schema tasks table):
  CREATED → SCHEDULED → WAITING → TRIGGERED → EXECUTED → COMPLETED
                ↑ (depends_on_task_id satisfied)  ↑

Supports:
  • One-time tasks with a due_at timestamp
  • Recurring tasks via recurrence_rule (simplified RFC5545-style)
  • Dependent tasks (task B waits until task A is completed)

Owned table: tasks  (Backend Schema §4)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from database import execute, fetch_all, fetch_one, row_to_dict, rows_to_dicts

logger = logging.getLogger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class TaskStatus(str, Enum):
    CREATED   = "created"
    SCHEDULED = "scheduled"
    WAITING   = "waiting"
    TRIGGERED = "triggered"
    EXECUTED  = "executed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TaskManager:
    """
    Full lifecycle management for tasks: create, update, complete, cancel,
    dependency resolution, and recurrence scheduling.
    """

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create(
        self,
        profile_id: str,
        title: str,
        description: Optional[str] = None,
        due_at: Optional[str] = None,
        recurrence_rule: Optional[str] = None,
        depends_on_task_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create a new task.  If it has a dependency, it starts in WAITING;
        otherwise in SCHEDULED (if due_at is set) or CREATED.
        """
        task_id = f"task_{uuid.uuid4().hex[:10]}"
        now = _utcnow()

        if depends_on_task_id is not None:
            status = TaskStatus.WAITING.value
        elif due_at is not None:
            status = TaskStatus.SCHEDULED.value
        else:
            status = TaskStatus.CREATED.value

        await execute(
            """
            INSERT INTO tasks
                (task_id, profile_id, title, description, status,
                 created_at, due_at, recurrence_rule, depends_on_task_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                profile_id,
                title,
                description,
                status,
                now,
                due_at,
                recurrence_rule,
                depends_on_task_id,
            ),
        )
        logger.info(
            "Task created: %s '%s' (profile=%s, status=%s)",
            task_id, title, profile_id, status,
        )
        return await self._get_or_raise(task_id)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get(self, task_id: str) -> Optional[dict[str, Any]]:
        row = await fetch_one(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        )
        return row_to_dict(row)

    async def list_for_profile(
        self,
        profile_id: str,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if status:
            rows = await fetch_all(
                """
                SELECT * FROM tasks
                 WHERE profile_id = ? AND status = ?
                 ORDER BY due_at ASC NULLS LAST, created_at ASC
                 LIMIT ?
                """,
                (profile_id, status, limit),
            )
        else:
            rows = await fetch_all(
                """
                SELECT * FROM tasks
                 WHERE profile_id = ?
                 ORDER BY due_at ASC NULLS LAST, created_at ASC
                 LIMIT ?
                """,
                (profile_id, limit),
            )
        return rows_to_dicts(rows)

    async def get_due_tasks(self) -> list[dict[str, Any]]:
        """Return all scheduled/waiting tasks whose due_at has passed."""
        rows = await fetch_all(
            """
            SELECT * FROM tasks
             WHERE status IN ('scheduled', 'waiting')
               AND due_at IS NOT NULL
               AND due_at <= ?
             ORDER BY due_at ASC
            """,
            (_utcnow(),),
        )
        return rows_to_dicts(rows)

    # ------------------------------------------------------------------
    # Update / lifecycle
    # ------------------------------------------------------------------

    async def update(
        self,
        task_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        due_at: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        task = await self.get(task_id)
        if task is None:
            return None

        new_title       = title       if title       is not None else task["title"]
        new_description = description if description is not None else task["description"]
        new_status      = status      if status      is not None else task["status"]
        new_due         = due_at      if due_at      is not None else task["due_at"]

        await execute(
            """
            UPDATE tasks
               SET title = ?, description = ?, status = ?, due_at = ?
             WHERE task_id = ?
            """,
            (new_title, new_description, new_status, new_due, task_id),
        )
        return await self._get_or_raise(task_id)

    async def complete(self, task_id: str) -> Optional[dict[str, Any]]:
        """
        Mark a task as completed and resolve any tasks that were waiting on it.
        """
        task = await self.get(task_id)
        if task is None:
            return None

        await execute(
            "UPDATE tasks SET status = 'completed' WHERE task_id = ?",
            (task_id,),
        )
        logger.info("Task completed: %s", task_id)

        # Resolve dependents — transition WAITING → TRIGGERED
        await self._resolve_dependents(task_id)

        # Handle recurrence
        if task.get("recurrence_rule"):
            await self._spawn_recurrence(task)

        return await self._get_or_raise(task_id)

    async def cancel(self, task_id: str) -> Optional[dict[str, Any]]:
        task = await self.get(task_id)
        if task is None:
            return None
        await execute(
            "UPDATE tasks SET status = 'cancelled' WHERE task_id = ?",
            (task_id,),
        )
        logger.info("Task cancelled: %s", task_id)
        return await self._get_or_raise(task_id)

    async def delete(self, task_id: str) -> bool:
        row = await fetch_one(
            "SELECT task_id FROM tasks WHERE task_id = ?", (task_id,)
        )
        if row is None:
            return False
        await execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))
        return True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_or_raise(self, task_id: str) -> dict[str, Any]:
        row = await fetch_one(
            "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
        )
        if row is None:
            raise RuntimeError(f"Task {task_id} not found after write — DB inconsistency.")
        return dict(row)

    async def _resolve_dependents(self, completed_task_id: str) -> None:
        """
        Find tasks waiting on `completed_task_id` and transition them
        to TRIGGERED so the scheduler can pick them up.
        """
        rows = await fetch_all(
            """
            SELECT task_id FROM tasks
             WHERE depends_on_task_id = ? AND status = 'waiting'
            """,
            (completed_task_id,),
        )
        for row in rows:
            await execute(
                "UPDATE tasks SET status = 'triggered' WHERE task_id = ?",
                (row["task_id"],),
            )
            logger.info(
                "Task %s unblocked (dependency %s completed).",
                row["task_id"],
                completed_task_id,
            )

    async def _spawn_recurrence(self, task: dict[str, Any]) -> None:
        """
        Create the next occurrence of a recurring task.
        Supports a minimal rule subset:
          DAILY, WEEKLY, MONTHLY — e.g. "FREQ=DAILY;INTERVAL=1"
        """
        rule = task.get("recurrence_rule", "")
        if not rule:
            return

        parts = dict(p.split("=") for p in rule.upper().split(";") if "=" in p)
        freq = parts.get("FREQ")
        interval = int(parts.get("INTERVAL", "1"))

        if not freq or not task.get("due_at"):
            return

        from datetime import timedelta
        try:
            due = datetime.fromisoformat(task["due_at"].replace("Z", "+00:00"))
        except ValueError:
            return

        if freq == "DAILY":
            next_due = due + timedelta(days=interval)
        elif freq == "WEEKLY":
            next_due = due + timedelta(weeks=interval)
        elif freq == "MONTHLY":
            # Approximate: 30 days per month
            next_due = due + timedelta(days=30 * interval)
        else:
            return

        await self.create(
            profile_id=task["profile_id"],
            title=task["title"],
            description=task.get("description"),
            due_at=next_due.strftime("%Y-%m-%dT%H:%M:%SZ"),
            recurrence_rule=rule,
        )
        logger.info(
            "Recurring task '%s' rescheduled for %s.", task["title"], next_due
        )
