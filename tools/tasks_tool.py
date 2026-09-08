"""
tools/tasks_tool.py
-------------------
Voice-driven task management tool — local, no network.

Handles "set a reminder", "what are my tasks", "complete task X" voice commands.
Delegates to tasks.TaskManager for all DB operations.

Permission levels:
  • list/get  → 0 (read local)
  • create    → 1 (create local)
  • complete/cancel/delete → 2 (modify/delete local)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .registry import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class TasksTool(BaseTool):
    name = "tasks"
    description = (
        "Create, list, complete, or cancel reminders and tasks for the current user."
    )
    parameters = {
        "action": {
            "type": "string",
            "description": "One of: 'list' | 'create' | 'complete' | 'cancel' | 'get'",
        },
        "title": {
            "type": "string",
            "description": "Task title (required for 'create')",
        },
        "due_at": {
            "type": "string",
            "description": "ISO-8601 datetime for 'create', e.g. '2026-09-07T18:00:00Z'",
        },
        "task_id": {
            "type": "string",
            "description": "Task ID for 'complete', 'cancel', or 'get'",
        },
        "depends_on_task_id": {
            "type": "string",
            "description": "Another task_id this task depends on (for 'create')",
        },
        "status_filter": {
            "type": "string",
            "description": "Filter 'list' by status, e.g. 'scheduled'",
        },
    }
    permission_level = 1
    network_required = False
    api_key_required = False
    cancellable = True
    timeout_seconds = 5.0

    async def execute(
        self,
        args: dict[str, Any],
        generation: int,
        profile_id: Optional[str] = None,
    ) -> ToolResult:
        from tasks.manager import TaskManager

        if not profile_id:
            return ToolResult(
                success=False,
                output=None,
                tool_name=self.name,
                generation=generation,
                error="No profile_id — cannot manage tasks without a known speaker.",
            )

        mgr = TaskManager()
        action = args.get("action", "list").lower()

        try:
            if action == "list":
                tasks = await mgr.list_for_profile(
                    profile_id=profile_id,
                    status=args.get("status_filter"),
                )
                return ToolResult(
                    success=True,
                    output={"tasks": tasks, "count": len(tasks)},
                    tool_name=self.name,
                    generation=generation,
                )

            elif action == "create":
                title = args.get("title", "").strip()
                if not title:
                    return ToolResult(
                        success=False, output=None, tool_name=self.name,
                        generation=generation, error="Task title is required.",
                    )
                task = await mgr.create(
                    profile_id=profile_id,
                    title=title,
                    due_at=args.get("due_at"),
                    depends_on_task_id=args.get("depends_on_task_id"),
                )
                return ToolResult(
                    success=True,
                    output=task,
                    tool_name=self.name,
                    generation=generation,
                )

            elif action == "complete":
                task_id = args.get("task_id", "")
                if not task_id:
                    return ToolResult(
                        success=False, output=None, tool_name=self.name,
                        generation=generation, error="task_id required.",
                    )
                task = await mgr.complete(task_id)
                return ToolResult(
                    success=task is not None,
                    output=task,
                    tool_name=self.name,
                    generation=generation,
                    error=None if task else f"Task {task_id} not found.",
                )

            elif action == "cancel":
                task_id = args.get("task_id", "")
                task = await mgr.cancel(task_id)
                return ToolResult(
                    success=task is not None,
                    output=task,
                    tool_name=self.name,
                    generation=generation,
                )

            elif action == "get":
                task_id = args.get("task_id", "")
                task = await mgr.get(task_id)
                return ToolResult(
                    success=task is not None,
                    output=task,
                    tool_name=self.name,
                    generation=generation,
                    error=None if task else f"Task {task_id} not found.",
                )

            else:
                return ToolResult(
                    success=False, output=None, tool_name=self.name,
                    generation=generation, error=f"Unknown action '{action}'.",
                )

        except Exception as exc:  # noqa: BLE001
            logger.error("TasksTool error: %s", exc)
            return ToolResult(
                success=False, output=None, tool_name=self.name,
                generation=generation, error=str(exc),
            )
