"""
tools/memory_tool.py
--------------------
Voice-driven memory management tool — local, no network.

Handles "remember that…", "what do you know about me", "forget X" commands.
Delegates to memory.MemoryService for all DB operations.

Permission levels:
  • list/get  → 0 (read local)
  • add       → 1 (create local)
  • delete    → 2 (modify/delete local)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .registry import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class MemoryTool(BaseTool):
    name = "memory"
    description = (
        "Store, retrieve, or delete persistent memories for the current user."
    )
    parameters = {
        "action": {
            "type": "string",
            "description": "One of: 'list' | 'add' | 'delete' | 'delete_category'",
        },
        "category": {
            "type": "string",
            "description": "Memory category: 'personal' | 'work' | 'travel' | 'conversation'",
            "default": "personal",
        },
        "content": {
            "type": "string",
            "description": "The text to remember (for 'add')",
        },
        "memory_id": {
            "type": "string",
            "description": "Specific memory ID (for 'delete')",
        },
        "importance": {
            "type": "integer",
            "description": "1–5 importance rating (for 'add'). Default: 3",
            "default": 3,
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
        from memory.memories import MemoryService

        if not profile_id:
            return ToolResult(
                success=False, output=None, tool_name=self.name,
                generation=generation,
                error="No profile_id — cannot manage memories without a known speaker.",
            )

        svc = MemoryService()
        action = args.get("action", "list").lower()

        try:
            if action == "list":
                memories = await svc.list_for_profile(
                    profile_id=profile_id,
                    category=args.get("category"),
                    level=3,  # Only persistent memories surfaced by default
                )
                return ToolResult(
                    success=True,
                    output={"memories": memories, "count": len(memories)},
                    tool_name=self.name,
                    generation=generation,
                )

            elif action == "add":
                content = args.get("content", "").strip()
                if not content:
                    return ToolResult(
                        success=False, output=None, tool_name=self.name,
                        generation=generation, error="Content required for 'add'.",
                    )
                memory_id = await svc.add(
                    profile_id=profile_id,
                    category=args.get("category", "personal"),
                    content=content,
                    memory_level=3,  # Voice-commanded memories are always persistent
                    importance=int(args.get("importance", 3)),
                )
                return ToolResult(
                    success=True,
                    output={"memory_id": memory_id, "content": content},
                    tool_name=self.name,
                    generation=generation,
                )

            elif action == "delete":
                memory_id = args.get("memory_id", "")
                if not memory_id:
                    return ToolResult(
                        success=False, output=None, tool_name=self.name,
                        generation=generation, error="memory_id required for 'delete'.",
                    )
                deleted = await svc.delete(memory_id, profile_id)
                return ToolResult(
                    success=deleted,
                    output={"deleted": deleted, "memory_id": memory_id},
                    tool_name=self.name,
                    generation=generation,
                    error=None if deleted else f"Memory {memory_id} not found.",
                )

            elif action == "delete_category":
                category = args.get("category", "")
                if not category:
                    return ToolResult(
                        success=False, output=None, tool_name=self.name,
                        generation=generation, error="category required.",
                    )
                count = await svc.delete_category(profile_id, category)
                return ToolResult(
                    success=True,
                    output={"deleted_count": count, "category": category},
                    tool_name=self.name,
                    generation=generation,
                )

            else:
                return ToolResult(
                    success=False, output=None, tool_name=self.name,
                    generation=generation, error=f"Unknown action '{action}'.",
                )

        except Exception as exc:  # noqa: BLE001
            logger.error("MemoryTool error: %s", exc)
            return ToolResult(
                success=False, output=None, tool_name=self.name,
                generation=generation, error=str(exc),
            )
