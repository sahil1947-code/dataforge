"""
tasks/scheduler.py
------------------
Background task scheduler for RIME.

Polls the `tasks` table on a fixed interval and fires triggered/due tasks
by calling the registered handler.  Also drives periodic cleanup jobs
(memory expiry, event rolling-window purge).

Design principles:
  • Runs as a single asyncio background task — no threads, no external queue
  • Uses asyncio.sleep for the poll interval — low overhead, easily cancellable
  • Fires a registered async callback per due task so the caller controls the
    notification mechanism (Rime speak, REST event, etc.)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Callable, Coroutine, Optional

from .manager import TaskManager

logger = logging.getLogger(__name__)

# How often the scheduler checks for due tasks (seconds)
_POLL_INTERVAL_SECONDS: int = 10

# How often the cleanup jobs run (seconds — ~daily in production)
_CLEANUP_INTERVAL_SECONDS: int = 86_400


TaskFireCallback = Callable[[dict], Coroutine]


class TaskScheduler:
    """
    Lightweight asyncio-based task scheduler.

    Usage:
        scheduler = TaskScheduler(task_manager)
        scheduler.on_task_due(my_async_handler)
        await scheduler.start()
        # ... app runs ...
        await scheduler.stop()
    """

    def __init__(
        self,
        task_manager: TaskManager,
        poll_interval: int = _POLL_INTERVAL_SECONDS,
    ) -> None:
        self._manager = task_manager
        self._poll_interval = poll_interval
        self._running = False
        self._poll_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._on_due_callbacks: list[TaskFireCallback] = []
        self._cleanup_service = None   # injected lazily to avoid circular import

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="rime-task-scheduler"
        )
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(), name="rime-cleanup-scheduler"
        )
        logger.info(
            "TaskScheduler started (poll interval=%ds).", self._poll_interval
        )

    async def stop(self) -> None:
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("TaskScheduler stopped.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_task_due(self, callback: TaskFireCallback) -> None:
        """Register a coroutine callback called with the task dict when it fires."""
        self._on_due_callbacks.append(callback)

    def set_cleanup_service(self, cleanup_service) -> None:
        self._cleanup_service = cleanup_service

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    async def _poll_loop(self) -> None:
        while self._running:
            try:
                await self._fire_due_tasks()
            except Exception as exc:  # noqa: BLE001
                logger.error("TaskScheduler poll error: %s", exc)
            await asyncio.sleep(self._poll_interval)

    async def _cleanup_loop(self) -> None:
        while self._running:
            await asyncio.sleep(_CLEANUP_INTERVAL_SECONDS)
            try:
                await self._run_cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.error("TaskScheduler cleanup error: %s", exc)

    async def _fire_due_tasks(self) -> None:
        """
        Fetch all due tasks, mark them TRIGGERED → EXECUTED, and call handlers.
        """
        due = await self._manager.get_due_tasks()
        for task in due:
            task_id = task["task_id"]
            # Mark as triggered before firing to prevent double-firing on
            # the next poll if the handler is slow
            await self._manager.update(task_id, status="triggered")

            logger.info(
                "Task due: %s '%s' (profile=%s)",
                task_id, task["title"], task["profile_id"],
            )
            for cb in self._on_due_callbacks:
                asyncio.create_task(self._safe_fire(cb, task))

    async def _safe_fire(self, cb: TaskFireCallback, task: dict) -> None:
        """Invoke a callback, catching and logging any error so one bad
        callback cannot crash the scheduler loop."""
        try:
            await cb(task)
            # Mark as executed after successful callback invocation
            await self._manager.update(task["task_id"], status="executed")
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Task callback error for %s: %s", task["task_id"], exc
            )

    async def _run_cleanup(self) -> None:
        if self._cleanup_service is None:
            return
        expired = await self._cleanup_service.expire_short_term_memories()
        events_purged = await self._cleanup_service.expire_old_events()
        logger.info(
            "Cleanup: expired %d memories, purged %d old events.",
            expired,
            events_purged,
        )
