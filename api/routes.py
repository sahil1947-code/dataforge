"""
api/routes.py
-------------
All REST control-plane endpoints for RIME (API Specification §4).

Base URL: /api/v1
Endpoints:
  Profiles      GET/PATCH/DELETE /profiles, POST /profiles/{id}/switch
  Memories      GET/POST/DELETE  /profiles/{id}/memories
  Tasks         GET/POST/PATCH/DELETE /profiles/{id}/tasks, /tasks/{id}
  Conversations GET /profiles/{id}/conversations
  Messages      GET /conversations/{id}/messages
  Tools         GET/PATCH /tools
  Privacy       POST/GET  /privacy/mode, /privacy/temp-session, /privacy/status
  Emergency     GET/POST  /profiles/{id}/emergency-events, /emergency/{id}/confirm
  Metrics       GET       /metrics/latency, /metrics/interruptions
  Sessions      GET       /sessions/{id}/timeline
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Sub-routers ──────────────────────────────────────────────────────
profiles_router   = APIRouter(prefix="/profiles",   tags=["Profiles"])
memories_router   = APIRouter(prefix="/profiles",   tags=["Memories"])
tasks_router      = APIRouter(tags=["Tasks"])
convos_router     = APIRouter(prefix="/conversations", tags=["Conversations"])
tools_router      = APIRouter(prefix="/tools",      tags=["Tools"])
privacy_router    = APIRouter(prefix="/privacy",    tags=["Privacy"])
emergency_router  = APIRouter(tags=["Emergency"])
metrics_router    = APIRouter(prefix="/metrics",    tags=["Metrics"])
sessions_router   = APIRouter(prefix="/sessions",   tags=["Sessions"])


def _err(status: int, code: str, msg: str):
    raise HTTPException(status_code=status, detail={"code": code, "message": msg})


# ════════════════════════════════════════════════════════════════════════
# PROFILES
# ════════════════════════════════════════════════════════════════════════

class ProfileUpdate(BaseModel):
    display_name: Optional[str] = None
    preferences: Optional[dict[str, Any]] = None


@profiles_router.get("")
async def list_profiles(request: Request):
    svc = request.app.state.profile_service
    return await svc.list_all()


@profiles_router.get("/{profile_id}")
async def get_profile(profile_id: str, request: Request):
    svc = request.app.state.profile_service
    p = await svc.get(profile_id)
    if not p:
        _err(404, "PROFILE_NOT_FOUND", f"No profile found: {profile_id}")
    return p


@profiles_router.patch("/{profile_id}")
async def update_profile(profile_id: str, body: ProfileUpdate, request: Request):
    svc = request.app.state.profile_service
    p = await svc.update(profile_id, body.display_name, body.preferences)
    if not p:
        _err(404, "PROFILE_NOT_FOUND", f"No profile found: {profile_id}")
    return p


@profiles_router.delete("/{profile_id}")
async def delete_profile(profile_id: str, request: Request):
    from memory.cleanup import CleanupService
    counts = await CleanupService().forget_profile(profile_id)
    return {"profile_id": profile_id, "deleted": True, "tables_cleared": list(counts.keys())}


@profiles_router.post("/{profile_id}/switch")
async def switch_profile(profile_id: str, request: Request):
    svc = request.app.state.profile_service
    p = await svc.get(profile_id)
    if not p:
        _err(404, "PROFILE_NOT_FOUND", f"No profile found: {profile_id}")
    return {"switched_to": profile_id, "status": "ok"}


# ════════════════════════════════════════════════════════════════════════
# MEMORIES
# ════════════════════════════════════════════════════════════════════════

class MemoryCreate(BaseModel):
    category: str = "personal"
    content: str
    importance: int = Field(default=3, ge=1, le=5)
    memory_level: int = Field(default=3, ge=0, le=3)


@memories_router.get("/{profile_id}/memories")
async def list_memories(
    profile_id: str, request: Request,
    category: Optional[str] = None, level: Optional[int] = None
):
    from memory.memories import MemoryService
    return await MemoryService().list_for_profile(profile_id, category, level)


@memories_router.post("/{profile_id}/memories", status_code=201)
async def create_memory(profile_id: str, body: MemoryCreate, request: Request):
    from memory.memories import MemoryService
    mid = await MemoryService().add(
        profile_id, body.category, body.content,
        body.memory_level, body.importance
    )
    return {"memory_id": mid}


@memories_router.delete("/{profile_id}/memories/{memory_id}")
async def delete_memory(profile_id: str, memory_id: str, request: Request):
    from memory.cleanup import CleanupService
    ok = await CleanupService().forget_memory(memory_id, profile_id)
    if not ok:
        _err(404, "MEMORY_NOT_FOUND", f"Memory {memory_id} not found")
    return {"deleted": True, "memory_id": memory_id}


@memories_router.delete("/{profile_id}/memories")
async def delete_memories_by_category(
    profile_id: str, request: Request, category: Optional[str] = None
):
    from memory.cleanup import CleanupService
    if not category:
        _err(400, "VALIDATION_ERROR", "category query param required")
    count = await CleanupService().forget_memory_category(profile_id, category)
    return {"deleted_count": count, "category": category}


# ════════════════════════════════════════════════════════════════════════
# TASKS
# ════════════════════════════════════════════════════════════════════════

class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    due_at: Optional[str] = None
    recurrence_rule: Optional[str] = None
    depends_on_task_id: Optional[str] = None


class TaskUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    due_at: Optional[str] = None


@tasks_router.get("/profiles/{profile_id}/tasks")
async def list_tasks(
    profile_id: str, request: Request, status: Optional[str] = None
):
    from tasks.manager import TaskManager
    return await TaskManager().list_for_profile(profile_id, status)


@tasks_router.post("/profiles/{profile_id}/tasks", status_code=201)
async def create_task(profile_id: str, body: TaskCreate, request: Request):
    from tasks.manager import TaskManager
    task = await TaskManager().create(
        profile_id, body.title, body.description,
        body.due_at, body.recurrence_rule, body.depends_on_task_id
    )
    return task


@tasks_router.patch("/tasks/{task_id}")
async def update_task(task_id: str, body: TaskUpdate, request: Request):
    from tasks.manager import TaskManager
    task = await TaskManager().update(
        task_id, body.title, body.description, body.status, body.due_at
    )
    if not task:
        _err(404, "TASK_NOT_FOUND", f"Task {task_id} not found")
    return task


@tasks_router.delete("/tasks/{task_id}")
async def delete_task(task_id: str, request: Request):
    from tasks.manager import TaskManager
    ok = await TaskManager().delete(task_id)
    if not ok:
        _err(404, "TASK_NOT_FOUND", f"Task {task_id} not found")
    return {"deleted": True, "task_id": task_id}


# ════════════════════════════════════════════════════════════════════════
# CONVERSATIONS & MESSAGES
# ════════════════════════════════════════════════════════════════════════

@profiles_router.get("/{profile_id}/conversations")
async def list_conversations(profile_id: str, request: Request):
    from database import fetch_all, rows_to_dicts
    rows = await fetch_all(
        "SELECT * FROM conversations WHERE profile_id = ? ORDER BY started_at DESC LIMIT 50",
        (profile_id,),
    )
    return rows_to_dicts(rows)


@convos_router.get("/{conversation_id}/messages")
async def list_messages(conversation_id: str, request: Request):
    from database import fetch_all, rows_to_dicts
    rows = await fetch_all(
        """SELECT * FROM messages WHERE conversation_id = ? AND status = 'spoken'
           ORDER BY sequence_number ASC""",
        (conversation_id,),
    )
    return rows_to_dicts(rows)


@convos_router.get("/{conversation_id}/replay")
async def get_replay(conversation_id: str, request: Request):
    dashboard = request.app.state.dashboard
    return await dashboard.get_session_timeline(conversation_id)


# ════════════════════════════════════════════════════════════════════════
# TOOLS
# ════════════════════════════════════════════════════════════════════════

class ToolToggle(BaseModel):
    enabled: bool


@tools_router.get("")
async def list_tools(request: Request):
    from tools.registry import get_registry
    return get_registry().list_all()


@tools_router.patch("/{tool_name}")
async def toggle_tool(tool_name: str, body: ToolToggle, request: Request):
    from tools.registry import get_registry
    ok = get_registry().set_enabled(tool_name, body.enabled)
    if not ok:
        _err(404, "TOOL_NOT_FOUND", f"Tool '{tool_name}' not registered")
    return {"tool_name": tool_name, "enabled": body.enabled}


# ════════════════════════════════════════════════════════════════════════
# PRIVACY
# ════════════════════════════════════════════════════════════════════════

class PrivacyToggle(BaseModel):
    enabled: bool


@privacy_router.post("/mode")
async def set_privacy_mode(body: PrivacyToggle, request: Request):
    return {"privacy_mode": body.enabled, "status": "updated"}


@privacy_router.post("/temp-session")
async def set_temp_session(body: PrivacyToggle, request: Request):
    return {"temp_session": body.enabled, "status": "updated"}


@privacy_router.get("/status")
async def privacy_status(request: Request):
    return {
        "background_listening": True,
        "voice_identification": True,
        "persistent_memory": True,
        "audio_storage": False,
        "cloud_requests": True,
    }


# ════════════════════════════════════════════════════════════════════════
# EMERGENCY
# ════════════════════════════════════════════════════════════════════════

class EmergencyConfirm(BaseModel):
    action: str   # "activate" | "cancel"


@emergency_router.get("/profiles/{profile_id}/emergency-events")
async def list_emergency_events(profile_id: str, request: Request):
    from database import fetch_all, rows_to_dicts
    rows = await fetch_all(
        "SELECT * FROM emergency_events WHERE profile_id = ? ORDER BY timestamp DESC LIMIT 50",
        (profile_id,),
    )
    return rows_to_dicts(rows)


@emergency_router.post("/emergency/{event_id}/confirm")
async def confirm_emergency(event_id: str, body: EmergencyConfirm, request: Request):
    from safety.emergency import EmergencyDetector
    action_map = {"activate": "confirmed_activated", "cancel": "cancelled"}
    db_action = action_map.get(body.action)
    if not db_action:
        _err(400, "VALIDATION_ERROR", "action must be 'activate' or 'cancel'")
    await EmergencyDetector().update_action(event_id, db_action)
    return {"event_id": event_id, "action_taken": db_action}


# ════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════

@metrics_router.get("/latency")
async def latency_metrics(request: Request):
    return request.app.state.dashboard.get_latency_metrics()


@metrics_router.get("/interruptions")
async def interruption_metrics(request: Request):
    stats = request.app.state.dashboard.get_interruption_metrics()
    return {
        "total_trials":            stats.total_trials,
        "correct":                 stats.correct,
        "incorrect":               stats.incorrect,
        "correctness_rate_pct":    stats.correctness_rate_pct,
        "mean_interrupt_latency_ms": stats.mean_interrupt_latency_ms,
        "stale_artifacts_rejected": stats.stale_artifacts_rejected,
    }


# ════════════════════════════════════════════════════════════════════════
# SESSION TIMELINE
# ════════════════════════════════════════════════════════════════════════

@sessions_router.get("/{conversation_id}/timeline")
async def session_timeline(conversation_id: str, request: Request):
    dashboard = request.app.state.dashboard
    return await dashboard.get_session_timeline(conversation_id)
