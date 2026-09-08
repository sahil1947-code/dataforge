"""
api/main.py
-----------
FastAPI application entry point for RIME.

Start the server:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

Or use the convenience script:
    python main.py
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from config import get_settings
from database import get_connection, close_connection
from observability.logger import setup_logging
from observability.traces import TraceCollector
from observability.dashboard import MetricsDashboard
from speaker.embedding import SpeakerEmbedder
from speaker.profiles import ProfileService
from speech.stt import SpeechToText
from speech.rime import RimeTTS
from agent.agent import ReasoningAgent
from conversation.session import SessionManager
from tasks.manager import TaskManager
from tasks.scheduler import TaskScheduler
from memory.cleanup import CleanupService
from tools.registry import get_registry
from tools.calculator import CalculatorTool
from tools.time_tool import TimeTool
from tools.weather import WeatherTool
from tools.transport import TransportTool
from tools.tasks_tool import TasksTool
from tools.memory_tool import MemoryTool

from .pipeline import VoicePipeline
from .ws_handler import WSSessionHandler
from .routes import (
    profiles_router, memories_router, tasks_router,
    convos_router, tools_router, privacy_router,
    emergency_router, metrics_router, sessions_router,
)

setup_logging()
logger = logging.getLogger(__name__)
_settings = get_settings()


# ── Startup / shutdown ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RIME starting up…")

    # Database
    await get_connection()

    # Register tools
    registry = get_registry()
    registry.register(CalculatorTool())
    registry.register(TimeTool())
    registry.register(WeatherTool())
    registry.register(TransportTool())
    registry.register(TasksTool())
    registry.register(MemoryTool())

    # Shared service singletons
    embedder         = SpeakerEmbedder()
    stt              = SpeechToText()
    rime             = RimeTTS()
    agent            = ReasoningAgent()
    trace_collector  = TraceCollector()
    dashboard        = MetricsDashboard(trace_collector)
    session_manager  = SessionManager()
    profile_service  = ProfileService()
    cleanup_service  = CleanupService()
    task_manager     = TaskManager()
    scheduler        = TaskScheduler(task_manager)

    # Load models in background (non-blocking startup)
    import asyncio
    asyncio.create_task(embedder.load())
    asyncio.create_task(stt.load())

    # Attach to app state (accessible in route handlers via request.app.state)
    app.state.embedder        = embedder
    app.state.stt             = stt
    app.state.rime            = rime
    app.state.agent           = agent
    app.state.trace_collector = trace_collector
    app.state.dashboard       = dashboard
    app.state.session_manager = session_manager
    app.state.profile_service = profile_service
    app.state.pipeline = VoicePipeline(stt, rime, agent, trace_collector)

    # Task scheduler
    scheduler.set_cleanup_service(cleanup_service)
    await scheduler.start()
    app.state.scheduler = scheduler

    logger.info("RIME ready on %s:%d", _settings.app.host, _settings.app.port)
    yield

    # Shutdown
    await scheduler.stop()
    await close_connection()
    logger.info("RIME shut down cleanly.")


# ── App construction ──────────────────────────────────────────────────

app = FastAPI(
    title="RIME — Persistent Voice Operating System",
    version="1.0.0",
    description=(
        "Local-first, identity-aware voice agent with interruption-safe "
        "response management and Rime TTS streaming."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routes
API_PREFIX = "/api/v1"
app.include_router(profiles_router,  prefix=API_PREFIX)
app.include_router(memories_router,  prefix=API_PREFIX)
app.include_router(tasks_router,     prefix=API_PREFIX)
app.include_router(convos_router,    prefix=API_PREFIX)
app.include_router(tools_router,     prefix=API_PREFIX)
app.include_router(privacy_router,   prefix=API_PREFIX)
app.include_router(emergency_router, prefix=API_PREFIX)
app.include_router(metrics_router,   prefix=API_PREFIX)
app.include_router(sessions_router,  prefix=API_PREFIX)

# Static frontend assets
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend", "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ── WebSocket ─────────────────────────────────────────────────────────

@app.websocket("/ws/session")
async def websocket_session(ws: WebSocket):
    handler = WSSessionHandler(
        ws=ws,
        pipeline=ws.app.state.pipeline,
        session_manager=ws.app.state.session_manager,
        embedder=ws.app.state.embedder,
        stt=ws.app.state.stt,
    )
    await handler.run()


# ── HTML companion UI ─────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    _tpl = os.path.join(
        os.path.dirname(__file__), "..", "frontend", "templates", "index.html"
    )
    if os.path.isfile(_tpl):
        return FileResponse(_tpl)
    return {"message": "RIME API is running. Connect at /ws/session"}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
