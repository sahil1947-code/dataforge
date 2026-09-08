"""
tests/memory/test_cleanup.py
-----------------------------
Tests for the "forget me" cascading deletion and granular forgetting.
Uses a real in-memory SQLite database (not mocks) to verify correctness.
"""

import asyncio
import pytest
from datetime import datetime, timezone

from database.db import get_connection, close_connection


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _seed_profile(conn, profile_id: str) -> None:
    """Insert a minimal profile + related rows for deletion testing."""
    now = _utcnow()
    await conn.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, created_at, last_seen) VALUES (?, ?, ?)",
        (profile_id, now, now),
    )
    # Conversation + message
    conv_id = f"conv_{profile_id}"
    await conn.execute(
        "INSERT OR IGNORE INTO conversations (conversation_id, profile_id, started_at) VALUES (?, ?, ?)",
        (conv_id, profile_id, now),
    )
    await conn.execute(
        "INSERT OR IGNORE INTO messages (message_id, conversation_id, role, text, timestamp, sequence_number, generation) VALUES (?, ?, 'user', 'hello', ?, 1, 1)",
        (f"msg_{profile_id}", conv_id, now),
    )
    # Tool call
    await conn.execute(
        "INSERT OR IGNORE INTO tool_calls (tool_call_id, conversation_id, tool_name, arguments, generation, started_at) VALUES (?, ?, 'calc', '{}', 1, ?)",
        (f"tc_{profile_id}", conv_id, now),
    )
    # Memory
    await conn.execute(
        "INSERT OR IGNORE INTO memories (memory_id, profile_id, category, content, importance, memory_level, created_at, updated_at) VALUES (?, ?, 'personal', 'test memory', 3, 3, ?, ?)",
        (f"mem_{profile_id}", profile_id, now, now),
    )
    # Task
    await conn.execute(
        "INSERT OR IGNORE INTO tasks (task_id, profile_id, title, created_at) VALUES (?, ?, 'test task', ?)",
        (f"task_{profile_id}", profile_id, now),
    )
    # Voice embedding (stub blob)
    await conn.execute(
        "INSERT OR IGNORE INTO voice_embeddings (embedding_id, profile_id, embedding_vector, created_at) VALUES (?, ?, ?, ?)",
        (f"emb_{profile_id}", profile_id, b"\x00" * 16, now),
    )
    await conn.commit()


@pytest.fixture(autouse=True)
async def fresh_db(tmp_path, monkeypatch):
    """Each test gets a fresh SQLite DB at a temp path."""
    db_path = str(tmp_path / "test_rime.sqlite")
    monkeypatch.setenv("DATABASE_PATH", db_path)
    # Reset the cached connection so it picks up the new path
    import database.db as dbmod
    dbmod._connection = None
    yield
    await close_connection()
    dbmod._connection = None


@pytest.mark.asyncio
async def test_forget_profile_removes_all_rows():
    from memory.cleanup import CleanupService

    conn = await get_connection()
    pid = "speaker_test01"
    await _seed_profile(conn, pid)

    # Verify seeded
    row = await conn.execute("SELECT profile_id FROM profiles WHERE profile_id = ?", (pid,))
    assert await row.fetchone() is not None

    svc = CleanupService()
    counts = await svc.forget_profile(pid)

    # All tables must be empty for this profile
    for table in ("profiles", "voice_embeddings", "conversations",
                  "messages", "tasks", "memories", "tool_calls"):
        col = "profile_id" if table not in ("messages", "tool_calls") else "conversation_id"
        if table in ("messages", "tool_calls"):
            check = await conn.execute(
                f"SELECT 1 FROM {table} WHERE {col} = ?",
                (f"conv_{pid}",),
            )
        else:
            check = await conn.execute(
                f"SELECT 1 FROM {table} WHERE profile_id = ?", (pid,)
            )
        assert await check.fetchone() is None, f"{table} still has rows after forget_profile"


@pytest.mark.asyncio
async def test_forget_profile_returns_counts():
    from memory.cleanup import CleanupService

    conn = await get_connection()
    pid = "speaker_test02"
    await _seed_profile(conn, pid)

    counts = await CleanupService().forget_profile(pid)
    assert "profiles" in counts
    assert "memories" in counts
    assert "conversations" in counts


@pytest.mark.asyncio
async def test_forget_memory_removes_single_row():
    from memory.cleanup import CleanupService

    conn = await get_connection()
    pid = "speaker_test03"
    await _seed_profile(conn, pid)

    svc = CleanupService()
    ok = await svc.forget_memory(f"mem_{pid}", pid)
    assert ok is True

    row = await conn.execute(
        "SELECT 1 FROM memories WHERE memory_id = ?", (f"mem_{pid}",)
    )
    assert await row.fetchone() is None


@pytest.mark.asyncio
async def test_forget_memory_wrong_profile_returns_false():
    from memory.cleanup import CleanupService

    conn = await get_connection()
    pid = "speaker_test04"
    await _seed_profile(conn, pid)

    svc = CleanupService()
    ok = await svc.forget_memory(f"mem_{pid}", "wrong_profile")
    assert ok is False


@pytest.mark.asyncio
async def test_forget_nonexistent_profile_completes_without_error():
    from memory.cleanup import CleanupService
    # Should not raise even if profile doesn't exist
    counts = await CleanupService().forget_profile("nonexistent_profile_xyz")
    assert isinstance(counts, dict)
