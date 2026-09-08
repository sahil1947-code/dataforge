"""
database/db.py
--------------
Async SQLite connection manager for RIME.

All database access in the application goes through get_db() or the
Database singleton.  The connection is configured with:
  • WAL journal mode for concurrent reads alongside writes
  • foreign_keys = ON (SQLite default is OFF)
  • row_factory = aiosqlite.Row so rows are accessible by column name
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiosqlite

from config import get_settings

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Module-level singleton connection (reused across the application lifetime).
_connection: Optional[aiosqlite.Connection] = None
_lock = asyncio.Lock()


def _get_db_path() -> str:
    return str(get_settings().app.database_path)


async def _apply_schema(conn: aiosqlite.Connection) -> None:
    """Execute schema.sql to create all tables if they do not yet exist."""
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    await conn.executescript(schema_sql)
    await conn.commit()


async def get_connection() -> aiosqlite.Connection:
    """Return (and lazily initialize) the singleton database connection."""
    global _connection
    async with _lock:
        if _connection is None:
            db_path = _get_db_path()
            if db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = await aiosqlite.connect(db_path, timeout=30.0)
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA foreign_keys = ON")
            await conn.execute("PRAGMA busy_timeout = 30000")
            if db_path != ":memory:":
                await conn.execute("PRAGMA journal_mode = WAL")
                await conn.execute("PRAGMA synchronous = NORMAL")
            await _apply_schema(conn)
            _connection = conn
            logger.info("Database connection initialized at %s", db_path)
    return _connection


async def close_connection() -> None:
    """Cleanly close the singleton connection (called on app shutdown)."""
    global _connection
    async with _lock:
        if _connection is not None:
            await _connection.close()
            _connection = None
            logger.info("Database connection closed.")


@asynccontextmanager
async def transaction() -> AsyncIterator[aiosqlite.Connection]:
    """
    Async context manager that wraps a block in a BEGIN/COMMIT transaction.
    Rolls back automatically on any exception.

    Usage:
        async with transaction() as conn:
            await conn.execute(...)
    """
    conn = await get_connection()
    try:
        yield conn
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise


# ---------------------------------------------------------------------------
# Convenience helpers used throughout the codebase
# ---------------------------------------------------------------------------

async def fetch_one(
    query: str, params: tuple[Any, ...] = ()
) -> Optional[aiosqlite.Row]:
    conn = await get_connection()
    async with conn.execute(query, params) as cursor:
        return await cursor.fetchone()


async def fetch_all(
    query: str, params: tuple[Any, ...] = ()
) -> list[aiosqlite.Row]:
    conn = await get_connection()
    async with conn.execute(query, params) as cursor:
        return await cursor.fetchall()


async def execute(query: str, params: tuple[Any, ...] = ()) -> int:
    """Execute a single DML statement and return lastrowid (for INSERTs)."""
    conn = await get_connection()
    async with conn.execute(query, params) as cursor:
        await conn.commit()
        return cursor.lastrowid or 0


async def executemany(query: str, params_seq: list[tuple[Any, ...]]) -> None:
    conn = await get_connection()
    await conn.executemany(query, params_seq)
    await conn.commit()


def row_to_dict(row: Optional[aiosqlite.Row]) -> Optional[dict[str, Any]]:
    """Convert an aiosqlite.Row to a plain dict, or return None."""
    if row is None:
        return None
    return dict(row)


def rows_to_dicts(rows: list[aiosqlite.Row]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows]


def encode_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def decode_json(text: str) -> Any:
    if not text:
        return {}
    return json.loads(text)
