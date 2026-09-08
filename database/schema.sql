-- ============================================================
-- RIME — Primary Database Schema
-- Engine: SQLite 3
-- Version: 1 (Phase 1 baseline — all tables)
-- ============================================================
-- Conventions:
--   • All PKs are TEXT (prefixed human-readable IDs or UUIDs)
--   • Timestamps stored as ISO-8601 TEXT
--   • PRAGMA foreign_keys = ON must be set at connection time
-- ============================================================

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

-- ============================================================
-- SCHEMA VERSION TRACKER
-- ============================================================
CREATE TABLE IF NOT EXISTS schema_meta (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version  INTEGER NOT NULL
);
INSERT OR IGNORE INTO schema_meta (id, schema_version) VALUES (1, 1);

-- ============================================================
-- PROFILES — anchor entity for identity, memory, and privacy
-- ============================================================
CREATE TABLE IF NOT EXISTS profiles (
    profile_id          TEXT PRIMARY KEY,
    display_name        TEXT,
    created_at          TEXT NOT NULL,
    last_seen           TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'active'
                          CHECK (status IN ('active', 'privacy_mode', 'disabled')),
    preferences_json    TEXT NOT NULL DEFAULT '{}'
);

-- ============================================================
-- VOICE EMBEDDINGS — one-to-many per profile
-- ============================================================
CREATE TABLE IF NOT EXISTS voice_embeddings (
    embedding_id        TEXT PRIMARY KEY,
    profile_id          TEXT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
    embedding_vector    BLOB NOT NULL,
    quality_score       REAL NOT NULL DEFAULT 0.0,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_voice_embeddings_profile
    ON voice_embeddings(profile_id);

-- ============================================================
-- CONVERSATIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id     TEXT PRIMARY KEY,
    profile_id          TEXT REFERENCES profiles(profile_id) ON DELETE CASCADE,
    started_at          TEXT NOT NULL,
    ended_at            TEXT,
    summary             TEXT,
    session_mode        TEXT NOT NULL DEFAULT 'normal'
                          CHECK (session_mode IN ('normal', 'temporary'))
);
CREATE INDEX IF NOT EXISTS idx_conversations_profile
    ON conversations(profile_id, started_at);

-- ============================================================
-- MESSAGES — append-only transcript per conversation turn
-- ============================================================
CREATE TABLE IF NOT EXISTS messages (
    message_id          TEXT PRIMARY KEY,
    conversation_id     TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role                TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    text                TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    sequence_number     INTEGER NOT NULL,
    generation          INTEGER NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending'
                          CHECK (status IN ('spoken', 'stale_discarded', 'pending'))
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_seq
    ON messages(conversation_id, sequence_number);
CREATE INDEX IF NOT EXISTS idx_messages_generation
    ON messages(conversation_id, generation);

-- ============================================================
-- TASKS — one-time, recurring, and dependent tasks per profile
-- ============================================================
CREATE TABLE IF NOT EXISTS tasks (
    task_id             TEXT PRIMARY KEY,
    profile_id          TEXT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
    title               TEXT NOT NULL,
    description         TEXT,
    status              TEXT NOT NULL DEFAULT 'created'
                          CHECK (status IN (
                              'created','scheduled','waiting',
                              'triggered','executed','completed','cancelled'
                          )),
    created_at          TEXT NOT NULL,
    due_at              TEXT,
    recurrence_rule     TEXT,
    depends_on_task_id  TEXT REFERENCES tasks(task_id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_profile_status_due
    ON tasks(profile_id, status, due_at);

-- ============================================================
-- MEMORIES — leveled 0-3 (TRD §3.11 memory model)
-- ============================================================
CREATE TABLE IF NOT EXISTS memories (
    memory_id           TEXT PRIMARY KEY,
    profile_id          TEXT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
    category            TEXT NOT NULL,
    content             TEXT NOT NULL,
    importance          INTEGER NOT NULL DEFAULT 1,
    memory_level        INTEGER NOT NULL DEFAULT 1
                          CHECK (memory_level BETWEEN 0 AND 3),
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    expires_at          TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_profile_category_level
    ON memories(profile_id, category, memory_level);

-- ============================================================
-- TOOL CALLS — every invocation tagged with generation ID
-- ============================================================
CREATE TABLE IF NOT EXISTS tool_calls (
    tool_call_id        TEXT PRIMARY KEY,
    conversation_id     TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    tool_name           TEXT NOT NULL,
    arguments           TEXT NOT NULL DEFAULT '{}',
    status              TEXT NOT NULL DEFAULT 'started'
                          CHECK (status IN (
                              'started','completed','failed','timed_out','stale'
                          )),
    generation          INTEGER NOT NULL,
    started_at          TEXT NOT NULL,
    completed_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_conversation_generation
    ON tool_calls(conversation_id, generation);

-- ============================================================
-- EVENTS — general-purpose audit/observability log
-- ============================================================
CREATE TABLE IF NOT EXISTS events (
    event_id            TEXT PRIMARY KEY,
    profile_id          TEXT REFERENCES profiles(profile_id) ON DELETE CASCADE,
    event_type          TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    metadata            TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_profile_timestamp
    ON events(profile_id, timestamp);

-- ============================================================
-- EMERGENCY EVENTS — safety-critical, separately retained
-- ============================================================
CREATE TABLE IF NOT EXISTS emergency_events (
    event_id            TEXT PRIMARY KEY,
    profile_id          TEXT REFERENCES profiles(profile_id) ON DELETE CASCADE,
    timestamp           TEXT NOT NULL,
    trigger_type        TEXT NOT NULL,
    confidence          REAL NOT NULL,
    action_taken        TEXT NOT NULL DEFAULT 'none'
                          CHECK (action_taken IN (
                              'none','confirmed_activated','cancelled'
                          ))
);
CREATE INDEX IF NOT EXISTS idx_emergency_events_profile
    ON emergency_events(profile_id, timestamp);
