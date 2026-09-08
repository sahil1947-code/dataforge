-- Migration 001 — initial schema
-- Idempotent: safe to run on an already-initialized database.
-- This mirrors schema.sql exactly; subsequent migrations add only new tables/columns.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    schema_version  INTEGER NOT NULL
);
INSERT OR IGNORE INTO schema_meta (id, schema_version) VALUES (1, 1);

-- All table creation is handled by schema.sql on first run.
-- This migration file is kept for historical reference and rollback tooling.
UPDATE schema_meta SET schema_version = 1 WHERE id = 1;
