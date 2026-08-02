"""
Stride — storage layer.

One design decision shapes this whole file: **Stride indexes, it does not
redistribute.** Exam papers are university copyright. A system whose value comes
from rehosting other institutions' PDFs is one takedown notice away from
disappearing, and it puts the person who built it in an awkward conversation with
their department.

So the schema stores what an index needs — extracted question text, topics,
marks, provenance — and a `source_url` pointing at the official page. There is
deliberately no `pdf_blob` column and no file store for source papers. The
constraint is in the data model rather than in a policy document, because policy
documents do not survive contact with a late-night feature idea.

Users can add papers they already have: those are processed locally, indexed the
same way, and marked with a visibility that keeps them out of the shared corpus.

SQLite with FTS5 rather than Postgres + pgvector: it is a single file, needs no
service, runs on free hosting, and full-text search over a few hundred thousand
questions is comfortably sub-100ms. The search layer is deliberately swappable if
this ever outgrows one file.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

DB_PATH = Path("stride.db")


SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- institutions
CREATE TABLE IF NOT EXISTS institution (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    short_name  TEXT NOT NULL,
    country     TEXT NOT NULL DEFAULT 'GB',
    -- Whether this institution has confirmed they are happy to be indexed.
    -- Displayed in the UI so the provenance of every result is legible.
    permission_status TEXT NOT NULL DEFAULT 'not_asked'
        CHECK (permission_status IN ('not_asked','requested','granted','declined')),
    permission_note   TEXT,
    created_at  INTEGER NOT NULL
);

-- --------------------------------------------------------------------- modules
CREATE TABLE IF NOT EXISTS module (
    id              INTEGER PRIMARY KEY,
    institution_id  INTEGER NOT NULL REFERENCES institution(id) ON DELETE CASCADE,
    code            TEXT NOT NULL,          -- e.g. COMP0005
    title           TEXT NOT NULL,
    level           TEXT,                   -- undergraduate year / masters
    subject         TEXT NOT NULL DEFAULT 'Computer Science',
    created_at      INTEGER NOT NULL,
    UNIQUE (institution_id, code)
);

-- ---------------------------------------------------------------------- papers
CREATE TABLE IF NOT EXISTS paper (
    id           INTEGER PRIMARY KEY,
    module_id    INTEGER NOT NULL REFERENCES module(id) ON DELETE CASCADE,
    year         INTEGER NOT NULL,
    session      TEXT NOT NULL DEFAULT 'main',   -- main | resit | mock
    -- The official page this paper lives on. Never a copy of the file itself.
    source_url   TEXT,
    -- 'indexed'  : text extracted from a publicly reachable paper, links out
    -- 'personal' : uploaded by a user, visible only to them
    visibility   TEXT NOT NULL DEFAULT 'indexed'
        CHECK (visibility IN ('indexed','personal')),
    owner_user_id INTEGER REFERENCES user(id) ON DELETE CASCADE,
    total_marks  INTEGER,
    duration_min INTEGER,
    ingested_at  INTEGER NOT NULL,
    -- How well segmentation went, surfaced in the UI rather than hidden.
    parse_quality REAL NOT NULL DEFAULT 1.0,
    parse_note    TEXT,
    UNIQUE (module_id, year, session, owner_user_id)
);

-- ------------------------------------------------------------------- questions
CREATE TABLE IF NOT EXISTS question (
    id          INTEGER PRIMARY KEY,
    paper_id    INTEGER NOT NULL REFERENCES paper(id) ON DELETE CASCADE,
    number      TEXT NOT NULL,        -- "3", "3(a)", "3(a)(ii)"
    parent_id   INTEGER REFERENCES question(id) ON DELETE CASCADE,
    depth       INTEGER NOT NULL DEFAULT 0,
    text        TEXT NOT NULL,
    marks       INTEGER,
    page        INTEGER,
    position    INTEGER NOT NULL,     -- order within the paper
    created_at  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_question_paper ON question(paper_id, position);
CREATE INDEX IF NOT EXISTS idx_paper_module ON paper(module_id, year);

-- ---------------------------------------------------------------------- topics
CREATE TABLE IF NOT EXISTS topic (
    id       INTEGER PRIMARY KEY,
    name     TEXT NOT NULL UNIQUE,
    subject  TEXT NOT NULL DEFAULT 'Computer Science',
    parent   TEXT
);

CREATE TABLE IF NOT EXISTS question_topic (
    question_id INTEGER NOT NULL REFERENCES question(id) ON DELETE CASCADE,
    topic_id    INTEGER NOT NULL REFERENCES topic(id) ON DELETE CASCADE,
    confidence  REAL NOT NULL DEFAULT 0.5,
    -- 'auto'      : assigned by the tagger
    -- 'confirmed' : a human agreed
    -- 'corrected' : a human overrode the tagger. These become training data.
    source      TEXT NOT NULL DEFAULT 'auto'
        CHECK (source IN ('auto','confirmed','corrected')),
    PRIMARY KEY (question_id, topic_id)
);

CREATE INDEX IF NOT EXISTS idx_qt_topic ON question_topic(topic_id);

-- ----------------------------------------------------------------------- users
CREATE TABLE IF NOT EXISTS user (
    id            INTEGER PRIMARY KEY,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name  TEXT,
    institution_id INTEGER REFERENCES institution(id) ON DELETE SET NULL,
    created_at    INTEGER NOT NULL,
    last_seen_at  INTEGER
);

-- Saved questions, the one feature that genuinely needs an account.
CREATE TABLE IF NOT EXISTS saved_question (
    user_id     INTEGER NOT NULL REFERENCES user(id) ON DELETE CASCADE,
    question_id INTEGER NOT NULL REFERENCES question(id) ON DELETE CASCADE,
    note        TEXT,
    saved_at    INTEGER NOT NULL,
    PRIMARY KEY (user_id, question_id)
);

-- ------------------------------------------------------------------- analytics
-- Privacy-respecting: no IP, no user agent, no cross-session identifier.
-- Enough to say "N searches this week, these were the popular queries" and
-- nothing that could re-identify a student.
CREATE TABLE IF NOT EXISTS search_event (
    id          INTEGER PRIMARY KEY,
    query       TEXT NOT NULL,
    n_results   INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    day         TEXT NOT NULL,        -- YYYY-MM-DD, no finer granularity
    filters     TEXT                  -- JSON, for understanding usage
);

CREATE INDEX IF NOT EXISTS idx_search_day ON search_event(day);

CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY,
    kind        TEXT NOT NULL,        -- wrong_topic | bad_split | other
    question_id INTEGER REFERENCES question(id) ON DELETE SET NULL,
    message     TEXT,
    created_at  INTEGER NOT NULL,
    resolved    INTEGER NOT NULL DEFAULT 0
);

-- --------------------------------------------------------------- full text idx
-- FTS5 over question text. `content=` keeps it as an external-content table so
-- the text is not duplicated; triggers keep it in sync.
CREATE VIRTUAL TABLE IF NOT EXISTS question_fts USING fts5(
    text,
    content='question',
    content_rowid='id',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS question_ai AFTER INSERT ON question BEGIN
    INSERT INTO question_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS question_ad AFTER DELETE ON question BEGIN
    INSERT INTO question_fts(question_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS question_au AFTER UPDATE ON question BEGIN
    INSERT INTO question_fts(question_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO question_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def now() -> int:
    return int(time.time())


@contextmanager
def connect(path: Path | str = DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(path: Path | str = DB_PATH) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------
# Insert helpers
# --------------------------------------------------------------------------


def upsert_institution(
    conn: sqlite3.Connection,
    name: str,
    short_name: str,
    permission_status: str = "not_asked",
    permission_note: str | None = None,
) -> int:
    cur = conn.execute("SELECT id FROM institution WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO institution (name, short_name, permission_status, permission_note, created_at)"
        " VALUES (?,?,?,?,?)",
        (name, short_name, permission_status, permission_note, now()),
    )
    return cur.lastrowid


def upsert_module(
    conn: sqlite3.Connection, institution_id: int, code: str, title: str,
    level: str | None = None, subject: str = "Computer Science",
) -> int:
    cur = conn.execute(
        "SELECT id FROM module WHERE institution_id = ? AND code = ?",
        (institution_id, code),
    )
    row = cur.fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO module (institution_id, code, title, level, subject, created_at)"
        " VALUES (?,?,?,?,?,?)",
        (institution_id, code, title, level, subject, now()),
    )
    return cur.lastrowid


def insert_paper(
    conn: sqlite3.Connection, module_id: int, year: int, session: str = "main",
    source_url: str | None = None, visibility: str = "indexed",
    owner_user_id: int | None = None, total_marks: int | None = None,
    duration_min: int | None = None, parse_quality: float = 1.0,
    parse_note: str | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO paper (module_id, year, session, source_url, visibility,"
        " owner_user_id, total_marks, duration_min, ingested_at, parse_quality, parse_note)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (module_id, year, session, source_url, visibility, owner_user_id,
         total_marks, duration_min, now(), parse_quality, parse_note),
    )
    return cur.lastrowid


def insert_question(
    conn: sqlite3.Connection, paper_id: int, number: str, text: str,
    position: int, marks: int | None = None, page: int | None = None,
    parent_id: int | None = None, depth: int = 0,
) -> int:
    cur = conn.execute(
        "INSERT INTO question (paper_id, number, parent_id, depth, text, marks, page, position, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        (paper_id, number, parent_id, depth, text, marks, page, position, now()),
    )
    return cur.lastrowid


def upsert_topic(conn: sqlite3.Connection, name: str, parent: str | None = None,
                 subject: str = "Computer Science") -> int:
    cur = conn.execute("SELECT id FROM topic WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur = conn.execute(
        "INSERT INTO topic (name, subject, parent) VALUES (?,?,?)", (name, subject, parent)
    )
    return cur.lastrowid


def tag_question(conn: sqlite3.Connection, question_id: int, topic_id: int,
                 confidence: float = 0.5, source: str = "auto") -> None:
    conn.execute(
        "INSERT OR REPLACE INTO question_topic (question_id, topic_id, confidence, source)"
        " VALUES (?,?,?,?)",
        (question_id, topic_id, confidence, source),
    )


def log_search(conn: sqlite3.Connection, query: str, n_results: int,
               duration_ms: float, filters: dict[str, Any] | None = None) -> None:
    """Record a search with no identifying information beyond the calendar day."""
    conn.execute(
        "INSERT INTO search_event (query, n_results, duration_ms, day, filters)"
        " VALUES (?,?,?,date('now'),?)",
        (query[:200], n_results, duration_ms, json.dumps(filters or {})),
    )
