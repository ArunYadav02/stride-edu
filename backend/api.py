"""
Stride API.

Auth notes, because this is where student projects usually go wrong:

- Passwords are bcrypt-hashed. Never stored, never logged, never returned.
- JWTs are signed with a secret read from the environment. The app refuses to
  start in production without one rather than falling back to a default — a
  hardcoded fallback secret is the single most common way a project like this
  gets compromised.
- **Search requires no account.** Auth gates only saving questions and uploading
  personal papers. Requiring a login to search would cost most of the users.
"""

from __future__ import annotations

import os
import secrets
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any

import jwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import bcrypt
from pydantic import BaseModel, EmailStr, Field

from .db import DB_PATH, connect, init_db, log_search, now
from .search import (
    SearchFilters, corpus_stats, search_questions, topic_overview,
    topic_profile, usage_stats,
)
from .topics import ONTOLOGY, correct_tag, tagger_accuracy

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

IS_PROD = os.environ.get("STRIDE_ENV") == "production"
SECRET = os.environ.get("STRIDE_SECRET")
if not SECRET:
    if IS_PROD:
        raise RuntimeError(
            "STRIDE_SECRET must be set in production. Generate one with: "
            "python -c 'import secrets; print(secrets.token_hex(32))'"
        )
    SECRET = "dev-only-not-for-production-" + secrets.token_hex(8)

ALGORITHM = "HS256"
TOKEN_HOURS = 24 * 14
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

# bcrypt directly rather than via passlib: bcrypt hashes at most 72 bytes and
# raises on anything longer, which turns a long passphrase into a 500 error.
# Truncating explicitly is the documented, correct handling — and a passphrase
# is exactly what a security-conscious user is most likely to type.
_BCRYPT_MAX = 72


def _pw_bytes(password: str) -> bytes:
    return password.encode("utf-8")[:_BCRYPT_MAX]


class _Pwd:
    @staticmethod
    def hash(password: str) -> str:
        return bcrypt.hashpw(_pw_bytes(password), bcrypt.gensalt()).decode()

    @staticmethod
    def verify(password: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(_pw_bytes(password), hashed.encode())
        except (ValueError, TypeError):
            return False


pwd = _Pwd()

app = FastAPI(
    title="Stride",
    description="Search past exam questions by topic, year and marks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if not IS_PROD else [os.environ.get("STRIDE_ORIGIN", "")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class SignUp(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    display_name: str | None = Field(default=None, max_length=60)


class SignIn(BaseModel):
    email: EmailStr
    password: str


class TokenOut(BaseModel):
    token: str
    email: str
    display_name: str | None


class SaveIn(BaseModel):
    question_id: int
    note: str | None = Field(default=None, max_length=500)


class TagCorrection(BaseModel):
    question_id: int
    topic: str
    correct: bool


class FeedbackIn(BaseModel):
    kind: str = Field(pattern="^(wrong_topic|bad_split|other)$")
    question_id: int | None = None
    message: str | None = Field(default=None, max_length=1000)


# --------------------------------------------------------------------------
# Auth helpers
# --------------------------------------------------------------------------


def make_token(user_id: int, email: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


def current_user_optional(request: Request) -> int | None:
    """Resolve a user if a valid token is present. Never raises.

    Used by search so that signed-in users additionally see their own uploads,
    while anonymous users get the public corpus.
    """
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        return None
    try:
        payload = jwt.decode(auth[7:], SECRET, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except Exception:
        return None


def current_user(request: Request) -> int:
    uid = current_user_optional(request)
    if uid is None:
        raise HTTPException(401, "Sign in to do that.")
    return uid


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


@app.on_event("startup")
def _startup() -> None:
    init_db(DB_PATH)


@app.exception_handler(sqlite3.IntegrityError)
def _integrity(request: Request, exc: sqlite3.IntegrityError) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": "That already exists."})


# --------------------------------------------------------------------------
# Auth routes
# --------------------------------------------------------------------------


@app.post("/api/auth/signup", response_model=TokenOut)
def signup(body: SignUp) -> Any:
    with connect() as conn:
        existing = conn.execute(
            "SELECT id FROM user WHERE email = ?", (body.email.lower(),)
        ).fetchone()
        if existing:
            raise HTTPException(409, "An account with that email already exists.")
        cur = conn.execute(
            "INSERT INTO user (email, password_hash, display_name, created_at, last_seen_at)"
            " VALUES (?,?,?,?,?)",
            (body.email.lower(), pwd.hash(body.password), body.display_name, now(), now()),
        )
        uid = cur.lastrowid
    return TokenOut(token=make_token(uid, body.email.lower()),
                    email=body.email.lower(), display_name=body.display_name)


@app.post("/api/auth/signin", response_model=TokenOut)
def signin(body: SignIn) -> Any:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, email, password_hash, display_name FROM user WHERE email = ?",
            (body.email.lower(),),
        ).fetchone()
        # Same message either way: revealing which emails exist is an
        # enumeration vector.
        if not row or not pwd.verify(body.password, row["password_hash"]):
            raise HTTPException(401, "Email or password is incorrect.")
        conn.execute("UPDATE user SET last_seen_at = ? WHERE id = ?", (now(), row["id"]))
    return TokenOut(token=make_token(row["id"], row["email"]),
                    email=row["email"], display_name=row["display_name"])


@app.get("/api/auth/me")
def me(uid: Annotated[int, Depends(current_user)]) -> Any:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, email, display_name, created_at FROM user WHERE id = ?", (uid,)
        ).fetchone()
        saved = conn.execute(
            "SELECT COUNT(*) n FROM saved_question WHERE user_id = ?", (uid,)
        ).fetchone()["n"]
    return {**dict(row), "saved_count": saved}


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


@app.get("/api/search")
def search(
    request: Request,
    q: str = Query(default="", max_length=200),
    institution_id: int | None = None,
    module_id: int | None = None,
    topic: str | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    marks_min: int | None = None,
    marks_max: int | None = None,
    limit: int = Query(default=30, le=100),
    offset: int = 0,
) -> Any:
    uid = current_user_optional(request)
    filters = SearchFilters(
        institution_id=institution_id, module_id=module_id, topic=topic,
        year_from=year_from, year_to=year_to, marks_min=marks_min,
        marks_max=marks_max, include_personal_for_user=uid,
    )
    with connect() as conn:
        out = search_questions(conn, q, filters, limit=limit, offset=offset)
        if q.strip():
            log_search(conn, q, out["total"], out["duration_ms"], filters.to_json())
    return out


@app.get("/api/question/{question_id}")
def get_question(question_id: int, request: Request) -> Any:
    uid = current_user_optional(request)
    with connect() as conn:
        row = conn.execute(
            """SELECT q.*, p.year, p.session, p.source_url, p.visibility, p.owner_user_id,
                      p.parse_quality, p.parse_note,
                      m.code module_code, m.title module_title,
                      i.short_name institution, i.permission_status
               FROM question q
               JOIN paper p ON p.id = q.paper_id
               JOIN module m ON m.id = p.module_id
               JOIN institution i ON i.id = m.institution_id
               WHERE q.id = ?""",
            (question_id,),
        ).fetchone()
        if not row:
            raise HTTPException(404, "No question with that id.")
        if row["visibility"] == "personal" and row["owner_user_id"] != uid:
            raise HTTPException(404, "No question with that id.")

        siblings = conn.execute(
            "SELECT id, number, text, marks FROM question"
            " WHERE paper_id = ? ORDER BY position",
            (row["paper_id"],),
        ).fetchall()
        topics = conn.execute(
            "SELECT t.name, qt.confidence, qt.source FROM question_topic qt"
            " JOIN topic t ON t.id = qt.topic_id WHERE qt.question_id = ?",
            (question_id,),
        ).fetchall()
    return {
        **dict(row),
        "topics": [dict(t) for t in topics],
        "paper_questions": [dict(s) for s in siblings],
    }


# --------------------------------------------------------------------------
# Topics — the headline feature
# --------------------------------------------------------------------------


@app.get("/api/topics")
def topics(module_id: int | None = None) -> Any:
    with connect() as conn:
        return {
            "topics": topic_overview(conn, module_id),
            "ontology": {
                name: {"parent": spec["parent"]} for name, spec in ONTOLOGY.items()
            },
            "tagger": tagger_accuracy(conn),
        }


@app.get("/api/topics/{topic_name}")
def topic_detail(topic_name: str, module_id: int | None = None) -> Any:
    with connect() as conn:
        profile = topic_profile(conn, topic_name, module_id)
        if profile["appearances"] == 0:
            return profile
        examples = conn.execute(
            """SELECT q.id, q.number, q.text, q.marks, p.year,
                      m.code module_code, i.short_name institution
               FROM question q
               JOIN paper p ON p.id = q.paper_id
               JOIN module m ON m.id = p.module_id
               JOIN institution i ON i.id = m.institution_id
               JOIN question_topic qt ON qt.question_id = q.id
               JOIN topic t ON t.id = qt.topic_id
               WHERE t.name = ? AND p.visibility='indexed' AND qt.source != 'corrected'
               ORDER BY p.year DESC LIMIT 8""",
            (topic_name,),
        ).fetchall()
    return {**profile, "examples": [dict(e) for e in examples]}


@app.post("/api/topics/correct")
def correct(body: TagCorrection, uid: Annotated[int, Depends(current_user)]) -> Any:
    with connect() as conn:
        correct_tag(conn, body.question_id, body.topic, body.correct)
    return {"ok": True}


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------


@app.get("/api/corpus")
def corpus() -> Any:
    with connect() as conn:
        institutions = conn.execute(
            """SELECT i.id, i.name, i.short_name, i.permission_status,
                      COUNT(DISTINCT m.id) modules,
                      COUNT(DISTINCT p.id) papers
               FROM institution i
               LEFT JOIN module m ON m.institution_id = i.id
               LEFT JOIN paper p ON p.module_id = m.id AND p.visibility='indexed'
               GROUP BY i.id ORDER BY papers DESC"""
        ).fetchall()
        modules = conn.execute(
            """SELECT m.id, m.code, m.title, m.subject, i.short_name institution,
                      COUNT(DISTINCT p.id) papers,
                      MIN(p.year) first_year, MAX(p.year) last_year
               FROM module m
               JOIN institution i ON i.id = m.institution_id
               LEFT JOIN paper p ON p.module_id = m.id AND p.visibility='indexed'
               GROUP BY m.id ORDER BY papers DESC"""
        ).fetchall()
        return {
            "stats": corpus_stats(conn),
            "institutions": [dict(i) for i in institutions],
            "modules": [dict(m) for m in modules],
        }


@app.get("/api/stats")
def stats() -> Any:
    with connect() as conn:
        return {"corpus": corpus_stats(conn), "usage": usage_stats(conn)}


# --------------------------------------------------------------------------
# Saved questions
# --------------------------------------------------------------------------


@app.post("/api/saved")
def save_question(body: SaveIn, uid: Annotated[int, Depends(current_user)]) -> Any:
    with connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO saved_question (user_id, question_id, note, saved_at)"
            " VALUES (?,?,?,?)",
            (uid, body.question_id, body.note, now()),
        )
    return {"ok": True}


@app.delete("/api/saved/{question_id}")
def unsave(question_id: int, uid: Annotated[int, Depends(current_user)]) -> Any:
    with connect() as conn:
        conn.execute(
            "DELETE FROM saved_question WHERE user_id = ? AND question_id = ?",
            (uid, question_id),
        )
    return {"ok": True}


@app.get("/api/saved")
def list_saved(uid: Annotated[int, Depends(current_user)]) -> Any:
    with connect() as conn:
        rows = conn.execute(
            """SELECT q.id, q.number, q.text, q.marks, s.note, s.saved_at,
                      p.year, m.code module_code, i.short_name institution
               FROM saved_question s
               JOIN question q ON q.id = s.question_id
               JOIN paper p ON p.id = q.paper_id
               JOIN module m ON m.id = p.module_id
               JOIN institution i ON i.id = m.institution_id
               WHERE s.user_id = ? ORDER BY s.saved_at DESC""",
            (uid,),
        ).fetchall()
    return {"saved": [dict(r) for r in rows]}


# --------------------------------------------------------------------------
# Personal uploads
# --------------------------------------------------------------------------


@app.post("/api/upload")
async def upload_paper(
    uid: Annotated[int, Depends(current_user)],
    file: UploadFile = File(...),
    module_code: str = Form(...),
    module_title: str = Form(...),
    institution: str = Form(...),
    year: int = Form(...),
) -> Any:
    """Index a paper the user already has.

    Stored as `visibility='personal'` and visible only to the uploader. The PDF
    itself is parsed and discarded — Stride keeps the extracted questions, not
    the file.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a PDF.")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Files must be under {MAX_UPLOAD_BYTES // 1024 // 1024} MB.")

    import tempfile
    from .ingest import parse_paper
    from .db import insert_paper, insert_question, upsert_institution, upsert_module
    from .topics import tag_and_store

    tmp = Path(tempfile.mkdtemp()) / "upload.pdf"
    tmp.write_bytes(data)
    try:
        parsed = parse_paper(tmp)
    finally:
        tmp.unlink(missing_ok=True)
        tmp.parent.rmdir()

    if not parsed.questions:
        raise HTTPException(422, parsed.note)

    with connect() as conn:
        inst_id = upsert_institution(conn, institution, institution[:12])
        mod_id = upsert_module(conn, inst_id, module_code, module_title)
        paper_id = insert_paper(
            conn, mod_id, year, visibility="personal", owner_user_id=uid,
            total_marks=parsed.total_marks, duration_min=parsed.duration_min,
            parse_quality=parsed.confidence, parse_note=parsed.note,
        )
        n = 0
        parent_map: dict[int, int] = {}
        for q in parsed.flat():
            qid = insert_question(
                conn, paper_id, q.number, q.text, q.position,
                marks=q.marks, page=q.page, depth=q.depth,
                parent_id=parent_map.get(q.depth - 1),
            )
            parent_map[q.depth] = qid
            tag_and_store(conn, qid, q.text)
            n += 1

    return {
        "ok": True, "questions_indexed": n,
        "confidence": parsed.confidence, "note": parsed.note,
    }


# --------------------------------------------------------------------------
# Feedback
# --------------------------------------------------------------------------


@app.post("/api/feedback")
def feedback(body: FeedbackIn) -> Any:
    with connect() as conn:
        conn.execute(
            "INSERT INTO feedback (kind, question_id, message, created_at) VALUES (?,?,?,?)",
            (body.kind, body.question_id, body.message, now()),
        )
    return {"ok": True}


@app.get("/api/health")
def health() -> Any:
    return {"ok": True, "time": int(time.time())}


# --------------------------------------------------------------------------
# Static frontend
# --------------------------------------------------------------------------

WEB_DIR = Path(__file__).parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def index() -> Any:
        return FileResponse(str(WEB_DIR / "index.html"))
