"""
Search, and the analytics that are the actual reason to build this.

Search is FTS5 with BM25 ranking plus filters. Nothing exotic, and deliberately
so — sub-100ms on a laptop with no service to run, and a search box that returns
instantly is worth more than a semantically clever one that takes two seconds.

The analytics are where Stride earns its place. Anyone can build a search box
over PDFs. Almost nobody can answer:

    "Dynamic programming appeared in 8 of the last 10 papers, usually as
     question 3, typically worth 20 marks."

That falls straight out of having questions tagged, positioned and marked, and
it is the thing a student revising actually wants to know. `topic_profile` is
the headline feature; treat it as the product.
"""

from __future__ import annotations

import re
import sqlite3
import statistics
import time
from dataclasses import dataclass
from typing import Any


def _fts_query(raw: str) -> str:
    """Turn user input into a safe FTS5 MATCH expression.

    FTS5 has its own syntax; passing user text straight through means a stray
    quote or `NEAR` throws a parse error at the person searching. Quote each
    token, and add a prefix match on the last one so results update as they type.
    """
    tokens = re.findall(r"[A-Za-z0-9_'-]+", raw.lower())
    tokens = [t for t in tokens if len(t) > 1][:12]
    if not tokens:
        return ""
    # OR rather than AND. Requiring every token means "dynamic programming
    # knapsack" returns nothing unless one question contains all three words,
    # which is exactly the query a student would type and exactly the moment an
    # empty result set makes the tool feel broken. BM25 already ranks documents
    # matching more terms higher, so OR loses precision at the top of the list
    # and gains a great deal of recall below it.
    quoted = [f'"{t}"' for t in tokens[:-1]]
    quoted.append(f'"{tokens[-1]}"*')
    return " OR ".join(quoted)


@dataclass
class SearchFilters:
    institution_id: int | None = None
    module_id: int | None = None
    topic: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    marks_min: int | None = None
    marks_max: int | None = None
    include_personal_for_user: int | None = None

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


def search_questions(
    conn: sqlite3.Connection,
    query: str,
    filters: SearchFilters | None = None,
    limit: int = 30,
    offset: int = 0,
) -> dict[str, Any]:
    f = filters or SearchFilters()
    started = time.perf_counter()

    where: list[str] = []
    params: list[Any] = []

    # Personal papers are visible only to their owner. This is enforced in every
    # query rather than by convention, because a leak here means showing someone
    # else's uploaded material.
    if f.include_personal_for_user is not None:
        where.append("(p.visibility = 'indexed' OR p.owner_user_id = ?)")
        params.append(f.include_personal_for_user)
    else:
        where.append("p.visibility = 'indexed'")

    if f.institution_id:
        where.append("m.institution_id = ?"); params.append(f.institution_id)
    if f.module_id:
        where.append("p.module_id = ?"); params.append(f.module_id)
    if f.year_from:
        where.append("p.year >= ?"); params.append(f.year_from)
    if f.year_to:
        where.append("p.year <= ?"); params.append(f.year_to)
    if f.marks_min:
        where.append("q.marks >= ?"); params.append(f.marks_min)
    if f.marks_max:
        where.append("q.marks <= ?"); params.append(f.marks_max)
    if f.topic:
        where.append(
            "EXISTS (SELECT 1 FROM question_topic qt JOIN topic t ON t.id = qt.topic_id"
            " WHERE qt.question_id = q.id AND t.name = ? AND qt.source != 'corrected')"
        )
        params.append(f.topic)

    match = _fts_query(query)
    if match:
        base = """
        SELECT q.id, q.number, q.text, q.marks, q.page, q.depth,
               p.year, p.session, p.source_url, p.parse_quality,
               m.code module_code, m.title module_title, m.id module_id,
               i.short_name institution, i.id institution_id,
               bm25(question_fts) AS rank
        FROM question_fts
        JOIN question q ON q.id = question_fts.rowid
        JOIN paper p ON p.id = q.paper_id
        JOIN module m ON m.id = p.module_id
        JOIN institution i ON i.id = m.institution_id
        WHERE question_fts MATCH ?
        """
        params = [match] + params
        order = "ORDER BY rank"
    else:
        base = """
        SELECT q.id, q.number, q.text, q.marks, q.page, q.depth,
               p.year, p.session, p.source_url, p.parse_quality,
               m.code module_code, m.title module_title, m.id module_id,
               i.short_name institution, i.id institution_id,
               0 AS rank
        FROM question q
        JOIN paper p ON p.id = q.paper_id
        JOIN module m ON m.id = p.module_id
        JOIN institution i ON i.id = m.institution_id
        WHERE 1=1
        """
        order = "ORDER BY p.year DESC, q.position"

    sql = base + ("" if not where else " AND " + " AND ".join(where))
    count_sql = f"SELECT COUNT(*) n FROM ({sql})"
    total = conn.execute(count_sql, params).fetchone()["n"]

    rows = conn.execute(
        f"{sql} {order} LIMIT ? OFFSET ?", params + [limit, offset]
    ).fetchall()

    results = []
    for r in rows:
        d = dict(r)
        d["topics"] = _topics_for(conn, r["id"])
        d["snippet"] = _snippet(r["text"], query)
        results.append(d)

    elapsed = (time.perf_counter() - started) * 1000
    return {
        "query": query,
        "total": total,
        "results": results,
        "duration_ms": round(elapsed, 1),
        "filters": f.to_json(),
    }


def _topics_for(conn: sqlite3.Connection, question_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT t.name, qt.confidence, qt.source FROM question_topic qt"
        " JOIN topic t ON t.id = qt.topic_id"
        " WHERE qt.question_id = ? AND qt.source != 'corrected'"
        " ORDER BY qt.confidence DESC",
        (question_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def _snippet(text: str, query: str, width: int = 260) -> str:
    """A window around the first query term, so results show why they matched."""
    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 2]
    lowered = text.lower()
    pos = -1
    for t in tokens:
        pos = lowered.find(t)
        if pos != -1:
            break
    if pos == -1:
        return text[:width] + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


# --------------------------------------------------------------------------
# The headline feature
# --------------------------------------------------------------------------


def topic_profile(
    conn: sqlite3.Connection, topic: str, module_id: int | None = None
) -> dict[str, Any]:
    """Everything a revising student wants to know about one topic.

    How often it appears, whether it is trending, where in the paper it usually
    sits, and what it is typically worth. This is the query that justifies the
    whole ingestion pipeline.
    """
    params: list[Any] = [topic]
    module_clause = ""
    if module_id:
        module_clause = " AND p.module_id = ?"
        params.append(module_id)

    rows = conn.execute(
        f"""
        SELECT p.year, p.id paper_id, q.id qid, q.number, q.marks, q.position,
               m.code module_code, m.id module_id
        FROM question q
        JOIN paper p ON p.id = q.paper_id
        JOIN module m ON m.id = p.module_id
        JOIN question_topic qt ON qt.question_id = q.id
        JOIN topic t ON t.id = qt.topic_id
        WHERE t.name = ? AND p.visibility = 'indexed' AND qt.source != 'corrected'
        {module_clause}
        ORDER BY p.year
        """,
        params,
    ).fetchall()

    if not rows:
        return {"topic": topic, "appearances": 0, "note": "Not seen in any indexed paper yet."}

    # Denominator: papers where this topic *could* have appeared. Counting every
    # paper in the corpus makes "3 of the last 37" — technically true and
    # useless, since an operating systems paper was never going to examine
    # dynamic programming. Restricting to the modules that actually teach the
    # topic gives the number a student needs: 3 of the 7 Algorithms papers.
    relevant_modules = sorted({r["module_id"] for r in rows})
    if module_id:
        total_clause = " WHERE module_id = ? AND visibility='indexed'"
        total_params: list[Any] = [module_id]
    else:
        placeholders = ",".join("?" * len(relevant_modules))
        total_clause = f" WHERE module_id IN ({placeholders}) AND visibility='indexed'"
        total_params = list(relevant_modules)
    total_papers = conn.execute(
        f"SELECT COUNT(*) n FROM paper{total_clause}", total_params
    ).fetchone()["n"]

    papers_with = len({r["paper_id"] for r in rows})
    years = sorted({r["year"] for r in rows})
    by_year = {}
    for r in rows:
        by_year.setdefault(r["year"], 0)
        by_year[r["year"]] += 1

    all_years = conn.execute(
        f"SELECT DISTINCT year FROM paper{total_clause} ORDER BY year", total_params
    ).fetchall()
    year_series = [
        {"year": y["year"], "count": by_year.get(y["year"], 0)} for y in all_years
    ]

    # Typical position: are these usually early or late in the paper?
    positions = [r["position"] for r in rows]
    numbers = [r["number"].split("(")[0] for r in rows]
    common_number = Counter_most(numbers)

    marks = [r["marks"] for r in rows if r["marks"]]

    # Trend: compare the recent half against the earlier half. Requires enough
    # years to be meaningful — with three papers this is noise, and saying so is
    # better than drawing a confident arrow.
    trend = None
    if len(year_series) >= 4:
        mid = len(year_series) // 2
        early = sum(p["count"] for p in year_series[:mid]) / max(1, mid)
        late = sum(p["count"] for p in year_series[mid:]) / max(1, len(year_series) - mid)
        if late > early * 1.3:
            trend = "rising"
        elif late < early * 0.7:
            trend = "falling"
        else:
            trend = "steady"

    return {
        "topic": topic,
        "appearances": len(rows),
        "papers_with_topic": papers_with,
        "total_papers": total_papers,
        "appearance_rate": round(papers_with / total_papers, 3) if total_papers else 0,
        "years": years,
        "year_series": year_series,
        "typical_question": common_number,
        "median_position": int(statistics.median(positions)) if positions else None,
        "marks": {
            "median": int(statistics.median(marks)) if marks else None,
            "min": min(marks) if marks else None,
            "max": max(marks) if marks else None,
            "total_available": sum(marks) if marks else None,
        },
        "trend": trend,
        "trend_note": (
            None if trend else "Too few papers to call a trend."
        ),
        "summary": _profile_sentence(topic, papers_with, total_papers, common_number, marks),
    }


def Counter_most(items: list[str]) -> str | None:
    if not items:
        return None
    counts: dict[str, int] = {}
    for i in items:
        counts[i] = counts.get(i, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _profile_sentence(
    topic: str, papers_with: int, total: int, common_number: str | None, marks: list[int]
) -> str:
    """The one-line answer, written the way a student would say it."""
    bits = [f"{topic} appeared in {papers_with} of the last {total} papers"
            if total else f"{topic} appeared in {papers_with} papers"]
    if common_number:
        bits.append(f"most often as question {common_number}")
    if marks:
        bits.append(f"typically worth {int(statistics.median(marks))} marks")
    return ", ".join(bits) + "."


def topic_overview(conn: sqlite3.Connection, module_id: int | None = None) -> list[dict[str, Any]]:
    """Every topic ranked by how much it is worth studying."""
    params: list[Any] = []
    clause = "WHERE p.visibility='indexed' AND qt.source != 'corrected'"
    if module_id:
        clause += " AND p.module_id = ?"
        params.append(module_id)

    rows = conn.execute(
        f"""
        SELECT t.name topic, t.parent,
               COUNT(DISTINCT q.id) questions,
               COUNT(DISTINCT p.id) papers,
               SUM(COALESCE(q.marks,0)) total_marks,
               MAX(p.year) last_seen
        FROM question_topic qt
        JOIN topic t ON t.id = qt.topic_id
        JOIN question q ON q.id = qt.question_id
        JOIN paper p ON p.id = q.paper_id
        {clause}
        GROUP BY t.name
        ORDER BY total_marks DESC, questions DESC
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


def corpus_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    def one(sql: str) -> int:
        return conn.execute(sql).fetchone()["n"]

    return {
        "institutions": one("SELECT COUNT(*) n FROM institution"),
        "modules": one("SELECT COUNT(*) n FROM module"),
        "papers": one("SELECT COUNT(*) n FROM paper WHERE visibility='indexed'"),
        "questions": one("SELECT COUNT(*) n FROM question"),
        "topics": one("SELECT COUNT(*) n FROM topic"),
        "years": [
            r["year"] for r in conn.execute(
                "SELECT DISTINCT year FROM paper WHERE visibility='indexed' ORDER BY year"
            ).fetchall()
        ],
        "low_confidence_papers": one(
            "SELECT COUNT(*) n FROM paper WHERE parse_quality < 0.5"
        ),
    }


def usage_stats(conn: sqlite3.Connection, days: int = 30) -> dict[str, Any]:
    """Aggregate usage. No per-user data exists to report."""
    rows = conn.execute(
        "SELECT day, COUNT(*) searches, AVG(duration_ms) avg_ms"
        " FROM search_event WHERE day >= date('now', ?)"
        " GROUP BY day ORDER BY day",
        (f"-{days} days",),
    ).fetchall()
    popular = conn.execute(
        "SELECT query, COUNT(*) n FROM search_event"
        " WHERE day >= date('now', ?) GROUP BY LOWER(query)"
        " ORDER BY n DESC LIMIT 12",
        (f"-{days} days",),
    ).fetchall()
    zero = conn.execute(
        "SELECT query, COUNT(*) n FROM search_event"
        " WHERE n_results = 0 AND day >= date('now', ?)"
        " GROUP BY LOWER(query) ORDER BY n DESC LIMIT 10",
        (f"-{days} days",),
    ).fetchall()
    return {
        "by_day": [dict(r) for r in rows],
        "total_searches": sum(r["searches"] for r in rows),
        "popular_queries": [dict(r) for r in popular],
        # Searches returning nothing are the most useful signal for what to
        # index next.
        "unanswered_queries": [dict(r) for r in zero],
    }
