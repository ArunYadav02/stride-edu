"""Tests for Stride's pipeline and privacy boundaries.

Two categories matter most here. The parsing tests guard the corpus: a
regression that silently mis-splits questions corrupts everything downstream
and is invisible in the UI. The visibility tests guard users: a personal upload
leaking into public search is the one bug that would genuinely harm someone.
"""
from pathlib import Path
import sys, tempfile
sys.path.insert(0, str(Path(__file__).parent.parent))

import fitz
import pytest

from backend.db import (connect, init_db, insert_paper, insert_question,
                        upsert_institution, upsert_module)
from backend.ingest import extract_marks, looks_scanned, parse_paper
from backend.search import SearchFilters, search_questions, topic_profile, _fts_query
from backend.topics import tag_text


@pytest.fixture
def db(tmp_path):
    p = tmp_path / "t.db"
    init_db(p)
    return p


RUBRIC = [
    ("Time allowed: two hours. Answer ALL questions in the booklet provided.", 10, False, 0),
    ("Calculators are permitted. Write your candidate number on every sheet.", 10, False, 0),
]


def make_pdf(path, lines, rubric=True):
    """Build a test PDF.

    Rubric lines are included by default because a real paper always carries
    them, and their absence would push these fixtures below the scan-detection
    floor — the tests would then be measuring the fixture rather than the parser.
    """
    doc = fitz.open(); page = doc.new_page(); y = 60
    for t, size, bold, indent in (lines if not rubric else lines[:1] + RUBRIC + lines[1:]):
        page.insert_text((60+indent, y), t, fontsize=size,
                         fontname="hebo" if bold else "helv")
        y += size + 6
    doc.save(path); doc.close()


# ------------------------------------------------------------------ parsing
def test_marks_extracted_from_every_notation():
    assert extract_marks("Do the thing. [10 marks]") == 10
    assert extract_marks("Do the thing. (8 marks)") == 8
    assert extract_marks("Do the thing. [12]") == 12
    assert extract_marks("worth 15 marks total") == 15


def test_marks_ignores_years_and_page_numbers():
    """A four-digit year must never be read as a mark total."""
    assert extract_marks("the 2019 paper") is None
    assert extract_marks("see figure 3 on page 400") != 400


def test_three_numbering_styles_all_parse(tmp_path):
    """Papers in the wild use incompatible numbering; all must work."""
    styles = {
        "A": [("Question 1", 11, True, 0), ("Explain hashing. [10 marks]", 10, False, 20),
              ("Question 2", 11, True, 0), ("Explain sorting. [10 marks]", 10, False, 20)],
        "B": [("1.", 11, True, 0), ("Explain hashing. (10)", 10, False, 20),
              ("2.", 11, True, 0), ("Explain sorting. (10)", 10, False, 20)],
        "C": [("Q1", 11, True, 0), ("Explain hashing. [10 marks]", 10, False, 20),
              ("Q2", 11, True, 0), ("Explain sorting. [10 marks]", 10, False, 20)],
    }
    for name, lines in styles.items():
        f = tmp_path / f"{name}.pdf"
        make_pdf(f, [("CS101 Exam", 14, True, 0), ("Answer all.", 10, False, 0)] + lines)
        r = parse_paper(f)
        assert len(r.questions) == 2, f"style {name} produced {len(r.questions)} questions"
        assert r.confidence > 0.4, f"style {name} confidence {r.confidence}"


def test_question_text_excludes_its_own_number(tmp_path):
    f = tmp_path / "p.pdf"
    make_pdf(f, [("CS101", 14, True, 0), ("Answer all questions.", 10, False, 0),
                 ("Question 1", 11, True, 0),
                 ("Define a hash table. [10 marks]", 10, False, 20)])
    r = parse_paper(f)
    assert not r.questions[0].text.startswith("Question 1")
    assert "hash table" in r.questions[0].text


def test_parent_marks_not_inherited_from_children(tmp_path):
    """A parent must not claim its first sub-part's marks; totals would double."""
    f = tmp_path / "p.pdf"
    make_pdf(f, [("CS101", 14, True, 0), ("Answer all.", 10, False, 0),
                 ("Question 1", 11, True, 0),
                 ("(a) Define a hash table. [6 marks]", 10, False, 20),
                 ("(b) Explain chaining. [4 marks]", 10, False, 20)])
    r = parse_paper(f)
    q = r.questions[0]
    assert q.marks is None
    assert [c.marks for c in q.children] == [6, 4]
    assert r.total_marks == 10


def test_scanned_pdf_rejected_with_reason(tmp_path):
    f = tmp_path / "scan.pdf"
    make_pdf(f, [("Page 1", 10, False, 0)], rubric=False)
    r = parse_paper(f)
    assert r.questions == []
    assert "scanned" in r.note.lower()


def test_short_real_paper_not_mistaken_for_scan(tmp_path):
    """Regression: a flat character threshold rejected valid short papers."""
    f = tmp_path / "short.pdf"
    make_pdf(f, [("CS101 Exam", 14, True, 0), ("Answer both questions.", 10, False, 0),
                 ("Question 1", 11, True, 0), ("Define a hash table. [10 marks]", 10, False, 20),
                 ("Question 2", 11, True, 0), ("Define a red-black tree. [10 marks]", 10, False, 20)])
    r = parse_paper(f)
    assert len(r.questions) == 2


# ------------------------------------------------------------------ tagging
def test_phrases_beat_bare_words():
    tags = tag_text("Describe the dynamic programming solution to the knapsack problem "
                    "and state the recurrence relation clearly.")
    assert tags and tags[0].topic == "Dynamic programming"


def test_negative_evidence_prevents_wrong_topic():
    """'decision tree' must not fire the tree data-structure topic."""
    tags = tag_text("Explain how a decision tree classifier splits on features "
                    "and how information gain is computed at each node.")
    names = [t.topic for t in tags]
    assert "Trees and balanced structures" not in names


def test_very_short_text_is_not_tagged():
    assert tag_text("Explain.") == []


# ------------------------------------------------------------------- search
def test_fts_query_is_or_not_and():
    """Regression: AND made multi-word searches return nothing."""
    q = _fts_query("dynamic programming knapsack")
    assert " OR " in q and " AND " not in q


def test_fts_query_survives_hostile_input():
    for bad in ['" OR 1=1 --', "NEAR(a b)", "*", "'; DROP TABLE question; --", ""]:
        _fts_query(bad)  # must not raise


def _seed(dbp):
    with connect(dbp) as conn:
        i = upsert_institution(conn, "Test Uni", "Test")
        m = upsert_module(conn, i, "CS101", "Algorithms")
        p = insert_paper(conn, m, 2024, source_url="https://example.ac.uk/x")
        insert_question(conn, p, "1", "Explain red-black tree rotations in detail.", 0, marks=10)
        return i, m, p


def test_search_finds_and_times(db):
    _seed(db)
    with connect(db) as conn:
        r = search_questions(conn, "red-black")
        assert r["total"] == 1
        assert r["duration_ms"] < 200


def test_personal_papers_hidden_from_anonymous_search(db):
    """The privacy boundary. A leak here exposes someone's uploaded material."""
    with connect(db) as conn:
        i = upsert_institution(conn, "Test Uni", "Test")
        m = upsert_module(conn, i, "CS101", "Algorithms")
        conn.execute("INSERT INTO user (email, password_hash, created_at) VALUES (?,?,?)",
                     ("u@x.ac.uk", "x", 0))
        uid = conn.execute("SELECT id FROM user").fetchone()["id"]
        p = insert_paper(conn, m, 2024, visibility="personal", owner_user_id=uid)
        insert_question(conn, p, "1", "A private question about quicksort.", 0)

    with connect(db) as conn:
        anon = search_questions(conn, "quicksort")
        assert anon["total"] == 0, "personal upload leaked to anonymous search"

        other = search_questions(conn, "quicksort",
                                 SearchFilters(include_personal_for_user=uid + 99))
        assert other["total"] == 0, "personal upload leaked to a different user"

        owner = search_questions(conn, "quicksort",
                                 SearchFilters(include_personal_for_user=uid))
        assert owner["total"] == 1, "owner cannot see their own upload"


def test_topic_profile_denominator_is_same_module(db):
    """'3 of 37 papers' across unrelated modules is true but useless."""
    from backend.topics import tag_and_store
    with connect(db) as conn:
        i = upsert_institution(conn, "Test Uni", "Test")
        algo = upsert_module(conn, i, "CS101", "Algorithms")
        os_m = upsert_module(conn, i, "CS202", "Operating Systems")
        for y in (2023, 2024):
            p = insert_paper(conn, algo, y)
            qid = insert_question(conn, p, "1",
                "Describe the dynamic programming solution to the knapsack problem.", 0, marks=10)
            tag_and_store(conn, qid, "Describe the dynamic programming solution to the knapsack problem.")
        for y in (2023, 2024, 2025):
            insert_paper(conn, os_m, y)

    with connect(db) as conn:
        prof = topic_profile(conn, "Dynamic programming")
        assert prof["total_papers"] == 2, \
            f"denominator {prof['total_papers']} should exclude unrelated modules"
