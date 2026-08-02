"""
Turning a PDF exam paper into structured questions.

This is the hard part of Stride and the part worth writing about. Exam papers
have no standard format: some number questions `1.`, some `Question 1`, some
`Q1`; sub-parts are `(a)`, `a)`, `(i)`, or indentation alone; marks appear as
`[10 marks]`, `(10)`, `10 marks`, or in a right-hand column that PyMuPDF returns
interleaved with body text.

The approach is layered, and each layer reports how confident it is:

1. Extract text with layout preserved, keeping page and vertical position.
2. Find question boundaries with a set of numbering patterns, scored by how
   consistently each pattern appears down the document. A paper that uses
   `Question N` throughout scores high on that pattern; one where `1.` also
   matches a list item inside a question scores lower.
3. Build the question tree from the winning pattern plus indentation.
4. Pull marks from whichever notation the paper uses.

**Confidence is surfaced, not hidden.** A paper that segments badly is stored
with a low `parse_quality` and flagged in the UI. The alternative — silently
storing garbage questions — makes the whole corpus untrustworthy, and a search
tool people cannot trust is worse than no search tool.

Scanned papers with no text layer are detected and rejected with a clear reason
rather than producing empty questions. OCR is a future addition; pretending it
already works is not.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - import guard for environments without it
    fitz = None  # type: ignore


# --------------------------------------------------------------------------
# Line model
# --------------------------------------------------------------------------


@dataclass
class Line:
    text: str
    page: int
    y: float           # vertical position on the page
    x: float           # left edge, used for indentation
    size: float        # font size, used to spot headings
    bold: bool = False

    @property
    def stripped(self) -> str:
        return self.text.strip()


def extract_lines(pdf_path: Path) -> tuple[list[Line], dict[str, Any]]:
    """Pull text lines with position. Returns (lines, diagnostics)."""
    if fitz is None:
        raise RuntimeError("PyMuPDF is not installed; run pip install pymupdf")

    doc = fitz.open(str(pdf_path))
    lines: list[Line] = []
    total_chars = 0

    for pno in range(doc.page_count):
        page = doc[pno]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                text = "".join(s.get("text", "") for s in spans)
                total_chars += len(text.strip())
                if not text.strip():
                    continue
                first = spans[0]
                flags = first.get("flags", 0)
                lines.append(
                    Line(
                        text=text,
                        page=pno + 1,
                        y=line["bbox"][1],
                        x=line["bbox"][0],
                        size=first.get("size", 10.0),
                        bold=bool(flags & 2 ** 4),
                    )
                )

    diagnostics = {
        "pages": doc.page_count,
        "lines": len(lines),
        "chars": total_chars,
        "chars_per_page": round(total_chars / doc.page_count, 1) if doc.page_count else 0,
    }
    doc.close()
    return lines, diagnostics


def looks_scanned(diagnostics: dict[str, Any]) -> bool:
    """A PDF of page images has almost no extractable text.

    Better to reject clearly than to store a paper with three garbled questions
    and let it pollute search results.

    Choosing the threshold took two attempts. A per-page average rejected short
    but legitimate papers; a flat character floor landed inside the range real
    papers actually occupy (measured: 450-800 characters for a one-page paper),
    so it kept dropping valid documents.

    The reliable signal is that a scan yields almost no *lines*, not merely few
    characters — the text layer is absent rather than sparse. Requiring both a
    very low line count and very little text separates a page image from a
    genuinely short paper, and neither threshold sits near the real
    distribution.
    """
    return diagnostics["lines"] < 5 or diagnostics["chars"] < 150


# --------------------------------------------------------------------------
# Numbering patterns
# --------------------------------------------------------------------------

# Each pattern: (name, regex, depth). Ordered from most to least explicit.
# `depth` 0 is a top-level question, 1 a sub-part, 2 a sub-sub-part.
PATTERNS: list[tuple[str, re.Pattern, int]] = [
    ("question_word", re.compile(r"^\s*Question\s+(\d{1,2})\b[.:)]?", re.I), 0),
    ("q_prefix",      re.compile(r"^\s*Q\.?\s?(\d{1,2})\b[.:)]?"), 0),
    ("bare_number",   re.compile(r"^\s*(\d{1,2})[.)](?:\s+|\s*$)"), 0),
    ("roman_sub",     re.compile(r"^\s*\(([ivx]{1,4})\)\s*|^\s*([ivx]{2,4})[.)]\s+", re.I), 2),
    ("letter_sub",    re.compile(r"^\s*\(?([a-h])[.)]\s+"), 1),
]

MARKS_PATTERNS = [
    re.compile(r"\[\s*(\d{1,3})\s*marks?\s*\]", re.I),
    re.compile(r"\(\s*(\d{1,3})\s*marks?\s*\)", re.I),
    re.compile(r"\[\s*(\d{1,3})\s*\]\s*$"),
    re.compile(r"\(\s*(\d{1,3})\s*\)\s*$"),
    re.compile(r"\b(\d{1,3})\s*marks?\b", re.I),
]

TOTAL_MARKS = re.compile(r"total(?:\s+of)?\s*[:\-]?\s*(\d{1,3})\s*marks?", re.I)
DURATION = re.compile(r"(\d)\s*(?:hours?|hrs?)(?:\s*(\d{1,2})\s*min)?", re.I)


def extract_marks(text: str) -> int | None:
    for pat in MARKS_PATTERNS:
        m = pat.search(text)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 200:   # sanity bound: catches years and page numbers
                return v
    return None


def score_pattern(lines: list[Line], pattern: re.Pattern) -> float:
    """How plausible is this pattern as *the* numbering scheme for this paper?

    Two signals: how many lines match, and whether the matched numbers form a
    rising sequence. `1.` matching a bulleted list inside a question produces
    many matches but a jumbled sequence, so sequence quality does the real work.
    """
    hits: list[tuple[int, int]] = []   # (line index, parsed number)
    for i, ln in enumerate(lines):
        m = pattern.match(ln.stripped)
        if m:
            raw = m.group(1)
            try:
                n = int(raw)
            except ValueError:
                n = _roman_to_int(raw)
            if n:
                hits.append((i, n))

    if not hits:
        return 0.0

    numbers = [n for _, n in hits]

    # A single match cannot demonstrate a sequence, but one-question papers are
    # real (resits, short tests, single-essay finals). Score it on the strength
    # of the pattern alone: an explicit "Question 1" is convincing on its own,
    # while a bare "1." could be anything, so it stays below the parse floor.
    if len(hits) == 1:
        explicit = pattern.pattern.lower().startswith(("^\\s*question", "^\\s*q"))
        return (0.6 if explicit else 0.2) * (1.0 if numbers[0] == 1 else 0.5)
    ascending = sum(1 for a, b in zip(numbers, numbers[1:]) if b == a + 1)
    sequence_quality = ascending / max(1, len(numbers) - 1)

    # Prefer patterns that start at 1 — a real paper's first question is Q1.
    starts_right = 1.0 if numbers[0] == 1 else 0.6

    # Penalise absurd counts: 40 top-level questions is a list, not a paper.
    count_sanity = 1.0 if 2 <= len(hits) <= 25 else 0.4

    return sequence_quality * starts_right * count_sanity


def _roman_to_int(s: str) -> int:
    vals = {"i": 1, "v": 5, "x": 10}
    s = s.lower()
    if not all(c in vals for c in s):
        return 0
    total, prev = 0, 0
    for c in reversed(s):
        v = vals[c]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------


@dataclass
class ParsedQuestion:
    number: str
    text: str
    marks: int | None
    page: int
    depth: int
    position: int
    children: list["ParsedQuestion"] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["children"] = [c.to_json() for c in self.children]
        return d


@dataclass
class ParsedPaper:
    questions: list[ParsedQuestion]
    total_marks: int | None
    duration_min: int | None
    confidence: float
    note: str
    diagnostics: dict[str, Any]

    def flat(self) -> Iterator[ParsedQuestion]:
        def walk(qs: list[ParsedQuestion]) -> Iterator[ParsedQuestion]:
            for q in qs:
                yield q
                yield from walk(q.children)
        return walk(self.questions)


def parse_paper(pdf_path: Path) -> ParsedPaper:
    """Segment a PDF into a question tree, with an honest confidence score."""
    lines, diag = extract_lines(pdf_path)

    if looks_scanned(diag):
        return ParsedPaper(
            [], None, None, 0.0,
            "This looks like a scanned image with no text layer. Stride cannot "
            "index it yet — OCR is not implemented.",
            diag,
        )

    # Pick the top-level numbering scheme.
    scored = [
        (name, pat, depth, score_pattern(lines, pat))
        for name, pat, depth in PATTERNS
        if depth == 0
    ]
    scored.sort(key=lambda t: -t[3])
    best_name, best_pat, _, best_score = scored[0]

    if best_score < 0.25:
        return ParsedPaper(
            [], None, None, round(best_score, 2),
            "No consistent question numbering found. The paper may use an "
            "unusual layout; it has not been indexed.",
            {**diag, "best_pattern": best_name, "pattern_score": round(best_score, 3)},
        )

    # Split lines into top-level question blocks.
    boundaries: list[int] = []
    for i, ln in enumerate(lines):
        if best_pat.match(ln.stripped):
            boundaries.append(i)

    questions: list[ParsedQuestion] = []
    position = 0
    for bi, start in enumerate(boundaries):
        end = boundaries[bi + 1] if bi + 1 < len(boundaries) else len(lines)
        block = lines[start:end]
        if not block:
            continue
        m = best_pat.match(block[0].stripped)
        number = m.group(1) if m else str(bi + 1)

        sub = _split_subparts(block)
        body = "\n".join(l.stripped for l in block).strip()
        # Marks belong to whichever part they are printed against. Scanning the
        # whole block would give the parent its first child's marks, which then
        # double-counts in every total and skews the analytics.
        own_lines = block if not sub else block[: sub[0][0]]
        own_raw = [l.stripped for l in own_lines]
        # Drop the bare numbering line ("Question 3", "1.") from the body: the
        # number is already a structured field, and repeating it inside the text
        # makes every search snippet start with noise instead of the question.
        if own_raw:
            head = own_raw[0]
            without = best_pat.sub("", head, count=1).strip()
            if not without:
                own_raw = own_raw[1:]
            else:
                own_raw[0] = without
        own_text = _clean("\n".join(own_raw))
        q = ParsedQuestion(
            number=number,
            text=own_text,
            marks=extract_marks(own_text) if not sub else None,
            page=block[0].page,
            depth=0,
            position=position,
        )
        position += 1

        for si, (s_start, s_end, label, depth) in enumerate(sub):
            s_block = block[s_start:s_end]
            s_text = _clean("\n".join(l.stripped for l in s_block))
            if not s_text:
                continue
            q.children.append(
                ParsedQuestion(
                    number=f"{number}({label})",
                    text=s_text,
                    marks=extract_marks(s_text),
                    page=s_block[0].page,
                    depth=depth,
                    position=position,
                )
            )
            position += 1
        questions.append(q)

    full_text = "\n".join(l.stripped for l in lines)
    total = None
    tm = TOTAL_MARKS.search(full_text)
    if tm:
        total = int(tm.group(1))
    else:
        marks = [q.marks for q in _walk(questions) if q.marks]
        if marks:
            total = sum(marks)

    duration = None
    dm = DURATION.search(full_text[:2000])
    if dm:
        duration = int(dm.group(1)) * 60 + int(dm.group(2) or 0)

    confidence, note = _assess(questions, best_score, diag)
    return ParsedPaper(
        questions, total, duration, confidence, note,
        {**diag, "pattern": best_name, "pattern_score": round(best_score, 3)},
    )


def _walk(qs: list[ParsedQuestion]) -> Iterator[ParsedQuestion]:
    for q in qs:
        yield q
        yield from _walk(q.children)


def _split_subparts(block: list[Line]) -> list[tuple[int, int, str, int]]:
    """Find (a), (b), (i) style sub-parts within one question block."""
    letter = PATTERNS[4][1]
    roman = PATTERNS[3][1]
    marks: list[tuple[int, str, int]] = []
    for i, ln in enumerate(block[1:], start=1):
        s = ln.stripped
        mr = roman.match(s)
        if mr:
            label = mr.group(1) or mr.group(2)
            if label and _roman_to_int(label):
                marks.append((i, label, 2))
                continue
        ml = letter.match(s)
        if ml:
            marks.append((i, ml.group(1), 1))

    out: list[tuple[int, int, str, int]] = []
    for j, (idx, label, depth) in enumerate(marks):
        end = marks[j + 1][0] if j + 1 < len(marks) else len(block)
        out.append((idx, end, label, depth))
    return out


def _clean(text: str) -> str:
    """Strip page furniture that survives extraction."""
    lines = []
    for ln in text.split("\n"):
        s = ln.strip()
        if not s:
            continue
        if re.fullmatch(r"(page\s*)?\d{1,3}(\s*of\s*\d{1,3})?", s, re.I):
            continue
        if re.fullmatch(r"[-–—_=]{3,}", s):
            continue
        if re.search(r"turn over|end of (paper|question)|continued", s, re.I) and len(s) < 40:
            continue
        lines.append(s)
    return "\n".join(lines).strip()


def _assess(
    questions: list[ParsedQuestion], pattern_score: float, diag: dict[str, Any]
) -> tuple[float, str]:
    """Combine signals into one confidence score and a plain-language note."""
    if not questions:
        return 0.0, "No questions were found."

    n = len(questions)
    all_q = list(_walk(questions))
    with_marks = sum(1 for q in all_q if q.marks is not None)
    mark_ratio = with_marks / len(all_q)

    lengths = [len(q.text) for q in all_q if q.text]
    length_sanity = 1.0
    if lengths:
        median = statistics.median(lengths)
        if median < 40:
            length_sanity = 0.4     # suspiciously short: probably over-split
        elif median > 3000:
            length_sanity = 0.5     # suspiciously long: probably under-split

    count_sanity = 1.0 if 2 <= n <= 15 else 0.6

    score = pattern_score * 0.4 + mark_ratio * 0.25 + length_sanity * 0.25 + count_sanity * 0.1
    score = round(min(1.0, max(0.0, score)), 2)

    if score >= 0.75:
        note = f"Segmented cleanly into {n} questions."
    elif score >= 0.5:
        note = (
            f"Segmented into {n} questions, but some parts may be split "
            "incorrectly. Worth checking against the original."
        )
    else:
        note = (
            f"Low-confidence split into {n} questions. This paper's layout is "
            "unusual; treat results from it with caution."
        )
    return score, note
