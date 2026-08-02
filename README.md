# Stride Edu

**Search past exam questions by topic, year and marks — and see what actually comes up.**

Past papers live in departmental portals with search that can't answer the
question students actually have. Not "find the 2023 paper", but *"show me every
question on red-black trees"*, and *"is dynamic programming worth revising?"*

Stride answers both.

> **Neural networks appeared in 5 of the last 7 papers, most often as question 3,
> typically worth 12 marks.**

---

## The line that matters most

**Stride indexes. It does not rehost.**

Exam papers are university copyright. A tool whose value comes from
redistributing other institutions' PDFs is one takedown notice from
disappearing — and it puts the person who built it in an awkward conversation
with their department.

So Stride stores extracted question text, topics, marks and provenance, and
links back to the official source. There is no `pdf_blob` column and no file
store for source papers: the constraint lives in the schema, not in a policy
document, because policy documents don't survive contact with a late-night
feature idea.

Every institution is listed with its permission status — agreed, requested, or
not yet contacted — and that status is visible in the UI next to the results.

Papers you upload yourself are parsed, indexed, and **kept private to your
account**. The PDF is discarded after parsing.

---

## What it does

**Search** — full-text across every indexed question, with filters for module,
year range, minimum marks and topic. Sub-15ms on the seeded corpus. No account
needed; requiring a login to search would cost most of the users.

**Topic profiles** — the headline feature. For any topic: how often it appears,
whether it's rising or falling, where in the paper it usually sits, and what
it's typically worth. This falls straight out of having questions tagged,
positioned and marked, and it's what a revising student actually wants.

**Coverage** — what's indexed, from where, and with whose permission.

**Saved questions** — the one feature that genuinely needs an account.

**Personal uploads** — index papers you already have, privately.

---

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python seed.py                                    # builds a demo corpus
uvicorn backend.api:app --reload                  # http://localhost:8000
```

Run the tests:

```bash
python -m pytest tests/ -q                        # 15 tests
```

**The seeded corpus is synthetic.** The papers are written to exercise the
pipeline, and the institution names are fictional. This keeps the demo shareable
without redistributing anyone's material — the same constraint the production
system enforces.

---

## The hard part: PDF segmentation

Exam papers have no standard format. Some number questions `1.`, some
`Question 1`, some `Q1`. Sub-parts are `(a)`, `a)`, `(i)`, or indentation alone.
Marks appear as `[10 marks]`, `(10)`, `10 marks`, or in a right-hand column that
PyMuPDF returns interleaved with body text.

The approach is layered, and each layer reports its confidence:

1. Extract text with layout preserved — page, vertical position, font size.
2. Score each candidate numbering pattern by how consistently it appears: a
   paper using `Question N` throughout scores high; one where `1.` also matches
   list items inside a question scores low. **Sequence quality does the real
   work** — a pattern matching many lines in jumbled order is a list, not a
   numbering scheme.
3. Build the question tree from the winning pattern plus indentation.
4. Pull marks from whichever notation the paper uses, with sanity bounds so a
   four-digit year is never read as a mark total.

**Confidence is surfaced, not hidden.** A paper that segments badly is stored
with low `parse_quality` and flagged in the UI. Silently storing garbage
questions would make the whole corpus untrustworthy, and a search tool people
can't trust is worse than no search tool.

Scanned papers with no text layer are detected and rejected with a clear reason.
OCR isn't implemented; pretending otherwise would be worse than saying so.

### Calibrating scan detection took three attempts

Worth recording, because it's the kind of thing that looks trivial and isn't:

- A per-page character average rejected short but legitimate papers.
- A flat character floor landed *inside* the range real papers occupy (measured:
  450–800 characters for a one-page paper), so it kept dropping valid documents.
- The reliable signal is that a scan yields almost no **lines** — the text layer
  is absent, not sparse. Requiring both a very low line count and very little
  text separates a page image from a genuinely short paper, with neither
  threshold near the real distribution.

---

## Topic tagging

Deliberately not an LLM. It must run on every question at ingest with no API
budget, it has to be explainable when a student asks why a tag is wrong, and on
a closed domain like a CS syllabus a curated ontology is competitive with
zero-shot classification at a fraction of the complexity.

What makes it more than keyword matching:

- **Phrases beat words.** "dynamic programming" is a strong signal;
  "programming" alone is nearly useless.
- **Negative evidence.** "decision tree" must not fire the tree data-structure
  topic. Topics list terms that argue against them.
- **Length normalisation.** Without it, long questions match everything and
  every topic appears to be trending upward.
- **Confidence is kept and shown**, and users can correct any tag.

Corrections are stored as `source='corrected'` and are the highest-value data
Stride collects: labels produced by people who know the material.
`training_export()` exists so that data can actually be used later rather than
sitting in a table forever.

**Measured accuracy is published in the UI** rather than kept internal. A tool
that tells you it's right 84% of the time is more trustworthy than one implying
perfection, and it sets expectations before someone finds a wrong tag.

---

## Architecture

```
backend/
  db.py        Schema and access. The no-rehosting rule is enforced here.
  ingest.py    PDF → question tree, with confidence scoring
  topics.py    Ontology, tagger, correction handling
  search.py    FTS5 search + the topic-frequency analytics
  api.py       FastAPI: auth, search, topics, uploads, feedback
web/
  index.html   Single-page frontend, no build step
tests/         15 tests
seed.py        Builds a synthetic demo corpus
```

**SQLite with FTS5, not Postgres.** One file, no service, runs on free hosting,
and full-text search over a few hundred thousand questions is comfortably
sub-100ms. The search layer is swappable if this outgrows one file.

**No build step.** One HTML file, no npm, no bundler. It deploys by copying.

---

## Security and privacy

- Passwords hashed with bcrypt, truncated to 72 bytes explicitly — passlib's
  backend *raises* on longer input, so a long passphrase used to produce a 500
  error. A passphrase is exactly what a security-conscious user types.
- JWT secret read from `STRIDE_SECRET`. **The app refuses to start in production
  without one** rather than falling back to a default. A hardcoded fallback
  secret is the most common way a project like this gets compromised.
- Sign-in returns the same error for unknown email and wrong password, so the
  endpoint can't be used to enumerate accounts.
- **Analytics record no IP, no user agent, and no cross-session identifier** —
  only the calendar day, the query, and result count. Enough to say "N searches
  this week", nothing that could re-identify a student.
- Personal-paper visibility is enforced in every query, not by convention, and
  there's a test asserting a personal upload is invisible to anonymous users and
  to other signed-in users.

---

## Deploying

Set the secret and origin, then run anywhere that serves Python:

```bash
export STRIDE_ENV=production
export STRIDE_SECRET=$(python -c 'import secrets; print(secrets.token_hex(32))')
export STRIDE_ORIGIN=https://your-domain
uvicorn backend.api:app --host 0.0.0.0 --port $PORT
```

SQLite lives on disk, so the host needs a persistent volume — free tiers with
ephemeral filesystems will lose the database on restart.

---

## Limitations

**The seeded corpus is synthetic.** No real papers ship with this repo, by
design. Real usage requires either departmental permission or users indexing
their own papers.

**No OCR.** Scanned papers are detected and rejected rather than badly parsed.

**Segmentation is heuristic.** It handles the three common numbering schemes and
reports low confidence on anything else. It will get some papers wrong; the
confidence score is how you find out which.

**The tagger's ontology is hand-built and CS-focused.** It covers algorithms,
data structures, databases, ML, systems, theory and security. Another subject
needs another ontology — the structure supports it, the content doesn't exist.

**Trend detection needs at least four years** of papers before it will call a
direction, and says so rather than drawing a confident arrow through noise.

**Single-node.** One SQLite file, one process. Fine for a department; not for a
national service.

---

## Roadmap

- [ ] OCR for scanned papers (Tesseract, with confidence gating)
- [ ] Semantic search alongside FTS, for "questions like this one"
- [ ] Use collected corrections to train a tagger and compare against the
      rule-based one
- [ ] Institution-scoped access, so a department can index privately
- [ ] Export a revision set as PDF

---

## Licence

MIT for the code. Indexed content belongs to the institutions that wrote it.
