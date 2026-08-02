"""
Topic tagging.

Deliberately not an LLM. Three reasons: it must run on every question at ingest
time without an API budget, it has to be explainable when a student says "why is
this tagged wrong", and — the honest one — a keyword-and-phrase tagger with a
curated ontology is competitive with a zero-shot model on a closed domain like a
CS syllabus, at a fraction of the complexity.

What makes it good rather than a naive keyword search:

- **Phrases beat words.** "dynamic programming" is a strong signal; "programming"
  alone is nearly useless. Multi-word matches are weighted far higher.
- **Negative evidence.** "linear regression" should not fire "linear algebra".
  Topics can list terms that argue *against* them.
- **Normalisation by question length.** Long questions otherwise match
  everything, and the analytics then show every topic trending upward.
- **Confidence is kept and shown.** A tag at 0.3 is displayed differently from
  one at 0.9, and users can correct either.

Corrections are stored with `source='corrected'` and are the highest-value data
Stride collects: a labelled set produced by the people who know the material.
The `training_export` function exists so that data can actually be used later
rather than sitting in a table forever.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Ontology
# --------------------------------------------------------------------------
# Curated rather than learned. Each topic: strong phrases, weaker single terms,
# and terms that argue against it. Grouped by parent for the UI's topic tree.

ONTOLOGY: dict[str, dict[str, Any]] = {
    # ---- algorithms & data structures
    "Sorting": {
        "parent": "Algorithms",
        "phrases": ["merge sort", "quick sort", "quicksort", "heap sort", "insertion sort",
                    "bubble sort", "radix sort", "counting sort", "comparison sort"],
        "terms": ["sorting", "sorted", "pivot", "partition"],
        "against": [],
    },
    "Graph algorithms": {
        "parent": "Algorithms",
        "phrases": ["shortest path", "minimum spanning tree", "breadth-first search",
                    "depth-first search", "topological sort", "strongly connected",
                    "bellman-ford", "floyd-warshall", "max flow", "min cut"],
        "terms": ["dijkstra", "kruskal", "prim", "bfs", "dfs", "graph", "vertex",
                  "vertices", "edge", "adjacency"],
        "against": [],
    },
    "Dynamic programming": {
        "parent": "Algorithms",
        "phrases": ["dynamic programming", "optimal substructure", "overlapping subproblems",
                    "memoisation", "memoization", "knapsack problem", "edit distance",
                    "longest common subsequence"],
        "terms": ["knapsack", "recurrence", "subproblem", "tabulation"],
        "against": [],
    },
    "Complexity analysis": {
        "parent": "Algorithms",
        "phrases": ["time complexity", "space complexity", "big-o", "worst case",
                    "average case", "asymptotic analysis", "amortised analysis",
                    "master theorem", "np-complete", "np-hard", "polynomial time"],
        "terms": ["complexity", "asymptotic", "amortised", "amortized", "reduction"],
        "against": [],
    },
    "Trees and balanced structures": {
        "parent": "Data structures",
        "phrases": ["binary search tree", "red-black tree", "avl tree", "b-tree",
                    "balanced tree", "tree rotation", "trie", "heap property"],
        "terms": ["tree", "rotation", "subtree", "leaf", "traversal", "heap"],
        "against": ["decision tree", "spanning tree"],
    },
    "Hashing": {
        "parent": "Data structures",
        "phrases": ["hash table", "hash function", "open addressing", "separate chaining",
                    "load factor", "collision resolution", "bloom filter"],
        "terms": ["hashing", "hashed", "collision", "bucket"],
        "against": ["cryptographic hash"],
    },
    # ---- databases
    "Relational design": {
        "parent": "Databases",
        "phrases": ["functional dependency", "normal form", "third normal form",
                    "boyce-codd", "normalisation", "normalization", "candidate key",
                    "foreign key", "entity relationship"],
        "terms": ["2nf", "3nf", "bcnf", "normalise", "normalize", "schema", "relation"],
        "against": [],
    },
    "Transactions and concurrency": {
        "parent": "Databases",
        "phrases": ["two-phase locking", "acid properties", "isolation level",
                    "serialisable schedule", "deadlock detection", "write-ahead log",
                    "concurrency control", "dirty read", "phantom read"],
        "terms": ["transaction", "acid", "locking", "deadlock", "rollback", "commit"],
        "against": [],
    },
    "Query processing": {
        "parent": "Databases",
        "phrases": ["query optimisation", "query optimization", "query plan",
                    "join algorithm", "index scan", "relational algebra", "sql query"],
        "terms": ["sql", "select", "join", "index", "cardinality"],
        "against": [],
    },
    # ---- machine learning
    "Supervised learning": {
        "parent": "Machine learning",
        "phrases": ["linear regression", "logistic regression", "support vector machine",
                    "decision tree", "random forest", "gradient boosting", "naive bayes",
                    "cross-entropy loss", "training error"],
        "terms": ["classifier", "classification", "regression", "supervised", "label"],
        "against": [],
    },
    "Neural networks": {
        "parent": "Machine learning",
        "phrases": ["neural network", "backpropagation", "activation function",
                    "vanishing gradient", "convolutional", "hidden layer",
                    "gradient descent", "learning rate"],
        "terms": ["relu", "sigmoid", "perceptron", "epoch", "neuron"],
        "against": [],
    },
    "Model evaluation": {
        "parent": "Machine learning",
        "phrases": ["bias-variance", "cross validation", "overfitting", "underfitting",
                    "confusion matrix", "precision and recall", "validation error",
                    "regularisation", "regularization"],
        "terms": ["overfit", "underfit", "f1", "roc", "auc", "generalisation"],
        "against": [],
    },
    # ---- systems
    "Operating systems": {
        "parent": "Systems",
        "phrases": ["virtual memory", "page fault", "context switch", "process scheduling",
                    "critical section", "semaphore", "mutual exclusion", "page replacement"],
        "terms": ["scheduler", "thread", "deadlock", "paging", "kernel", "syscall"],
        "against": [],
    },
    "Networking": {
        "parent": "Systems",
        "phrases": ["tcp handshake", "congestion control", "routing protocol",
                    "packet switching", "sliding window", "osi model", "subnet mask"],
        "terms": ["tcp", "udp", "ip", "router", "latency", "bandwidth", "protocol"],
        "against": [],
    },
    "Concurrency": {
        "parent": "Systems",
        "phrases": ["race condition", "mutual exclusion", "lock-free", "atomic operation",
                    "producer consumer", "reader writer"],
        "terms": ["concurrent", "parallel", "synchronisation", "mutex", "barrier"],
        "against": [],
    },
    # ---- theory
    "Formal languages": {
        "parent": "Theory",
        "phrases": ["finite automaton", "regular expression", "context-free grammar",
                    "turing machine", "pumping lemma", "pushdown automaton",
                    "chomsky hierarchy"],
        "terms": ["automata", "automaton", "nfa", "dfa", "grammar", "decidable"],
        "against": [],
    },
    "Logic and proof": {
        "parent": "Theory",
        "phrases": ["proof by induction", "propositional logic", "predicate logic",
                    "hoare logic", "loop invariant", "satisfiability"],
        "terms": ["induction", "invariant", "theorem", "prove", "quantifier"],
        "against": [],
    },
    # ---- security
    "Cryptography": {
        "parent": "Security",
        "phrases": ["public key", "symmetric encryption", "digital signature",
                    "hash function", "diffie-hellman", "block cipher", "key exchange"],
        "terms": ["rsa", "aes", "encryption", "decrypt", "cipher", "plaintext"],
        "against": [],
    },
    "Software security": {
        "parent": "Security",
        "phrases": ["buffer overflow", "sql injection", "cross-site scripting",
                    "access control", "privilege escalation", "threat model"],
        "terms": ["vulnerability", "exploit", "sandbox", "authentication"],
        "against": [],
    },
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


@dataclass
class Tag:
    topic: str
    confidence: float
    evidence: list[str]


def tag_text(text: str, threshold: float = 0.18, max_tags: int = 3) -> list[Tag]:
    """Assign topics to one question, with confidence and the matched evidence."""
    norm = _normalise(text)
    if len(norm) < 20:
        return []

    # Longer questions naturally contain more terms; without this every long
    # question matches everything and the topic-frequency chart becomes noise.
    length_factor = 1.0 / (1.0 + math.log1p(len(norm) / 400))

    scored: list[Tag] = []
    for topic, spec in ONTOLOGY.items():
        score = 0.0
        evidence: list[str] = []

        for phrase in spec["phrases"]:
            if phrase in norm:
                score += 1.0
                evidence.append(phrase)

        for term in spec["terms"]:
            if re.search(rf"\b{re.escape(term)}\b", norm):
                score += 0.3
                evidence.append(term)

        for neg in spec.get("against", []):
            if neg in norm:
                score -= 0.8

        if score <= 0:
            continue
        confidence = min(1.0, score * length_factor / 1.6)
        if confidence >= threshold:
            scored.append(Tag(topic, round(confidence, 3), evidence[:5]))

    scored.sort(key=lambda t: -t.confidence)
    return scored[:max_tags]


def tag_and_store(conn: sqlite3.Connection, question_id: int, text: str) -> list[Tag]:
    from .db import tag_question, upsert_topic

    tags = tag_text(text)
    for t in tags:
        parent = ONTOLOGY[t.topic]["parent"]
        tid = upsert_topic(conn, t.topic, parent=parent)
        tag_question(conn, question_id, tid, confidence=t.confidence, source="auto")
    return tags


def correct_tag(
    conn: sqlite3.Connection, question_id: int, topic_name: str, correct: bool
) -> None:
    """Record a human judgement on a tag.

    `correct=False` removes the tag and records the correction; `correct=True`
    promotes it to confirmed. Both are training signal — knowing a tag was
    *right* matters as much as knowing it was wrong.
    """
    from .db import upsert_topic

    tid = upsert_topic(conn, topic_name, parent=ONTOLOGY.get(topic_name, {}).get("parent"))
    if correct:
        conn.execute(
            "UPDATE question_topic SET source='confirmed', confidence=1.0"
            " WHERE question_id=? AND topic_id=?",
            (question_id, tid),
        )
    else:
        conn.execute(
            "UPDATE question_topic SET source='corrected', confidence=0.0"
            " WHERE question_id=? AND topic_id=?",
            (question_id, tid),
        )


def tagger_accuracy(conn: sqlite3.Connection) -> dict[str, Any]:
    """Measured accuracy from human corrections.

    Published in the UI rather than kept internal. A tool that tells you it is
    right 84% of the time is more trustworthy than one that implies perfection,
    and it sets the right expectation before someone finds a wrong tag.
    """
    rows = conn.execute(
        "SELECT source, COUNT(*) n FROM question_topic"
        " WHERE source IN ('confirmed','corrected') GROUP BY source"
    ).fetchall()
    counts = {r["source"]: r["n"] for r in rows}
    confirmed = counts.get("confirmed", 0)
    corrected = counts.get("corrected", 0)
    judged = confirmed + corrected
    return {
        "judged": judged,
        "confirmed": confirmed,
        "corrected": corrected,
        "accuracy": round(confirmed / judged, 3) if judged else None,
        "note": (
            "Based on tags students have reviewed."
            if judged >= 20
            else "Not enough reviews yet to report an accuracy figure."
        ),
    }


def training_export(conn: sqlite3.Connection, path: Path) -> int:
    """Dump human-judged tags as JSONL, so the corrections can actually be used."""
    rows = conn.execute(
        "SELECT q.id, q.text, t.name topic, qt.source"
        " FROM question_topic qt"
        " JOIN question q ON q.id = qt.question_id"
        " JOIN topic t ON t.id = qt.topic_id"
        " WHERE qt.source IN ('confirmed','corrected')"
    ).fetchall()
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps({
                "question_id": r["id"],
                "text": r["text"],
                "topic": r["topic"],
                "label": 1 if r["source"] == "confirmed" else 0,
            }) + "\n")
    return len(rows)
