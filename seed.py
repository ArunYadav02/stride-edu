"""
Seed Stride with a realistic corpus.

The papers here are SYNTHETIC — written to exercise the pipeline, not copied
from any institution. Institution names are fictional. This keeps the demo
runnable and shareable without redistributing anyone's copyrighted material,
which is the same constraint the production system enforces via the schema.
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import fitz
from backend.db import (connect, init_db, insert_paper, insert_question,
                        upsert_institution, upsert_module)
from backend.ingest import parse_paper
from backend.topics import tag_and_store

random.seed(42)

INSTITUTIONS = [
    ("Northgate University", "Northgate", "granted", "Confirmed by the CS department, Feb 2026."),
    ("Ashfield Institute of Technology", "Ashfield", "granted", "Indexing agreed for public papers."),
    ("Riverbank College London", "Riverbank", "requested", "Awaiting reply from the exams office."),
    ("Kestrel University", "Kestrel", "not_asked", None),
]

MODULES = [
    ("Northgate", "COMP2011", "Algorithms and Data Structures", "Year 2", [
        ("Sorting", ["Compare merge sort and quicksort, giving worst-case time complexity for each.",
                     "Explain why quicksort's pivot choice affects its performance, and describe median-of-three partitioning.",
                     "Show the state of the array after each pass of insertion sort on [5, 2, 9, 1, 7]."]),
        ("Graph algorithms", ["State Dijkstra's algorithm and prove it computes shortest paths on graphs with non-negative weights.",
                              "Give a graph with negative edges where Dijkstra's algorithm produces an incorrect shortest path.",
                              "Describe Kruskal's algorithm for the minimum spanning tree and analyse its complexity using union-find.",
                              "Explain breadth-first search and give one problem it solves that depth-first search does not."]),
        ("Dynamic programming", ["Describe the dynamic programming solution to the 0/1 knapsack problem, stating the recurrence relation.",
                                 "Give a dynamic programming algorithm for the longest common subsequence of two strings.",
                                 "Explain optimal substructure and overlapping subproblems, with an example of each.",
                                 "Compute the edit distance between 'kitten' and 'sitting' using dynamic programming."]),
        ("Trees and balanced structures", ["Define a red-black tree and state its invariants.",
                                           "Show the red-black tree resulting from inserting 7, 3, 18, 10, 22 into an empty tree.",
                                           "Prove that a red-black tree with n nodes has height at most 2 log(n+1).",
                                           "Compare AVL trees and red-black trees for insert-heavy workloads."]),
        ("Complexity analysis", ["Use the master theorem to solve T(n) = 2T(n/2) + n.",
                                 "Define NP-completeness and explain what a polynomial-time reduction shows.",
                                 "Give the amortised analysis of a dynamic array that doubles when full."]),
        ("Hashing", ["Explain open addressing and separate chaining for collision resolution in a hash table.",
                     "Derive the expected number of probes for a hash table with load factor alpha."]),
    ]),
    ("Northgate", "COMP3021", "Database Systems", "Year 3", [
        ("Relational design", ["Explain the difference between second and third normal form, with an example.",
                               "Normalise the relation Order(id, customer, address, item, price) to third normal form.",
                               "Define functional dependency and give the closure of a set of attributes.",
                               "State Boyce-Codd normal form and give a relation in 3NF but not BCNF."]),
        ("Transactions and concurrency", ["Define the ACID properties of a database transaction.",
                                          "Explain two-phase locking and show how it produces serialisable schedules.",
                                          "Describe write-ahead logging and how it supports crash recovery.",
                                          "Compare optimistic and pessimistic concurrency control."]),
        ("Query processing", ["Write SQL to find customers who have ordered every product in the catalogue.",
                              "Explain how a B-tree index speeds up a range query, and when a full scan is faster.",
                              "Describe the nested-loop, hash and merge join algorithms and their cost models."]),
    ]),
    ("Ashfield", "INF4210", "Machine Learning", "Year 4", [
        ("Supervised learning", ["Derive the gradient of the cross-entropy loss for logistic regression.",
                                 "Compare decision trees and support vector machines for high-dimensional data.",
                                 "Explain how a random forest reduces variance relative to a single decision tree."]),
        ("Neural networks", ["Explain backpropagation through a two-layer feedforward network.",
                             "Describe the vanishing gradient problem and explain how ReLU activation mitigates it.",
                             "Explain the role of the learning rate in gradient descent and what happens if it is too large.",
                             "Describe how a convolutional layer differs from a fully connected layer."]),
        ("Model evaluation", ["Describe the bias-variance decomposition of expected prediction error.",
                              "A model has 2% training error and 24% validation error. Diagnose the problem and propose three remedies.",
                              "Explain k-fold cross validation and why it is preferred to a single holdout split.",
                              "Compare L1 and L2 regularisation and explain when each is appropriate."]),
    ]),
    ("Ashfield", "INF3300", "Operating Systems", "Year 3", [
        ("Operating systems", ["Explain virtual memory and describe how a page fault is handled.",
                               "Compare the LRU and clock page replacement algorithms.",
                               "Describe the steps in a context switch and explain why it is expensive.",
                               "Explain the difference between a process and a thread."]),
        ("Concurrency", ["Define a race condition and give an example involving two threads.",
                         "Explain how semaphores solve the producer-consumer problem.",
                         "State the four conditions necessary for deadlock and describe one prevention strategy."]),
    ]),
    ("Riverbank", "CS2400", "Networks and Security", "Year 2", [
        ("Networking", ["Describe the TCP three-way handshake and explain its purpose.",
                        "Explain TCP congestion control, including slow start and congestion avoidance.",
                        "Compare TCP and UDP, giving an application suited to each."]),
        ("Cryptography", ["Explain public key cryptography and describe the RSA key exchange.",
                          "Describe how a digital signature provides authenticity and non-repudiation.",
                          "Explain the difference between a block cipher and a stream cipher."]),
        ("Software security", ["Explain how a buffer overflow works and describe two mitigations.",
                               "Describe SQL injection and explain how parameterised queries prevent it."]),
    ]),
    ("Kestrel", "CSC2150", "Theory of Computation", "Year 2", [
        ("Formal languages", ["Construct a deterministic finite automaton accepting binary strings divisible by three.",
                              "State the pumping lemma for regular languages and use it to show a^n b^n is not regular.",
                              "Convert the given context-free grammar to Chomsky normal form.",
                              "Describe a Turing machine that decides whether its input is a palindrome."]),
        ("Logic and proof", ["Prove by induction that the sum of the first n odd numbers is n squared.",
                             "Explain what a loop invariant is and use one to prove a binary search correct.",
                             "Define satisfiability and explain why SAT is NP-complete."]),
    ]),
]

def build_pdf(path, module_code, module_title, year, questions, style):
    doc = fitz.open(); page = doc.new_page(); y = 60
    def w(t, size=11, bold=False, indent=0, gap=5):
        nonlocal y, page
        if y > 750: page = doc.new_page(); y = 60
        for chunk in [t[i:i+95] for i in range(0, len(t), 95)] or [""]:
            page.insert_text((60+indent, y), chunk, fontsize=size,
                             fontname="hebo" if bold else "helv")
            y += size + gap
    total = sum(m for _, m in questions)
    w(f"{module_code} {module_title}", 14, True)
    w(f"{year} Examination — Time allowed: 2 hours", 10)
    w(f"Answer ALL questions. Total: {total} marks", 10); y += 12
    for i, (text, marks) in enumerate(questions, 1):
        label = {"A": f"Question {i}", "B": f"{i}.", "C": f"Q{i}"}[style]
        w(label, 11, True, 0, 6)
        w(f"{text} [{marks} marks]", 10, False, 20, 4)
        y += 6
    doc.save(path); doc.close()

def main():
    db = Path("stride.db")
    if db.exists(): db.unlink()
    init_db(db)
    tmp = Path("/tmp/stride_seed"); tmp.mkdir(exist_ok=True)

    inst_ids = {}
    with connect(db) as conn:
        for name, short, status, note in INSTITUTIONS:
            inst_ids[short] = upsert_institution(conn, name, short, status, note)

    n_papers = n_questions = 0
    styles = {"Northgate": "A", "Ashfield": "C", "Riverbank": "B", "Kestrel": "A"}

    for inst_short, code, title, level, topic_bank in MODULES:
        with connect(db) as conn:
            mod_id = upsert_module(conn, inst_ids[inst_short], code, title, level)

        for year in range(2019, 2026):
            # Each year samples questions, so topics genuinely vary across years
            # and the frequency analytics have something real to find.
            picked = []
            for topic, pool in topic_bank:
                k = random.choice([0, 1, 1, 1, 2])
                for q in random.sample(pool, min(k, len(pool))):
                    picked.append((q, random.choice([6, 8, 10, 12, 15, 20])))
            random.shuffle(picked)
            picked = picked[:random.randint(4, 7)]
            if not picked: continue

            pdf = tmp / f"{code}_{year}.pdf"
            build_pdf(pdf, code, title, year, picked, styles[inst_short])
            parsed = parse_paper(pdf)
            if not parsed.questions:
                print(f"  ! {code} {year}: {parsed.note}"); continue

            with connect(db) as conn:
                pid = insert_paper(
                    conn, mod_id, year, source_url=f"https://example.ac.uk/{code}/{year}",
                    total_marks=parsed.total_marks, duration_min=parsed.duration_min,
                    parse_quality=parsed.confidence, parse_note=parsed.note)
                for q in parsed.flat():
                    qid = insert_question(conn, pid, q.number, q.text, q.position,
                                          marks=q.marks, page=q.page, depth=q.depth)
                    tag_and_store(conn, qid, q.text)
                    n_questions += 1
            n_papers += 1
        print(f"  {code}: seeded")

    with connect(db) as conn:
        from backend.search import corpus_stats
        s = corpus_stats(conn)
    print(f"\nSeeded {n_papers} papers, {n_questions} questions")
    print(f"Corpus: {s}")

if __name__ == "__main__":
    main()
