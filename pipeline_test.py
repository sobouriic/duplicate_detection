#!/usr/bin/env python3
"""
Pipeline demo that:
  1) records posts
  2) batch-fills missing embeddings
  3) queries with similar_posts (raw) and similar_posts_calibrated (calibrated + raw)
  4) writes results to Markdown and CSV

Outputs:
  - pipeline_results.md
  - pipeline_results.csv
"""

from dupdet import record_post, batch_fill, similar_posts, similar_posts_old
from pathlib import Path
import csv

TOPIC = "topic1"

POSTS = [
    ("p1", "this is about cars"),
    ("p2", "c'est à propos des voitures"),
    ("p3", "oranges are the best fruit"),
    ("p4", "les oranges sont des fruits"),
    ("p5", "Ukraine war"),
]

QUERIES = [
    "this is about cars",
    "c'est à propos des voitures",
    "oranges are the best fruit",
    "les oranges sont des fruits",
    "Ukraine war",
]

MIN_SCORE = 0.80
TOP_K = 10


def _write_markdown(rows, md_path: Path):
    md_lines = []
    md_lines.append("=== Pipeline Test Results ===\n")
    md_lines.append("\n--- Workflow A: record_post + batch_fill(topic) ---\n")
    md_lines.append("\nQuery | Post ID | Calibrated | Raw\n")
    md_lines.append("---|---:|---:|---:\n")
    for r in rows:
        md_lines.append(f"{r['query']} | {r['post_id']} | "
                        f"{'' if r['calibrated'] is None else f'{r['calibrated']:.3f}'} | "
                        f"{'' if r['raw'] is None else f'{r['raw']:.3f}'}\n")
    md_path.write_text("".join(md_lines), encoding="utf-8")


def _write_csv(rows, csv_path: Path):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["workflow", "query", "post_id", "calibrated", "raw"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_pipeline_demo():
    print("=== Pipeline Demo: record -> batch_fill -> similar_posts ===\n")

    # --- Workflow A: record_post + batch_fill(topic) ---
    print("\n--- Workflow A: record_post + batch_fill(topic) ---\n")
    for pid, text in POSTS:
        record_post(pid, text, TOPIC)
    batch_fill(TOPIC)  # fill embeddings for those posts

    rows = []
    for q in QUERIES:
        hits = similar_posts(q, topic=TOPIC, top_k=TOP_K, min_score=MIN_SCORE)
        if not hits:
            print(f"Query: {q}\n  -> No matches above threshold\n")
            rows.append({"workflow": "A", "query": q, "post_id": "-", "calibrated": None, "raw": None})
            continue

        print(f"Query: {q}")
        for pid, cal, raw in hits:
            print(f"  ->  {pid}  (cal={cal:.3f}, raw={raw:.3f})")
            rows.append({"workflow": "A", "query": q, "post_id": pid, "calibrated": cal, "raw": raw})
        print()

    # --- Workflow B: batch_fill(topic, posts) ---
    print("\n--- Workflow B: batch_fill(topic, posts) ---\n")
    batch_fill(TOPIC, POSTS)

    for q in QUERIES:
        hits = similar_posts(q, topic=TOPIC, top_k=TOP_K, min_score=MIN_SCORE)
        if not hits:
            print(f"Query: {q}\n  -> No matches above threshold\n")
            rows.append({"workflow": "B", "query": q, "post_id": "-", "calibrated": None, "raw": None})
            continue

        print(f"Query: {q}")
        for pid, cal, raw in hits:
            print(f"  ->  {pid}  (cal={cal:.3f}, raw={raw:.3f})")
            rows.append({"workflow": "B", "query": q, "post_id": pid, "calibrated": cal, "raw": raw})
        print()

    # Write files
    out_md = Path("pipeline_results.md")
    out_csv = Path("pipeline_results.csv")
    _write_markdown(rows, out_md)
    _write_csv(rows, out_csv)
    print(f"\nSaved: {out_md.resolve()}")
    print(f"Saved: {out_csv.resolve()}")


if __name__ == "__main__":
    try:
        # newer signature requires (topic, posts)
        batch_fill(TOPIC, POSTS)  # dry run; embedding will be refilled later anyway
    except TypeError:
        # older signature: batch_fill(topic) only
        pass

    run_pipeline_demo()
