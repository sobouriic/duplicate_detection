#!/usr/bin/env python3
"""
Pairwise similarity test using the same model as dupdet.
Shows RAW cosine and CALIBRATED scores and writes to:
  - embedding_results.md
  - embedding_results.csv
"""

from dupdet.embedder import embed_text_document
from dupdet.calibration import calibrate
from itertools import product
import numpy as np
from pathlib import Path
import csv

SENTS = [
    "this is about cars",
    "oranges are the best fruit",
    "Ukraine war",
    "grapefruit",
    "c'est à propos des voitures",
    "les oranges sont des fruits",
]


def _l2(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return (v / max(n, 1e-12)).astype(np.float32)


def pairwise_similarities(texts):
    # Pre-embed
    embs = [_l2(embed_text_document(t)) for t in texts]
    results = []
    for i, j in product(range(len(texts)), repeat=2):
        a = embs[i]; b = embs[j]
        raw = float(np.dot(a, b))
        cal = calibrate(raw)
        results.append((texts[i], texts[j], cal, raw))
    return results


def write_markdown(results, md_path: Path):
    lines = []
    lines.append("=== Pairwise Similarities (bge-m3) ===\n\n")
    lines.append("Text A | Text B | Calibrated | Raw\n")
    lines.append("---|---|---:|---:\n")
    for a, b, cal, raw in results:
        lines.append(f"{a} | {b} | {cal:.6f} | {raw:.6f}\n")
    md_path.write_text("".join(lines), encoding="utf-8")


def write_csv(results, csv_path: Path):
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text_a", "text_b", "calibrated", "raw"])
        for a, b, cal, raw in results:
            w.writerow([a, b, f"{cal:.6f}", f"{raw:.6f}"])


if __name__ == "__main__":
    res = pairwise_similarities(SENTS)

    # Pretty print to console
    print("Pairwise cosine similarities (Calibrated | Raw):\n")
    # Group as blocks like your old output
    for i, a in enumerate(SENTS):
        for j, b in enumerate(SENTS):
            cal = res[i*len(SENTS)+j][2]
            raw = res[i*len(SENTS)+j][3]
            print(f"{a:<30} {b:<30} {cal:.6f} | {raw:.6f}")
        print()

    # Write files
    md = Path("embedding_results.md")
    csvf = Path("embedding_results.csv")
    write_markdown(res, md)
    write_csv(res, csvf)
    print(f"Saved: {md.resolve()}")
    print(f"Saved: {csvf.resolve()}")
