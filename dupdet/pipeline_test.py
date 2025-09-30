# pipeline_test.py
import json
from dupdet import batch_fill, similar_posts, delete_post

def run_pipeline_demo():
    print("=== Pipeline Demo: batch_fill -> similar_posts ===\n")

    # Demo posts in EN + FR
    posts = [
        ("p1", "this is about cars"),
        ("p2", "c'est à propos des voitures"),
        ("p3", "oranges are the best fruit"),
        ("p4", "les oranges sont des fruits"),
        ("p5", "Ukraine war"),
    ]

    topic = "topic1"

    # 1) Ingest posts + embed any missing vectors
    print("Batch filling (ingest + embed missing)...")
    batch_fill(topic, posts)

    # 2) Query examples
    queries = [
        "this is about cars",
        "c'est à propos des voitures",
        "oranges are the best fruit",
        "les oranges sont des fruits",
        "Ukraine war",
    ]

    for q in queries:
        print(f"\nQuery: {q}")
        sims = similar_posts(q, topic=topic, min_score=0.80, top_k=5)
        if sims:
            for pid, score in sims:
                print(f"  -> {pid}  (score={score:.3f})")
        else:
            print("  -> No matches above threshold")

    # 3) Cleanup demo data (optional)
    for pid, _ in posts:
        delete_post(pid)

if __name__ == "__main__":
    run_pipeline_demo()
