# simple_test.py
from dupdet import record_post, similar_posts, batch_fill, delete_post

TOPIC = "demo"

POSTS = [
    ("e1", "This idea is excellent!"),
    ("e2", "C’est une excellente idée !"),
    ("e3", "Bu fikir harika!"),
    ("e4", "We must cut CO₂ emissions significantly."),
    ("e5", "I love strawberry ice cream."),
]
TEXT = dict(POSTS)

def line():
    print("-" * 72)

def show_hits(query, hits):
    print(f"\nQuery: {query!r}")
    line()
    print(f"{'Post ID':<8} {'Score':<7} Text")
    line()
    for pid, score in hits:
        print(f"{pid:<8} {score:.3f}   {TEXT.get(pid, '(text not in map)')}")
    line()

def check(cond, msg):
    if cond:
        print(f"✅ {msg}")
    else:
        print(f"❌ {msg}")
        raise SystemExit(1)

def main():
    # Start from known state: batch insert our demo posts
    batch_fill(TOPIC, POSTS)

    # 1) Great idea!
    hits1 = similar_posts("Great idea!", top_k=5, min_score=0.35, topic=TOPIC)
    show_hits("Great idea!", hits1)
    # Expect idea variants high (e1/e2/e3)
    check(any(pid in {"e1","e2","e3"} for pid, _ in hits1), "Found at least one 'idea' post")

    # 2) Emissions
    hits2 = similar_posts("Reduce CO2 emissions", top_k=5, min_score=0.35, topic=TOPIC)
    show_hits("Reduce CO2 emissions", hits2)
    # Expect emissions post e4
    check(any(pid == "e4" for pid, _ in hits2), "Found the emissions post")

    # 3) Ice cream
    hits3 = similar_posts("ice cream", top_k=5, min_score=0.35, topic=TOPIC)
    show_hits("ice cream", hits3)
    # Expect ice cream post e5
    check(any(pid == "e5" for pid, _ in hits3), "Found the ice cream post")

    # 4) Delete behavior 
    record_post("to_del", "This idea is excellent indeed!", topic=TOPIC)
    hits4 = similar_posts("Great idea!", top_k=10, min_score=0.35, topic=TOPIC)
    check(any(pid == "to_del" for pid, _ in hits4), "Post to delete appears before deletion")
    delete_post("to_del")
    hits5 = similar_posts("Great idea!", top_k=10, min_score=0.35, topic=TOPIC)
    check(all(pid != "to_del" for pid, _ in hits5), "Deleted post no longer appears")

    print("\n🎉 SIMPLE TEST PASSED")

if __name__ == "__main__":
    main()
