import os, sys, time
from dupdet import record_post, similar_posts, batch_fill, delete_post
from dupdet.config import CFG

def expect(cond, msg):
    if not cond:
        print("❌", msg); sys.exit(1)
    print("✅", msg)

def main():
    # fresh DB
    if os.path.exists(CFG.db_path): os.remove(CFG.db_path)
    print("🧪 Using DB:", CFG.db_path)

    # 1) record_post
    record_post("r1", "This is a great idea!", topic="demo")
    record_post("r2", "C’est une excellente idée !", topic="demo")
    expect(os.path.exists(CFG.db_path), "DB file created by record_post")

    # 2) similar_posts
    hits = similar_posts("Great idea!", top_k=5, min_score=0.30, topic="demo")
    print("similar_posts hits:", hits)
    expect(len(hits) >= 1, "similar_posts returns >= 1 result")
    expect(isinstance(hits[0], tuple) and isinstance(hits[0][0], str) and isinstance(hits[0][1], float),
           "similar_posts returns [(post_id:str, score:float), ...]")

    # 3) batch_fill
    batch_fill("demo", [("b1", "Bu fikir harika!"), ("b2", "هذه فكرة رائعة!")])
    hits2 = similar_posts("This proposal is great", top_k=10, min_score=0.30, topic="demo")
    print("after batch, hits:", hits2)
    expect(any(pid in {"b1", "b2"} for pid, _ in hits2), "batch_fill posts appear in results")

    # 4) delete_post
    record_post("d1", "Esta propuesta es fantástica.", topic="demo")
    hits3 = similar_posts("Esta propuesta es fantástica.", top_k=10, min_score=0.30, topic="demo")
    print("before delete:", hits3)
    expect(any(pid == "d1" for pid, _ in hits3), "post to delete is searchable first")

    ok = delete_post("d1")
    expect(ok is True, "delete_post returns True")

    hits4 = similar_posts("Esta propuesta es fantástica.", top_k=10, min_score=0.30, topic="demo")
    print("after delete:", hits4)
    expect(all(pid != "d1" for pid, _ in hits4), "deleted post is not returned")

    print("\n🎉 All function-level tests passed.")

if __name__ == "__main__":
    main()
