import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def l2(v):
    v = np.asarray(v, dtype=np.float32)
    return v / max(np.linalg.norm(v), 1e-12)

def cos(a, b):
    return float(np.dot(a, b))

def run_test(model_name, sentences):
    print(f"\n=== Testing model: {model_name} ===")
    embedder = HuggingFaceEmbedding(model_name=model_name)
    embs = [l2(embedder.get_text_embedding(s)) for s in sentences]

    print("\nPairwise cosine similarities:\n")
    for i, si in enumerate(sentences):
        for j, sj in enumerate(sentences):
            if j >= i:  # upper triangle
                print(f"{si:<35} {sj:<35} {cos(embs[i], embs[j]):.6f}")
    print("-" * 60)


if __name__ == "__main__":
    # Test sentences in English & French
    sentences = [
        "this is about cars",
        "oranges are the best fruit",
        "Ukraine war",
        "grapefruit",
        "c'est à propos des voitures",
        "les oranges sont des fruits",
    ]

    # Old model (Gemma or E5)
    run_test("intfloat/multilingual-e5-base", sentences)

    # New model (BGE-m3)
    run_test("BAAI/bge-m3", sentences)
