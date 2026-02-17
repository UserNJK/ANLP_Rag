from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_index() -> tuple[faiss.IndexFlatIP, list[dict]]:
    index_path = ARTIFACTS_DIR / "index.faiss"
    meta_path = ARTIFACTS_DIR / "metadata.json"
    if not index_path.exists() or not meta_path.exists():
        raise SystemExit("Index not found. Run: python src\\build_index.py")

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metadata


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    index, metadata = _load_index()
    model = SentenceTransformer(MODEL_NAME)
    query_vec = model.encode([question])
    query_vec = np.asarray(query_vec, dtype="float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = metadata[int(idx)]
        results.append({"score": float(score), **chunk})
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Retrieve top-k chunks for a query.")
    parser.add_argument("question", help="User question")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    hits = retrieve(args.question, args.top_k)
    for h in hits:
        print(f"[{h['score']:.3f}] {h['source']}::{h['chunk']}")
        print(h["text"])
        print("-")
