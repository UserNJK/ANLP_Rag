from __future__ import annotations

import json
import re
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _tokenize(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-Z]+", text.lower())
        if len(tok) > 2
    }


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    query_vec = model.encode([question], device=device)
    query_vec = np.asarray(query_vec, dtype="float32")
    faiss.normalize_L2(query_vec)

    fetch_k = min(len(metadata), max(top_k * 8, 40))
    scores, indices = index.search(query_vec, fetch_k)

    question_tokens = _tokenize(question)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = metadata[int(idx)]

        text_tokens = _tokenize(chunk.get("text", ""))
        source_tokens = _tokenize(str(chunk.get("source", "")))
        chapter_tokens = _tokenize(str(chunk.get("chapter", "")))

        text_overlap = (
            len(question_tokens & text_tokens) / max(len(question_tokens), 1)
        )
        source_overlap = (
            len(question_tokens & source_tokens) / max(len(question_tokens), 1)
        )
        chapter_overlap = (
            len(question_tokens & chapter_tokens) / max(len(question_tokens), 1)
        )

        combined_score = (
            (0.72 * float(score))
            + (0.16 * text_overlap)
            + (0.08 * source_overlap)
            + (0.04 * chapter_overlap)
        )

        results.append(
            {
                "score": combined_score,
                "semantic_score": float(score),
                "keyword_overlap": text_overlap,
                "source_overlap": source_overlap,
                "chapter_overlap": chapter_overlap,
                **chunk,
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)

    deduped_results = []
    seen_keys = set()
    for item in results:
        dedup_key = (
            str(item.get("id", "")),
            str(item.get("source", "")),
            str(item.get("chunk", "")),
        )
        if dedup_key in seen_keys:
            continue
        seen_keys.add(dedup_key)
        deduped_results.append(item)
        if len(deduped_results) >= top_k:
            break

    return deduped_results


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
