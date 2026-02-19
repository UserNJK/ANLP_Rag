from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "artifacts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize_text(text: str) -> str:
    text = text.replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _read_jsonl_files(data_dir: Path) -> Iterable[dict]:
    """Read JSONL files and yield enriched document chunks."""
    for jsonl_path in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        yield {
                            "source_file": jsonl_path.name,
                            "chunk_id": record.get("chunk_id", ""),
                            "book_title": record.get("book_title", ""),
                            "chapter_title": record.get("chapter_title", ""),
                            "content": _normalize_text(record.get("content", "")),
                        }
                    except json.JSONDecodeError:
                        continue


def _is_high_quality_chunk(text: str) -> bool:
    words = text.split()
    if len(words) < 35:
        return False

    chars = [ch for ch in text if not ch.isspace()]
    if not chars:
        return False

    alpha_chars = sum(1 for ch in chars if ch.isalpha())
    alpha_ratio = alpha_chars / len(chars)
    if alpha_ratio < 0.62:
        return False

    bad_char_ratio = sum(1 for ch in chars if ch in {"~", "|", "_", "^"}) / len(chars)
    if bad_char_ratio > 0.08:
        return False

    garbage_tokens = 0
    for token in words:
        clean = re.sub(r"[^A-Za-z]", "", token)
        if len(clean) >= 6:
            vowels = sum(1 for ch in clean.lower() if ch in "aeiou")
            if vowels == 0:
                garbage_tokens += 1

    if garbage_tokens / max(len(words), 1) > 0.08:
        return False

    return True



def build_index() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = []
    skipped_low_quality = 0
    skipped_empty = 0
    for record in _read_jsonl_files(DATA_DIR):
        content = record["content"]
        if not content:
            skipped_empty += 1
            continue
        if not _is_high_quality_chunk(content):
            skipped_low_quality += 1
            continue

        docs.append(
            {
                "id": record["chunk_id"],
                "source": record["book_title"] or record["source_file"],
                "chapter": record["chapter_title"],
                "chunk": record["chunk_id"].split("_")[-1] if "_" in record["chunk_id"] else "0",
                "text": content,
            }
        )

    if not docs:
        raise SystemExit("No JSONL content found. Add JSONL files to ./data and retry.")

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = SentenceTransformer(MODEL_NAME, device=device)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, device=device)
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(docs, indent=2), encoding="utf-8"
    )

    print(f"Indexed {len(docs)} chunks to {OUT_DIR}")
    print(f"Skipped empty chunks: {skipped_empty}")
    print(f"Skipped low-quality chunks: {skipped_low_quality}")


if __name__ == "__main__":
    build_index()
