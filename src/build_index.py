from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from PyPDF2 import PdfReader
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


def _read_pdfs(data_dir: Path) -> Iterable[tuple[str, str]]:
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        pages = [page.extract_text() or "" for page in reader.pages]
        raw_text = "\n".join(pages)
        yield pdf_path.name, _normalize_text(raw_text)


def _chunk_words(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def build_index() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    docs = []
    for source, text in _read_pdfs(DATA_DIR):
        for idx, chunk in enumerate(_chunk_words(text)):
            docs.append(
                {
                    "id": f"{source}::chunk-{idx}",
                    "source": source,
                    "chunk": idx,
                    "text": chunk,
                }
            )

    if not docs:
        raise SystemExit("No PDF text found. Add PDFs to ./data and retry.")

    model = SentenceTransformer(MODEL_NAME)
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(docs, indent=2), encoding="utf-8"
    )

    print(f"Indexed {len(docs)} chunks to {OUT_DIR}")


if __name__ == "__main__":
    build_index()
