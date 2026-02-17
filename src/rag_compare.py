from __future__ import annotations

import json
from pathlib import Path

from retrieve import retrieve

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


def _make_rag_answer(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant context found."
    # Simple extractive answer from top chunks.
    top_texts = [c["text"] for c in chunks[:2]]
    return " ".join(top_texts)


def _make_non_rag_answer() -> str:
    return "Insufficient information without retrieval context."


def run(top_k: int = 3) -> None:
    questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
    outputs = []
    for item in questions:
        q = item["question"]
        hits = retrieve(q, top_k=top_k)
        outputs.append(
            {
                "question": q,
                "reference": item["answer"],
                "rag_answer": _make_rag_answer(hits),
                "non_rag_answer": _make_non_rag_answer(),
                "contexts": hits,
            }
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "answers.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(outputs)} answers to artifacts/answers.json")


if __name__ == "__main__":
    run()
