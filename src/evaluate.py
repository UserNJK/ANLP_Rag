from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

ROOT_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def evaluate() -> None:
    answers_path = ARTIFACTS_DIR / "answers.json"
    if not answers_path.exists():
        raise SystemExit("answers.json not found. Run: python src\\rag_compare.py")

    data = json.loads(answers_path.read_text(encoding="utf-8"))

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    model = SentenceTransformer(MODEL_NAME)

    results = []
    for item in data:
        ref = item["reference"]
        rag = item["rag_answer"]
        non_rag = item["non_rag_answer"]

        rag_rouge = scorer.score(ref, rag)["rougeL"].fmeasure
        non_rag_rouge = scorer.score(ref, non_rag)["rougeL"].fmeasure

        emb_ref, emb_rag, emb_non = model.encode([ref, rag, non_rag])
        rag_sim = _cosine(emb_ref, emb_rag)
        non_rag_sim = _cosine(emb_ref, emb_non)

        results.append(
            {
                "question": item["question"],
                "rag_rougeL": rag_rouge,
                "non_rag_rougeL": non_rag_rouge,
                "rag_semantic": rag_sim,
                "non_rag_semantic": non_rag_sim,
            }
        )

    avg = {
        "rag_rougeL": float(np.mean([r["rag_rougeL"] for r in results])),
        "non_rag_rougeL": float(np.mean([r["non_rag_rougeL"] for r in results])),
        "rag_semantic": float(np.mean([r["rag_semantic"] for r in results])),
        "non_rag_semantic": float(np.mean([r["non_rag_semantic"] for r in results])),
    }

    out = {"per_question": results, "average": avg}
    (ARTIFACTS_DIR / "metrics.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    print("Average scores:")
    print(json.dumps(avg, indent=2))


if __name__ == "__main__":
    evaluate()
