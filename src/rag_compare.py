from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from retrieve import retrieve

# Load environment variables
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def _generate_llm_answer(question: str, chunks: list[dict]) -> str:
    """Generate an answer using LLM with retrieved context."""
    if not chunks:
        return "No relevant context found."
    
    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Source: {c['source']}, Chunk {c['chunk']}]\n{c['text']}"
        for c in chunks
    ])
    
    # Create prompt for the LLM
    prompt = f"""Use only the context below to answer the question.
Write a detailed and comprehensive answer (target 200-350 words) with key facts, dates, names, and important terms when available.
If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Prefer completeness and detail over brevity — aim for 200-350 words."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def _make_rag_answer(question: str, chunks: list[dict]) -> str:
    """Generate RAG answer using LLM with retrieved context."""
    return _generate_llm_answer(question, chunks)


def _make_non_rag_answer(question: str) -> str:
    """Generate answer without RAG context (direct LLM query)."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question based on your training knowledge. Keep your answer concise — aim for 200-350 words. Do not write lengthy essays."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def run(top_k: int = 3) -> None:
    if not OPENROUTER_API_KEY:
        raise SystemExit(
            "OPENROUTER_API_KEY not found. Please set it in your .env file or environment."
        )
    
    questions = json.loads((DATA_DIR / "questions.json").read_text(encoding="utf-8"))
    outputs = []
    for i, item in enumerate(questions, 1):
        q = item["question"]
        print(f"Processing question {i}/{len(questions)}: {q}")
        
        hits = retrieve(q, top_k=top_k)
        rag_answer = _make_rag_answer(q, hits)
        non_rag_answer = _make_non_rag_answer(q)
        
        outputs.append(
            {
                "question": q,
                "reference": item["answer"],
                "rag_answer": rag_answer,
                "non_rag_answer": non_rag_answer,
                "contexts": hits,
            }
        )
        print(f"  RAG: {rag_answer[:100]}...")
        print(f"  Non-RAG: {non_rag_answer[:100]}...")
        print()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "answers.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8"
    )
    print(f"✓ Wrote {len(outputs)} answers to artifacts/answers.json")


if __name__ == "__main__":
    run()
