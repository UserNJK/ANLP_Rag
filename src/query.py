from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import unicodedata

import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from retrieve import retrieve

# Load environment variables
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = "meta-llama/llama-3.1-70b-instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Initialize OpenAI client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _remove_emojis(text: str) -> str:
    out: list[str] = []
    for char in text:
        cp = ord(char)
        if (
            0x1F300 <= cp <= 0x1FAFF
            or 0x1F1E6 <= cp <= 0x1F1FF
            or 0x2600 <= cp <= 0x27BF
            or 0xFE00 <= cp <= 0xFE0F
            or cp == 0x200D
        ):
            continue
        out.append(char)
    return "".join(out)


def _sanitize_plain_text(text: str) -> str:
    cleaned = text or ""
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = _remove_emojis(cleaned)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _generate_rag_answer(question: str, chunks: list[dict]) -> str:
    """Generate an answer using LLM with retrieved context."""
    if not chunks:
        return "No relevant context found in the database."
    
    # Build context from retrieved chunks
    context = "\n\n".join([
        f"[Source: {c['source']}, Chunk {c['chunk']}]\n{c['text']}"
        for c in chunks
    ])
    
    # Create prompt for the LLM
    prompt = f"""Use only the context below to answer the question.
Write a complete and specific answer (target 180-300 words) with key details, timeline, and important terms when available.
If the context is insufficient, clearly state what is missing instead of guessing.
Output plain text only: do not use markdown formatting (especially bold like **text**) and do not use emojis.
Use this structure:
1) Direct answer
2) Key details
3) Timeline / periodization (if present)
4) Sources used

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a historical RAG assistant. Provide detailed, faithful answers grounded in the provided context only. Prefer completeness over brevity, but avoid fluff. Return plain text only. Do not use markdown bold or emojis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.13,
            max_tokens=1200,
        )
        return _sanitize_plain_text(response.choices[0].message.content.strip())
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def _generate_non_rag_answer(question: str) -> str:
    """Generate answer without RAG context (direct LLM query)."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question based on your training knowledge. Keep your answer concise — aim for 200-350 words. Do not write lengthy essays. Return plain text only. Do not use markdown bold or emojis."},
                {"role": "user", "content": question}
            ],
            temperature=0.13,
            max_tokens=600,
        )
        return _sanitize_plain_text(response.choices[0].message.content.strip())
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def _calculate_metrics(answer1: str, answer2: str) -> dict:
    """Calculate comparison metrics between two answers."""
    # ROUGE score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_score = scorer.score(answer1, answer2)["rougeL"].fmeasure
    
    # Semantic similarity
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    emb1, emb2 = model.encode([answer1, answer2], device=device)
    semantic_sim = _cosine(emb1, emb2)
    
    return {
        "rouge_l": rouge_score,
        "semantic_similarity": semantic_sim,
    }


def query(question: str, top_k: int = 10, show_context: bool = True) -> None:
    """Query the RAG system and compare with non-RAG answer."""
    if not OPENROUTER_API_KEY:
        raise SystemExit(
            "OPENROUTER_API_KEY not found. Please set it in your .env file."
        )
    
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # Retrieve context
    print(f"\nRetrieving top-{top_k} relevant chunks...")
    chunks = retrieve(question, top_k=top_k)
    
    if show_context and chunks:
        print("\nRETRIEVED CONTEXT:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  [{i}] Score: {chunk['score']:.3f} | {chunk['source']} (Chunk {chunk['chunk']})")
            print(f"      {chunk['text'][:200]}...")
    
    # Generate RAG answer
    print(f"\nGenerating RAG answer (with context)...")
    rag_answer = _generate_rag_answer(question, chunks)
    
    # Generate non-RAG answer
    print(f"Generating non-RAG answer (without context)...")
    non_rag_answer = _generate_non_rag_answer(question)
    
    # Calculate comparison metrics
    print(f"\nCalculating comparison metrics...")
    metrics = _calculate_metrics(rag_answer, non_rag_answer)
    
    # Display results
    print("\n" + "=" * 80)
    print("RAG ANSWER (with context):")
    print("=" * 80)
    print(rag_answer)
    
    print("\n" + "=" * 80)
    print("NON-RAG ANSWER (without context):")
    print("=" * 80)
    print(non_rag_answer)
    
    print("\n" + "=" * 80)
    print("COMPARISON METRICS:")
    print("=" * 80)
    print(f"  ROUGE-L Score:         {metrics['rouge_l']:.4f}")
    print(f"  Semantic Similarity:   {metrics['semantic_similarity']:.4f}")
    print(f"\n  Interpretation:")
    print(f"    - ROUGE-L measures word overlap (0=no overlap, 1=identical)")
    print(f"    - Semantic Similarity measures meaning similarity (0=different, 1=same)")
    print("=" * 80)
    
    # Additional insights
    if metrics['semantic_similarity'] > 0.8:
        print("\nBoth answers are semantically very similar.")
    elif metrics['semantic_similarity'] < 0.5:
        print("\nThe answers differ significantly - RAG provides context-specific information.")
    else:
        print("\nThe answers have moderate similarity.")


def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG system and compare with non-RAG answer."
    )
    parser.add_argument("question", nargs="?", help="Your question")
    parser.add_argument("--top-k", type=int, default=10, help="Number of chunks to retrieve (default: 10)")
    parser.add_argument("--no-context", action="store_true", help="Hide retrieved context")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or not args.question:
        print("Interactive Query Mode (Ctrl+C to exit)")
        print("-" * 80)
        while True:
            try:
                question = input("\nEnter your question: ").strip()
                if not question:
                    print("Please enter a question.")
                    continue
                query(question, top_k=args.top_k, show_context=not args.no_context)
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    else:
        query(args.question, top_k=args.top_k, show_context=not args.no_context)


if __name__ == "__main__":
    main()
