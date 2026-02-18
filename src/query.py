from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
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
    prompt = f"""Based on the following context, answer the question. If the answer cannot be found in the context, say so.

Context:
{context}

Question: {question}

Answer:"""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. Be concise and accurate."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def _generate_non_rag_answer(question: str) -> str:
    """Generate answer without RAG context (direct LLM query)."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question based on your training knowledge."},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def _calculate_metrics(answer1: str, answer2: str) -> dict:
    """Calculate comparison metrics between two answers."""
    # ROUGE score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_score = scorer.score(answer1, answer2)["rougeL"].fmeasure
    
    # Semantic similarity
    model = SentenceTransformer(EMBEDDING_MODEL)
    emb1, emb2 = model.encode([answer1, answer2])
    semantic_sim = _cosine(emb1, emb2)
    
    return {
        "rouge_l": rouge_score,
        "semantic_similarity": semantic_sim,
    }


def query(question: str, top_k: int = 3, show_context: bool = True) -> None:
    """Query the RAG system and compare with non-RAG answer."""
    if not OPENROUTER_API_KEY:
        raise SystemExit(
            "OPENROUTER_API_KEY not found. Please set it in your .env file."
        )
    
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # Retrieve context
    print(f"\n🔍 Retrieving top-{top_k} relevant chunks...")
    chunks = retrieve(question, top_k=top_k)
    
    if show_context and chunks:
        print("\n📚 RETRIEVED CONTEXT:")
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  [{i}] Score: {chunk['score']:.3f} | {chunk['source']} (Chunk {chunk['chunk']})")
            print(f"      {chunk['text'][:200]}...")
    
    # Generate RAG answer
    print(f"\n🤖 Generating RAG answer (with context)...")
    rag_answer = _generate_rag_answer(question, chunks)
    
    # Generate non-RAG answer
    print(f"🤖 Generating non-RAG answer (without context)...")
    non_rag_answer = _generate_non_rag_answer(question)
    
    # Calculate comparison metrics
    print(f"\n📊 Calculating comparison metrics...")
    metrics = _calculate_metrics(rag_answer, non_rag_answer)
    
    # Display results
    print("\n" + "=" * 80)
    print("✅ RAG ANSWER (with context):")
    print("=" * 80)
    print(rag_answer)
    
    print("\n" + "=" * 80)
    print("❌ NON-RAG ANSWER (without context):")
    print("=" * 80)
    print(non_rag_answer)
    
    print("\n" + "=" * 80)
    print("📊 COMPARISON METRICS:")
    print("=" * 80)
    print(f"  ROUGE-L Score:         {metrics['rouge_l']:.4f}")
    print(f"  Semantic Similarity:   {metrics['semantic_similarity']:.4f}")
    print(f"\n  Interpretation:")
    print(f"    - ROUGE-L measures word overlap (0=no overlap, 1=identical)")
    print(f"    - Semantic Similarity measures meaning similarity (0=different, 1=same)")
    print("=" * 80)
    
    # Additional insights
    if metrics['semantic_similarity'] > 0.8:
        print("\n💡 Both answers are semantically very similar.")
    elif metrics['semantic_similarity'] < 0.5:
        print("\n💡 The answers differ significantly - RAG provides context-specific information.")
    else:
        print("\n💡 The answers have moderate similarity.")


def main():
    parser = argparse.ArgumentParser(
        description="Query the RAG system and compare with non-RAG answer."
    )
    parser.add_argument("question", nargs="?", help="Your question")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve (default: 3)")
    parser.add_argument("--no-context", action="store_true", help="Hide retrieved context")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive or not args.question:
        print("Interactive Query Mode (Ctrl+C to exit)")
        print("-" * 80)
        while True:
            try:
                question = input("\n❓ Enter your question: ").strip()
                if not question:
                    print("Please enter a question.")
                    continue
                query(question, top_k=args.top_k, show_context=not args.no_context)
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    else:
        query(args.question, top_k=args.top_k, show_context=not args.no_context)


if __name__ == "__main__":
    main()
