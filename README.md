# RAG System (Database Systems)

This repo implements a complete **RAG (Retrieval-Augmented Generation)** pipeline for the **Database Systems** domain using OpenRouter with **meta-llama/llama-3.1-70b-instruct**.

## What is included

- A small database systems PDF knowledge base in `data/`.
- A script that extracts text, chunks it, builds MiniLM embeddings, and writes a FAISS index.
- A retrieval script that returns top-k chunks for a query.
- **Full LLM integration** using OpenRouter API for answer generation.
- RAG vs non-RAG comparison and evaluation (ROUGE + semantic similarity).

## Features

- ✅ Vector-based retrieval using FAISS and sentence-transformers
- ✅ LLM-powered answer generation using OpenRouter
- ✅ RAG (with context) vs. non-RAG (without context) comparison
- ✅ Automatic evaluation with ROUGE metrics

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API key

1. Get your OpenRouter API key from [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```
   OPENROUTER_API_KEY=your_actual_api_key_here
   ```

## Build the index

```bash
python src\build_index.py
```

## Retrieve top-k chunks

```bash
python src\retrieve.py "What does ACID stand for?" --top-k 3
```

## RAG vs non-RAG comparison

This generates answers using both RAG (with retrieved context) and non-RAG (LLM only) approaches:

```bash
python src\rag_compare.py
```

## Evaluate answers

```bash
python src\evaluate.py
```

Outputs are written to `artifacts/`:

- `index.faiss`
- `metadata.json`

## Next steps (planned)

- Add a generator (LLaMA-3-7B or FLAN-T5).
