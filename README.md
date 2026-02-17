# Mini-RAG (Database Systems)

This repo implements the **data prep + chunking + embeddings + retrieval + evaluation** portion of a mini-RAG pipeline for the **Database Systems** domain.

## What is included

- A small database systems PDF knowledge base in `data/`.
- A script that extracts text, chunks it, builds MiniLM embeddings, and writes a FAISS index.
- A retrieval script that returns top-k chunks for a query.
- A simple RAG vs non-RAG comparison and evaluation (ROUGE + semantic similarity).

## What is not yet included

- Generator model integration (LLaMA-3-7B).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
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
