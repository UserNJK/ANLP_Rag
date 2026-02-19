# ANLP_Rag — Retrieval-Augmented Generation (RAG)

This repo implements a small end-to-end **RAG (Retrieval-Augmented Generation)** pipeline:

1. Build a FAISS index from a local **JSONL chunk corpus** in `data/`
2. Retrieve top-k chunks for a question
3. Generate answers using **OpenRouter** (LLM) with and without retrieved context
4. (Optional) Run a batch comparison + evaluation (ROUGE-L + semantic similarity)

The PowerShell scripts are aimed at Windows, but you can also run the Python entrypoints directly.

## Requirements

- Python **3.9+**
- An OpenRouter API key: https://openrouter.ai/keys
- Internet access (downloads the embedding model, calls OpenRouter)

## Dataset format (JSONL)

Put one or more `*.jsonl` files anywhere under `data/` (for example `data/DB/`). Each line must be a JSON object. The indexer expects (at minimum) a `content` field.

### Dataset source (this project)

- Kaggle: [IndHist-RAG: Indian History Retrieval Corpus](https://www.kaggle.com/datasets/kanav608/input-data)
- Description: OCR-cleaned, structured Indian history corpus for RAG / retrieval systems
- Typical size (as listed on Kaggle): ~167 MB
- File format: JSON Lines (`.jsonl`)
- File count: ~112–114 files (varies by version)
- Chunk count: ~124k passage-level records

This repo expects these files under `data/DB/` (or any folder inside `data/`).

Common metadata fields in records include:

- `chunk_id`
- `series_id` / `series_title`
- `volume_title`
- `chapter_title`
- `page`
- `content`

Note: Kaggle currently shows license information as unknown on the dataset page. Verify usage rights before redistribution.

Expected fields (used by `src/build_index.py`):

```json
{
  "chunk_id": "Book_Chapter_0001",
  "book_title": "Some Book",
  "chapter_title": "Chapter Name",
  "content": "Text content for this chunk..."
}
```

Notes:

- Empty / low-quality chunks are skipped automatically.
- If CUDA is available, embeddings will run on GPU automatically.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the repo root:

```env
OPENROUTER_API_KEY=your_key_here
```

## Build the index

```powershell
python src\build_index.py
```

Outputs (written to `artifacts/`):

- `index.faiss` — FAISS vector index
- `metadata.json` — chunk metadata + text

## Query (recommended)

```powershell
.\query.ps1 "Your question here"
```

Examples:

```powershell
# Interactive mode
.\query.ps1 -Interactive

# Retrieve 5 chunks
.\query.ps1 "Who was Ashoka?" -TopK 5

# Hide retrieved context printing
.\query.ps1 "What is the Mauryan Empire?" -NoContext
```

The query tool prints:

- Retrieved context (optional)
- RAG answer (with context)
- Non-RAG answer (without context)
- Comparison metrics: ROUGE-L and semantic similarity

## Web UI

This runs the same RAG answering flow in a simple local webpage (no login).

```powershell
pip install -r requirements.txt
python -m uvicorn web_app:app --reload --port 8000
```

Open:

- http://127.0.0.1:8000

### Alternative: run Python directly

```powershell
python src\query.py "Who was Akbar?" --top-k 10
python src\query.py --interactive
```

Options:

- `--top-k N` (default: 10)
- `--no-context`
- `--interactive` / `-i`

## Retrieve only (no LLM)

```powershell
python src\retrieve.py "Vijayanagar empire" --top-k 5
```

## Batch compare + evaluation

1) Edit `data/questions.json` to contain your evaluation set:

```json
[
  {"question": "...", "answer": "..."}
]
```

2) Run:

```powershell
python src\rag_compare.py
python src\evaluate.py
```

Outputs (written to `artifacts/`):

- `answers.json` — for each question: reference, rag answer, non-rag answer, retrieved contexts
- `metrics.json` — per-question and average ROUGE-L + semantic similarity

## Run the full pipeline

```powershell
.\run_all.ps1
```

## Models used

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- LLM (OpenRouter): `meta-llama/llama-3.1-70b-instruct`

## Troubleshooting

- Missing index: run `python src\build_index.py` first.
- Missing key: set `OPENROUTER_API_KEY` in `.env` or your environment.
