# RAG System (Database Systems)

This repo implements a complete **RAG (Retrieval-Augmented Generation)** pipeline for the **Database Systems** domain using OpenRouter with **meta-llama/llama-3.1-70b-instruct**.

## Prerequisites

Before running this project, you need:

1. **Python 3.8+** installed on your system
2. **OpenRouter API Key** - Get it from [https://openrouter.ai/keys](https://openrouter.ai/keys)
3. **PDF Dataset** - Place your database systems PDFs in the `data/` folder
4. **Internet connection** - For downloading embeddings model and API calls

## What is included

- A small database systems PDF knowledge base in `data/`.
- A script that extracts text, chunks it, builds MiniLM embeddings, and writes a FAISS index.
- A retrieval script that returns top-k chunks for a query.
- **Full LLM integration** using OpenRouter API for answer generation.
- RAG vs non-RAG comparison and evaluation (ROUGE + semantic similarity).

## Features

- ✅ Vector-based retrieval using FAISS and sentence-transformers
- ✅ LLM-powered answer generation using OpenRouter with **meta-llama/llama-3.1-70b-instruct**
- ✅ RAG (with context) vs. non-RAG (without context) comparison
- ✅ Automatic evaluation with ROUGE-L and semantic similarity metrics
- ✅ Interactive query interface

## Setup

### 1. Clone and navigate to the project

```bash
cd ANLP_Rag
```

### 2. Add your dataset

Place your database systems PDF files in the `data/` folder. The system will:
- Extract text from PDFs
- Chunk them for efficient retrieval
- Build a vector index using FAISS

Example PDF: `data/db_intro.pdf` (already included)

### 3. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies include:**
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Text embeddings (all-MiniLM-L6-v2)
- `openai` - OpenRouter API client
- `python-dotenv` - Environment variable management
- `P🚀 Quick Query (Recommended)

**Short PowerShell command:**

```powershell
.\query.ps1 "Your question here"
```

**Examples:**
```powershell
# Ask about ACID
.\query.ps1 "What is ACID?"

# Ask about normalization
.\query.ps1 "What is normalization?"

# Interactive mode
.\query.ps1 -Interactive

# Get top 5 chunks
.\query.ps1 "How do indexes work?" -TopK 5

# Hide retrieved context
.\query.ps1 "What is ACID?" -NoContext
```

**What you get:**
- 🔍 Retrieved context from your knowledge base
- ✅ RAG answer (with context) using **meta-llama/llama-3.1-70b-instruct**
- ❌ Non-RAG answer (without context)
- 📊 **ROUGE-L Score** - Measures word overlap (0=no match, 1=identical)
- 📊 **Semantic Similarity** - Measures meaning similarity (0=different, 1=same)

**Example Output:**

```
================================================================================
QUESTION: What is normalization?
================================================================================

🔍 Retrieving top-3 relevant chunks...

📚 RETRIEVED CONTEXT:
  [1] Score: 0.466 | db_intro.pdf (Chunk 0)
      Database systems overview... Normalization reduces redundancy and anomalies...

🤖 Generating RAG answer (with context)...
🤖 Generating non-RAG answer (without context)...

================================================================================
✅ RAG ANSWER (with context):
================================================================================
Normalization reduces redundancy and anomalies in database design.

================================================================================
❌ NON-RAG ANSWER (without context):
================================================================================
Normalization is a process in data preparation that involves scaling numeric 
data to a common range, usually between 0 and 1...

================================================================================
📊 COMPARISON METRICS:
================================================================================
  ROUGE-L Score:         0.0211
  Semantic Similarity:   0.5278

  Interpretation:
    - ROUGE-L measures word overlap (0=no overlap, 1=identical)
    - Semantic Similarity measures meaning similarity (0=different, 1=same)
================================================================================

💡 The answers differ significantly - RAG provides context-specific information.
```

**Why this matters:** In this example, without context, the LLM talks about data scaling 
(ML normalization), but with context, it correctly explains database normalization. This 
demonstrates the power of RAG in providing domain-specific answers!

---

### Alternative: Python command

```bash
# Single question
python src\query.py "What is ACID?"

# Interactive mode (ask multiple questions)
python src\query.py --interactive
```

**Options:**
- `--top-k N`: Retrieve top N chunks (default: 3)
- `--no-context`: Hide retrieved context in output
- `--interactive` or `-i`: Interactive mode for multiple questions

---

### Other Commands

**Retrieve top-k chunks only:**

```bash
python src\retrieve.py "What does ACID stand for?" --top-k 3
```

**Batch RAG vs non-RAG comparison:**

This generates answers using both RAG (with retrieved context) and non-RAG (LLM only) approaches for all questions in `data/questions.json`:

```bash
python src\rag_compare.py
```

**Evaluate answers:**

```bash
python src\evaluate.py
```

**Run complete pipeline:**

```bash
.\run_all.ps1
```

## Outputs

All outputs are written to `artifacts/`:

- `index.faiss` - FAISS vector index
- `metadata.json` - Chunk metadata
- `answers.json` - Generated answers (from rag_compare.py)
- `metrics.json` - Evaluation metrics (from evaluate.py)
