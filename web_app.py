from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import re

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.responses import FileResponse, HTMLResponse
from openai import OpenAI
from rouge_score import rouge_scorer


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
WEB_DIR = ROOT_DIR / "web"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_env() -> None:
    load_dotenv(dotenv_path=ROOT_DIR / ".env")


def _get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENROUTER_API_KEY not set. Add it to .env (repo root) or your environment.",
        )
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _import_retrieve():
    # Keep existing src scripts working (they expect src on sys.path).
    import sys

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    from retrieve import retrieve  # type: ignore

    return retrieve


MODEL_NAME = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")


def _build_chat_history_messages(chat: dict[str, Any] | None, limit: int = 6) -> list[dict[str, str]]:
    """Build recent chat history as LLM messages for contextualised RAG."""
    if not chat:
        return []
    messages = chat.get("messages", [])
    history: list[dict[str, str]] = []
    # Take last `limit` messages for context
    recent = messages[-limit:] if len(messages) > limit else messages
    for m in recent:
        role = m.get("role", "user")
        content = m.get("rag_answer") or m.get("content") or ""
        if role == "assistant":
            content = m.get("rag_answer") or m.get("content") or ""
        else:
            content = m.get("content") or ""
        history.append({"role": role, "content": content})
    return history


def _generate_rag_answer(question: str, chunks: list[dict[str, Any]], chat: dict[str, Any] | None = None) -> str:
    if not chunks:
        return "No relevant context found in the local index. Try rebuilding the index or rephrasing your question."

    context = "\n\n".join(
        [
            f"[Source: {c.get('source','')}, Chunk {c.get('chunk','')}]\n{c.get('text','')}"
            for c in chunks
        ]
    )

    prompt = f"""Use only the context below to answer the question.
Write a complete and specific answer.
If the context is insufficient, say what is missing instead of guessing.

Context:
{context}

Question: {question}

Answer:"""

    llm_messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "You are a grounded RAG assistant. Use ONLY the provided context. "
                       "You may use the conversation history to understand the user's intent, "
                       "but your answer must be grounded in the retrieved context. Be direct and factual.",
        },
    ]
    # Add chat history for contextualisation
    llm_messages.extend(_build_chat_history_messages(chat))
    llm_messages.append({"role": "user", "content": prompt})

    client = _get_openrouter_client()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=llm_messages,
            temperature=0.2,
            max_tokens=900,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")


def _generate_non_rag_answer(question: str, chat: dict[str, Any] | None = None) -> str:
    """Generate an answer using only the LLM's parametric knowledge (no retrieval context)."""
    llm_messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "You are a knowledgeable assistant specialising in Indian history. "
                       "Answer the user's question using your own knowledge. Be direct and factual.",
        },
    ]
    llm_messages.append({"role": "user", "content": question})

    client = _get_openrouter_client()
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=llm_messages,
            temperature=0.2,
            max_tokens=900,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")


_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def _tokenize(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-zA-Z]+", text.lower()) if len(tok) > 2}


def _truncate_text(text: str, max_chars: int = 15000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _calculate_metrics(answer: str, chunks: list[dict[str, Any]]) -> dict[str, float] | None:
    if not answer:
        return None

    context_text = "\n\n".join([str(c.get("text", "")) for c in (chunks or []) if c])
    context_text = _truncate_text(context_text)
    if not context_text.strip():
        return None

    rouge_l = float(_ROUGE_SCORER.score(context_text, answer)["rougeL"].fmeasure)

    semantic_scores = [
        float(c.get("semantic_score"))
        for c in (chunks or [])
        if c is not None and c.get("semantic_score") is not None
    ]
    semantic_score = float(sum(semantic_scores) / len(semantic_scores)) if semantic_scores else 0.0

    answer_tokens = _tokenize(answer)
    context_tokens = _tokenize(context_text)
    if not answer_tokens:
        hallucination_ratio = 0.0
    else:
        supported = len(answer_tokens & context_tokens)
        hallucination_ratio = float((len(answer_tokens) - supported) / len(answer_tokens))

    return {
        "rouge_l": rouge_l,
        "semantic_score": semantic_score,
        "hallucination_ratio": hallucination_ratio,
    }


CHAT_STORE_PATH = ARTIFACTS_DIR / "chats.json"


def _read_chat_store() -> list[dict[str, Any]]:
    if not CHAT_STORE_PATH.exists():
        return []
    try:
        return json.loads(CHAT_STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_chat_store(chats: list[dict[str, Any]]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHAT_STORE_PATH.write_text(json.dumps(chats, indent=2), encoding="utf-8")


def _find_chat(chats: list[dict[str, Any]], chat_id: str) -> dict[str, Any] | None:
    for c in chats:
        if c.get("id") == chat_id:
            return c
    return None


_load_env()

app = FastAPI(title="ANLP_Rag Web")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="Web UI not found. Expected web/index.html")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/assets/{path}")
def assets(path: str):
    file_path = (WEB_DIR / path).resolve()
    if WEB_DIR.resolve() not in file_path.parents:
        raise HTTPException(status_code=404, detail="Not found")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(str(file_path))


@app.get("/api/chats")
def list_chats() -> dict[str, Any]:
    chats = _read_chat_store()
    summaries = []
    for c in chats:
        messages = c.get("messages", [])
        first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        summaries.append(
            {
                "id": c.get("id"),
                "created_at": c.get("created_at"),
                "title": (first_user[:60] + "…") if len(first_user) > 60 else first_user,
            }
        )
    summaries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"chats": summaries}


@app.get("/api/chats/{chat_id}")
def get_chat(chat_id: str) -> dict[str, Any]:
    chats = _read_chat_store()
    chat = _find_chat(chats, chat_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return {"chat": chat}


@app.delete("/api/chats/{chat_id}")
def delete_chat(chat_id: str) -> dict[str, Any]:
    chats = _read_chat_store()
    remaining = [c for c in chats if c.get("id") != chat_id]
    if len(remaining) == len(chats):
        raise HTTPException(status_code=404, detail="Chat not found")

    _write_chat_store(remaining)
    return {"ok": True, "deleted": chat_id}


@app.post("/api/ask")
def ask(payload: dict[str, Any]) -> dict[str, Any]:
    question = str(payload.get("question", "")).strip()
    chat_id = str(payload.get("chat_id", "")).strip() or None
    top_k = int(payload.get("top_k", 10))

    if not question:
        raise HTTPException(status_code=400, detail="Question is required")

    retrieve = _import_retrieve()

    try:
        chunks = retrieve(question, top_k=top_k)
    except SystemExit as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    # Resolve or create the chat first so RAG can use history
    chats = _read_chat_store()
    if chat_id:
        chat = _find_chat(chats, chat_id)
        if not chat:
            chat = {"id": str(uuid.uuid4()), "created_at": _utc_now_iso(), "messages": []}
            chats.append(chat)
    else:
        chat = {"id": str(uuid.uuid4()), "created_at": _utc_now_iso(), "messages": []}
        chats.append(chat)

    # Generate both RAG (contextualised) and non-RAG answers
    rag_answer = _generate_rag_answer(question, chunks, chat)
    non_rag_answer = _generate_non_rag_answer(question, chat)
    rag_metrics = _calculate_metrics(rag_answer, chunks)

    chat["updated_at"] = _utc_now_iso()
    chat["messages"].append({"role": "user", "content": question, "ts": _utc_now_iso()})
    chat["messages"].append(
        {
            "role": "assistant",
            "rag_answer": rag_answer,
            "non_rag_answer": non_rag_answer,
            "content": rag_answer,  # backward compat
            "ts": _utc_now_iso(),
            "rag_metrics": rag_metrics,
            "metrics": rag_metrics,  # backward compat
        }
    )

    _write_chat_store(chats)

    return {
        "chat_id": chat["id"],
        "rag_answer": rag_answer,
        "non_rag_answer": non_rag_answer,
        "question": question,
        "rag_metrics": rag_metrics,
    }
