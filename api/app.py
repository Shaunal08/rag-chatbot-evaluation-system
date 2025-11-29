#!/usr/bin/env python3
# api/app.py
"""
FastAPI backend for the RAG chatbot.

Endpoints:
- POST /query      -> single-turn query (no session)
- POST /chat       -> multi-turn chat using session_id (session stored on disk)
- GET  /sessions   -> list saved session IDs
"""

from __future__ import annotations
from pathlib import Path
import sys
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Make scripts/ importable (project_root/scripts)
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

# Import your rag module (scripts/rag.py)
try:
    import rag  # must provide retrieve, assemble_context, call_llm_system, TOP_K, answer_query
except Exception as e:
    raise RuntimeError(f"Failed to import scripts/rag.py: {e}")

# Sessions directory
SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

def session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"

def load_session(session_id: str) -> List[Dict[str,str]]:
    p = session_path(session_id)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))

def append_session(session_id: str, role: str, text: str) -> None:
    hist = load_session(session_id)
    hist.append({"role": role, "text": text})
    session_path(session_id).write_text(json.dumps(hist, ensure_ascii=False, indent=2), encoding="utf-8")

def list_sessions() -> List[str]:
    return [p.stem for p in SESSIONS_DIR.glob("*.json")]

# FastAPI app
app = FastAPI(title="RAG Chat API")

# Request / Response schemas
class QueryRequest(BaseModel):
    question: str
    topk: Optional[int] = None

class ChatRequest(BaseModel):
    session_id: str
    question: str
    topk: Optional[int] = None

class AnswerResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    retrieved: List[Dict[str, Any]]

# Helper: build retrieval query from history (conversation-aware)
def conversation_to_query(history: List[Dict[str,str]], max_turns: int = 4) -> str:
    if not history:
        return ""
    relevant = history[-(max_turns*2):]
    parts = []
    for turn in relevant:
        prefix = "[User]" if turn["role"] == "user" else "[Assistant]"
        parts.append(f"{prefix} {turn['text']}")
    return "\n".join(parts)

# Single-turn endpoint (no session storage)
@app.post("/query", response_model=AnswerResponse)
def query_endpoint(req: QueryRequest):
    topk = req.topk or getattr(rag, "TOP_K", 6)
    try:
        result = rag.answer_query(req.question, top_k=topk)
    except Exception as e:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
    return AnswerResponse(answer=result["answer"], sources=result["sources"], retrieved=result["retrieved"])

# Multi-turn chat endpoint (session-aware)
@app.post("/chat", response_model=AnswerResponse)
def chat_endpoint(req: ChatRequest):
    session_id = req.session_id
    topk = req.topk or getattr(rag, "TOP_K", 6)

    # load history and persist user turn
    history = load_session(session_id)
    append_session(session_id, "user", req.question)

    # build retrieval query from history + current question
    retrieval_query = conversation_to_query(history + [{"role":"user","text":req.question}])
    if not retrieval_query:
        retrieval_query = req.question

    try:
        # use rag.retrieve + assemble_context + call_llm_system
        retrieved = rag.retrieve(retrieval_query, top_k=topk)
        context = rag.assemble_context(retrieved)

        # include recent chat history in prompt for better context (recent 8 turns)
        recent_history = (history + [{"role":"user","text":req.question}])[-8:]
        chat_history_text = "\n".join(
            (f"User: {t['text']}" if t['role']=="user" else f"Assistant: {t['text']}") for t in recent_history
        )
        combined_context = f"Chat history:\n{chat_history_text}\n\nRetrieved context:\n{context}"

        answer = rag.call_llm_system(req.question, combined_context)

        # persist assistant reply
        append_session(session_id, "assistant", answer)

        sources = [{"id": r["id"], "source": r["meta"].get("source_file") or r["meta"].get("source")} for r in retrieved]
        return AnswerResponse(answer=answer, sources=sources, retrieved=retrieved)
    except Exception as e:
        log.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
def get_sessions():
    return {"sessions": list_sessions()}
