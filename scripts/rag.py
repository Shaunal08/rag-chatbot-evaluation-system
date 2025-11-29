#!/usr/bin/env python3
"""
scripts/rag.py
Day 3 — Retrieval + RAG (OpenAI v1+ and chromadb.PersistentClient compatible)

Usage:
    python scripts/rag.py -q "Your question here"

Requirements:
- chromadb >= 1.3.x (uses PersistentClient)
- openai >= 1.0.0 (new OpenAI client)
- sentence-transformers (optional, for local embeddings)
"""

from __future__ import annotations
import os
import argparse
import logging
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------
# Config (env overrides)
# ---------------------------
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./db/chroma")
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local").lower()  # "local" or "openai"
SENTENCE_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("TOP_K", 6))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", 3000))

# ---------------------------
# Optional imports (lazy)
# ---------------------------
_local_model = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# New OpenAI client (v1+)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Chroma
try:
    import chromadb
except Exception:
    chromadb = None

# ---------------------------
# Validate essentials
# ---------------------------
if chromadb is None:
    raise SystemExit("chromadb not installed. Install with: pip install chromadb")

# create chroma persistent client
try:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
except Exception as e:
    raise SystemExit(f"Failed to initialize chromadb.PersistentClient: {e}")

# get or create collection
COLLECTION_NAME = "rag_chunks"
try:
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
except Exception as e:
    raise SystemExit(f"Failed to open/create collection '{COLLECTION_NAME}': {e}")

# instantiate OpenAI client if key present
openai_client = None
if OPENAI_API_KEY:
    if OpenAI is None:
        raise SystemExit("OpenAI SDK not installed. Install with: pip install openai")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------
# Embedding helpers
# ---------------------------
def embed_local(texts: List[str]) -> List[List[float]]:
    """Local embeddings using sentence-transformers."""
    global _local_model
    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers not installed. Install with: pip install sentence-transformers")
    if _local_model is None:
        log.info(f"Loading local SentenceTransformer model: {SENTENCE_MODEL}")
        _local_model = SentenceTransformer(SENTENCE_MODEL)
    embs = _local_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [e.tolist() for e in embs]

def embed_openai(texts: List[str]) -> List[List[float]]:
    """OpenAI embeddings using the new OpenAI client."""
    if openai_client is None:
        raise SystemExit("OPENAI_API_KEY required for OpenAI embeddings. Set OPENAI_API_KEY in env or .env")
    resp = openai_client.embeddings.create(model=OPENAI_EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def get_query_embedding(query: str) -> List[float]:
    if EMBEDDING_MODE == "openai":
        return embed_openai([query])[0]
    else:
        return embed_local([query])[0]

# ---------------------------
# Retrieval
# ---------------------------
def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Return list of retrieved items: {id, doc, meta, distance}"""
    q_emb = get_query_embedding(query)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]  # do NOT include "ids" here
    )
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0]  # ids are present even if not in include
    results = []
    for doc, meta, dist, _id in zip(docs, metas, dists, ids):
        results.append({"id": _id, "doc": doc, "meta": meta, "distance": dist})
    return results

# ---------------------------
# Assemble context for LLM
# ---------------------------
def assemble_context(retrieved: List[Dict[str, Any]]) -> str:
    parts = []
    total = 0
    for item in retrieved:
        src = item["meta"].get("source_file") or item["meta"].get("source") or "unknown"
        header = f"Source: {src} | id: {item['id']} | distance: {item['distance']:.4f}\n"
        text = item["doc"] or ""
        block = header + text + "\n---\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total
            if remaining <= 0:
                break
            parts.append(block[:remaining])
            total += remaining
            break
        parts.append(block)
        total += len(block)
    return "\n".join(parts)

# ---------------------------
# LLM call (new OpenAI API)
# ---------------------------
def call_llm_system(user_question: str, context_text: str) -> str:
    if openai_client is None:
        raise SystemExit("OPENAI_API_KEY required to call the LLM. Set OPENAI_API_KEY in env or .env")
    system_prompt = (
        "You are a helpful assistant that answers user questions using only the provided context. "
        "Cite sources by filename and id like [Source: filename | id]. If the answer is not supported by the context, say you don't know."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {user_question}\n\nAnswer concisely and include citations."
    resp = openai_client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=800,
        temperature=0.0
    )
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        d = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
        return d["choices"][0]["message"]["content"].strip()

# ---------------------------
# High-level pipeline
# ---------------------------
def answer_query(question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    retrieved = retrieve(question, top_k=top_k)
    if not retrieved:
        return {"answer": "No relevant documents found.", "sources": [], "retrieved": []}
    context = assemble_context(retrieved)
    answer = call_llm_system(question, context)
    sources = [{"id": r["id"], "source": r["meta"].get("source_file") or r["meta"].get("source")} for r in retrieved]
    return {"answer": answer, "sources": sources, "retrieved": retrieved}

# ---------------------------
# CLI
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG query CLI")
    parser.add_argument("--question", "-q", required=True, help="Question to answer")
    parser.add_argument("--topk", "-k", type=int, default=TOP_K, help="Number of documents to retrieve")
    args = parser.parse_args()

    out = answer_query(args.question, top_k=args.topk)
    print("\n--- ANSWER ---\n")
    print(out["answer"])
    print("\n--- SOURCES ---\n")
    for s in out["sources"]:
        print(s)

if __name__ == "__main__":
    main()