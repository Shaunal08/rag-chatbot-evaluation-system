#!/usr/bin/env python3
"""
DAY 2 — INDEXING (improved)
Generate embeddings (local or OpenAI) and store them in ChromaDB.

Improvements vs original:
- clearer error messages for missing deps / env vars
- persistent local sentence-transformer model load
- OpenAI embedding batching with retry/backoff
- id normalization and duplicate ID detection
- optional overwrite (CHROMA_OVERWRITE=true)
- safer Chroma client creation + errors handled
- more robust retrieval test that handles empty results
"""

import os
import json
import time
import math
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

# Optional: If you want logging instead of print, swap to logging module
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from openai import OpenAI

# Vector DB
try:
    import chromadb

except Exception:
    chromadb = None
    Settings = None

# Load .env
load_dotenv()

# ---------- Config ----------
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "local").lower()
SENTENCE_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./db/chroma")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOP_K = int(os.getenv("TOP_K", 6))
BATCH_SIZE_OPENAI = int(os.getenv("OPENAI_BATCH_SIZE", 16))
BATCH_SIZE_INSERT = int(os.getenv("CHROMA_INSERT_BATCH", 256))
OVERWRITE = os.getenv("CHROMA_OVERWRITE", "false").lower() in ("1", "true", "yes")
# max retries for openai requests
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", 5))
# ---------------------------
# OpenAI v1+ client (if API key provided)
# ---------------------------
openai_client = None
try:
    if OPENAI_API_KEY:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    # openai not installed or client initialization failed; keep openai_client as None
    openai_client = None

# ---------- Load chunks ----------
chunks_path = Path("data/chunks.json")
if not chunks_path.exists():
    raise SystemExit("⚠️  data/chunks.json not found. Run Day 1 ingestion first.")

with open(chunks_path, "r", encoding="utf-8") as f:
    chunks = json.load(f)

log.info(f"Loaded {len(chunks)} chunks.")

if len(chunks) == 0:
    raise SystemExit("⚠️  No chunks found in data/chunks.json.")

# ---------- Prepare data (ids/docs/meta) ----------
ids = []
documents = []
metadatas = []
texts = []

seen_ids = set()
duplicate_count = 0

for idx, item in enumerate(chunks):
    raw_id = item.get("id") or f"chunk-{idx}"
    # normalize ID to string
    uid = str(raw_id)
    if uid in seen_ids:
        duplicate_count += 1
        # append suffix to make unique while warning user
        uid = f"{uid}-{idx}"
        log.warning(f"Duplicate id detected: {raw_id}. Renaming to {uid}")
    seen_ids.add(uid)

    ids.append(uid)
    text = item.get("text", "")
    texts.append(text)
    documents.append(text)
    metadatas.append(item.get("meta", {}))

if duplicate_count > 0:
    log.info(f"Fixed {duplicate_count} duplicate ids.")

# ---------- Embedding helpers ----------
_local_model = None  # persistent local model instance


def embed_local(texts_batch):
    global _local_model
    if SentenceTransformer is None:
        raise SystemExit("sentence-transformers not installed. Install with: pip install sentence-transformers")
    if _local_model is None:
        log.info(f"Loading local SentenceTransformer model: {SENTENCE_MODEL}")
        _local_model = SentenceTransformer(SENTENCE_MODEL)
    # model.encode returns numpy array; we convert to list-of-lists for Chroma compatibility
    embeddings = _local_model.encode(texts_batch, show_progress_bar=False, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]


def embed_openai_batch(texts_batch):
    """
    Send a batch to OpenAI embeddings endpoint with basic retry/backoff.
    Returns list of embeddings (one per input).
    """
    if openai_client is None:
        raise SystemExit("OPENAI_API_KEY missing in .env or OpenAI client not available")
    attempt = 0
    while True:
        try:
            # new v1 client API
            resp = openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts_batch
            )
            # resp.data -> list of objects with .embedding
            return [item.embedding for item in resp.data]
        except Exception as e:
            attempt += 1
            if attempt > OPENAI_MAX_RETRIES:
                raise SystemExit(f"OpenAI embedding failed after {OPENAI_MAX_RETRIES} attempts: {e}")
            sleep_for = 2 ** attempt
            log.warning(f"OpenAI request failed (attempt {attempt}/{OPENAI_MAX_RETRIES}): {e}. Retrying in {sleep_for}s...")
            time.sleep(sleep_for)


def generate_embeddings(texts_all):
    """
    Generates embeddings for all texts using appropriate mode and batching.
    Returns list of embeddings aligned with texts_all.
    """
    log.info(f"\n🔧 Generating embeddings using: {EMBEDDING_MODE}")
    embeddings = []

    if EMBEDDING_MODE == "openai":
        # process in batches
        total = len(texts_all)
        for i in tqdm(range(0, total, BATCH_SIZE_OPENAI), desc="OpenAI embed batches"):
            batch = texts_all[i:i + BATCH_SIZE_OPENAI]
            embs = embed_openai_batch(batch)
            embeddings.extend(embs)
    else:
        # local embedding in reasonably sized batches to avoid memory spikes
        total = len(texts_all)
        local_batch = 512  # safe-to-process local batch size (adjustable)
        for i in tqdm(range(0, total, local_batch), desc="Local embed batches"):
            batch = texts_all[i:i + local_batch]
            embs = embed_local(batch)
            embeddings.extend(embs)

    if len(embeddings) != len(texts_all):
        raise RuntimeError("Embedding length mismatch.")
    log.info(f"Generated {len(embeddings)} embeddings.")
    return embeddings


# ---------- Initialize Chroma (updated) ----------
import chromadb

# Create a persistent local Chroma DB (new API)
try:
    client = chromadb.PersistentClient(path=CHROMA_DIR)
except Exception as e:
    raise SystemExit(f"Failed to initialize ChromaDB PersistentClient: {e}")

collection_name = "rag_chunks"

try:
    # Optional: overwrite behavior
    if OVERWRITE and collection_name in [c.name for c in client.list_collections()]:
        log.info(f"OVERWRITE is true — deleting existing collection '{collection_name}'")
        client.delete_collection(name=collection_name)

    collection = client.get_or_create_collection(name=collection_name)
    log.info(f"Using collection: {collection_name}")
except Exception as e:
    raise SystemExit(f"Failed to get or create Chroma collection '{collection_name}': {e}")

# ---------- Generate embeddings ----------
embeddings = generate_embeddings(texts)

# ---------- Insert into Chroma ----------
log.info("\n📦 Adding embeddings to ChromaDB...")
total = len(ids)
for i in tqdm(range(0, total, BATCH_SIZE_INSERT), desc="Chroma inserts"):
    b_ids = ids[i:i + BATCH_SIZE_INSERT]
    b_docs = documents[i:i + BATCH_SIZE_INSERT]
    b_meta = metadatas[i:i + BATCH_SIZE_INSERT]
    b_embs = embeddings[i:i + BATCH_SIZE_INSERT]

    try:
        collection.add(
            ids=b_ids,
            documents=b_docs,
            metadatas=b_meta,
            embeddings=b_embs
        )
    except Exception as e:
        # provide helpful debugging info
        log.error(f"Failed to add batch starting at {i}: {e}")
        # try adding individually to surface problematic item (slow but useful for debug)
        for j, (iid, doc, meta, emb) in enumerate(zip(b_ids, b_docs, b_meta, b_embs)):
            try:
                collection.add(ids=[iid], documents=[doc], metadatas=[meta], embeddings=[emb])
            except Exception as e2:
                log.error(f"  -> Failed to add id {iid}: {e2}")
                raise

# Persist / save DB if client supports it. PersistentClient usually auto-saves.
try:
    if hasattr(client, "persist") and callable(getattr(client, "persist")):
        client.persist()
        log.info(f"\n✅ Indexing complete. Database saved to: {CHROMA_DIR}")
    else:
        # PersistentClient is typically persistent by default
        log.info(f"\n✅ Indexing complete. PersistentClient auto-saves to: {CHROMA_DIR}")
except Exception as e:
    # Non-fatal: we already added documents; just inform the user
    log.warning(f"Persist failed or not supported: {e}")

# ---------- Retrieval Test ----------
def test_retrieval(query):
    log.info(f"\n🔎 Testing retrieval for: {query}")

    if EMBEDDING_MODE == "openai":
        q_emb = embed_openai_batch([query])[0]
    else:
        q_emb = embed_local([query])[0]

    try:
        result = collection.query(
            query_embeddings=[q_emb],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        log.error(f"Query failed: {e}")
        return

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    dists = result.get("distances", [[]])[0]

    if not docs:
        log.warning("No results returned for retrieval test.")
        return

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
        log.info(f"\nRank {i+1} — Distance: {dist}")
        src = meta.get("source_file", meta.get("source", "unknown"))
        log.info("Source: %s", src)
        preview = (doc[:400] + "...") if isinstance(doc, str) and len(doc) > 400 else doc
        log.info("Chunk text preview: %s", preview)

    log.info("\n🔥 Retrieval test complete.")

if __name__ == "__main__":
    # small convenient quick test
    test_retrieval("test")
