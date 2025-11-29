#!/usr/bin/env python3

"""
DAY 1 — INGESTION & CHUNKING
Loads PDFs / text files, extracts text, chunks it using token-based logic,
and saves all chunks into data/chunks.json with metadata.
"""

import os
import json
import pdfplumber
from pathlib import Path
from tqdm import tqdm
import tiktoken


# ---------------------------
# Tokenizer setup
# ---------------------------
try:
    enc = tiktoken.get_encoding("cl100k_base")
except:
    enc = None


def count_tokens(text: str) -> int:
    if enc:
        return len(enc.encode(text))
    return len(text.split())  # fallback


def chunk_text(text, chunk_size=400, overlap=120):
    """
    Chunk text into ~800-token blocks with 150-token overlap.
    """
    if not text or not text.strip():
        return []

    if enc:
        tokens = enc.encode(text)
        chunks = []
        i = 0
        stride = chunk_size - overlap

        while i < len(tokens):
            chunk_tokens = tokens[i:i + chunk_size]
            chunks.append(enc.decode(chunk_tokens))
            i += stride

        return chunks

    else:  # basic fallback
        words = text.split()
        chunks = []
        i = 0
        stride = chunk_size - overlap
        while i < len(words):
            chunks.append(" ".join(words[i:i + chunk_size]))
            i += stride
        return chunks


# ---------------------------
# Document Loaders
# ---------------------------
def load_pdf(path: Path):
    """
    Extract text page-by-page from a PDF.
    """
    with pdfplumber.open(path) as pdf:
        pages = []
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"page": i, "text": text})
        return pages


def load_text_file(path: Path):
    """
    For .txt / .md
    """
    with open(path, "r", encoding="utf-8") as f:
        return [{"page": 1, "text": f.read()}]


# ---------------------------
# Ingestion Pipeline
# ---------------------------
def ingest_documents(input_dir="data/docs", output_file="data/chunks.json"):
    input_dir = Path(input_dir)
    output_file = Path(output_file)

    all_chunks = []
    supported = [".pdf", ".txt", ".md"]

    files = [f for f in input_dir.rglob("*") if f.suffix.lower() in supported]

    print(f"Found {len(files)} documents.")

    for fpath in tqdm(files):
        ext = fpath.suffix.lower()

        # load text
        if ext == ".pdf":
            pages = load_pdf(fpath)
        else:
            pages = load_text_file(fpath)

        # chunk pages
        for page in pages:
            chunks = chunk_text(page["text"])
            for i, ch in enumerate(chunks):
                chunk_id = f"{fpath.stem}_p{page['page']}_c{i}"

                all_chunks.append({
                    "id": chunk_id,
                    "text": ch,
                    "meta": {
                        "source_file": fpath.name,
                        "page": page["page"],
                        "chunk_index": i,
                        "path": str(fpath.resolve())
                    }
                })

    # save JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_chunks)} chunks to {output_file}")


# ---------------------------
# Run script
# ---------------------------
if __name__ == "__main__":
    ingest_documents()
