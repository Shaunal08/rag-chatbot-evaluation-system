#!/usr/bin/env python3
"""
Simple multi-turn RAG chat wrapper.

- Conversation-aware retrieval: we use the last N messages when creating the retrieval query.
- Keeps short in-memory chat history (append-only). You can persist this yourself later.
- Reuses the retrieval and LLM functions from scripts/rag.py
"""

from __future__ import annotations
import argparse
import textwrap

# Import from your rag implementation. This will reuse the chroma client, embedding functions, and openai client.
# Make sure scripts/ is a package or run from repo root so Python can import scripts.rag
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))  # ensure scripts/ is on sys.path
import rag
retrieve = rag.retrieve
assemble_context = rag.assemble_context
call_llm_system = rag.call_llm_system
TOP_K = getattr(rag, "TOP_K", 6)

# Config: how many prior turns to include in retrieval query
CONTEXT_TURNS = 4  # number of most recent user+assistant turns to concat for retrieval
MAX_RETRIEVE_K = TOP_K

def conversation_to_query(history: list[dict]) -> str:
    """
    Build a retrieval query from recent conversation history.
    history is a list of dicts: {"role": "user"|"assistant", "text": "..."}
    We'll concatenate the last few turns into a single string.
    """
    if not history:
        return ""
    # take last CONTEXT_TURNS*2 items (role pairs) or as many as exist
    relevant = history[-CONTEXT_TURNS*2:]
    parts = []
    for turn in relevant:
        prefix = "[User]" if turn["role"] == "user" else "[Assistant]"
        parts.append(f"{prefix} {turn['text']}")
    # join with newline and truncate lightly
    return "\n".join(parts)

def chat_loop():
    print("Multi-turn RAG chat. Type 'exit' or 'quit' to stop.")
    history = []  # list of {"role":..., "text":...}
    while True:
        try:
            user = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not user:
            continue
        if user.lower() in ("exit", "quit"):
            print("Goodbye.")
            break

        # append user message to history
        history.append({"role": "user", "text": user})

        # Build retrieval query from recent conversation (conversation-aware retrieval)
        retrieval_query = conversation_to_query(history)

        # If retrieval_query is empty (first turn), fallback to just user message
        if not retrieval_query:
            retrieval_query = user

        # Retrieve top-k chunks
        retrieved = retrieve(retrieval_query, top_k=MAX_RETRIEVE_K)

        # Assemble context text to include in prompt
        context_text = assemble_context(retrieved)

        # For better grounding, include the chat history in the final prompt that goes to the LLM.
        # We'll build a single prompt string that contains:
        #  - a short system instruction
        #  - recent chat history
        #  - retrieved context
        #  - the current user question
        chat_history_text = ""
        # include all history or only recent turns for length safety
        recent_history = history[-(CONTEXT_TURNS*2):]
        for turn in recent_history:
            role = "User" if turn["role"] == "user" else "Assistant"
            chat_history_text += f"{role}: {turn['text']}\n"

        combined_prompt = (
            "Chat history:\n"
            f"{chat_history_text}\n"
            "Retrieved context:\n"
            f"{context_text}\n\n"
            f"Current question: {user}\n\n"
            "Answer the current question based only on the retrieved context and the chat history. "
            "Cite sources like [Source: filename | id] for any factual claims."
        )

        # Call LLM (reuse call_llm_system, which expects user_question and context_text)
        # We pass user and combined_prompt as context_text so the LLM sees both history and retrieval context.
        try:
            assistant_answer = call_llm_system(user, combined_prompt)
        except Exception as e:
            print(f"Error calling LLM: {e}")
            break

        # print and append assistant answer
        print("\nAssistant:\n")
        print(textwrap.fill(assistant_answer, width=100))
        history.append({"role": "assistant", "text": assistant_answer})

if __name__ == "__main__":
    chat_loop()