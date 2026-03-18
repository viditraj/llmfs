"""
ollama_demo.py — Scripted LLMFS demo using a local Ollama model.

Runs a fixed sequence of interactions that prove LLMFS is working:
  1. Agent writes several memories via tool calls
  2. Agent searches and retrieves them
  3. Agent updates a memory and verifies the change
  4. Agent reads back across a fresh MemoryFS instance (persistence check)

Requirements:
  pip install openai          # Ollama speaks the OpenAI-compatible API
  ollama pull llama3.2        # or any model that supports tool calls

Run:
  python examples/ollama_demo.py
  python examples/ollama_demo.py --model mistral
  python examples/ollama_demo.py --base-url http://192.168.1.10:11434/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

# ── Dependency check ──────────────────────────────────────────────────────────
try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.\n  pip install openai")
    sys.exit(1)

from llmfs import MemoryFS
from llmfs.integrations.openai_tools import LLMFS_TOOLS, LLMFSToolHandler


# ── Helpers ───────────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def _run_tool_loop(
    client: OpenAI,
    model: str,
    handler: LLMFSToolHandler,
    messages: list[dict],
    max_rounds: int = 6,
) -> str:
    """Send messages to the model and keep executing tool calls until done."""
    for round_num in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=LLMFS_TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message

        # Append assistant turn
        messages.append(msg.model_dump(exclude_none=True))

        if choice.finish_reason != "tool_calls" or not msg.tool_calls:
            return msg.content or ""

        # Execute every tool call the model requested
        print(f"\n  [round {round_num + 1}] model called {len(msg.tool_calls)} tool(s):")
        tool_results = handler.handle_batch(list(msg.tool_calls))

        for tc, result in zip(msg.tool_calls, tool_results):
            print(f"    • {tc.function.name}({tc.function.arguments[:60]}…)")
            print(f"      → {result[:80]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "(max rounds reached)"


# ── Main demo ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="LLMFS + Ollama scripted demo")
    parser.add_argument("--model", default="llama3.2",
                        help="Ollama model name (default: llama3.2)")
    parser.add_argument("--base-url", default="http://localhost:11434/v1",
                        help="Ollama OpenAI-compatible base URL")
    parser.add_argument("--store", default=None,
                        help="Path to persist memories (default: temp dir)")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="ollama")

    # Verify Ollama is reachable
    try:
        client.models.list()
    except Exception as exc:
        print(f"ERROR: Cannot reach Ollama at {args.base_url}\n  {exc}")
        print("\nMake sure Ollama is running:  ollama serve")
        sys.exit(1)

    use_tmp = args.store is None
    store_path = args.store or tempfile.mkdtemp(prefix="llmfs_ollama_")
    print(f"Memory store: {store_path}")
    print(f"Model:        {args.model}")

    mem = MemoryFS(path=store_path)
    handler = LLMFSToolHandler(mem)

    system = (
        "You are a helpful assistant with persistent memory via LLMFS tools. "
        "Use memory_write to store important facts, memory_search to retrieve "
        "relevant context, and memory_read to fetch specific memories by path. "
        "Always store key facts before answering so they persist for future sessions. "
        "IMPORTANT: Before calling memory_update, always call memory_search first "
        "to find the exact path of the memory you want to update."
    )

    # ── Step 1: Store some knowledge ─────────────────────────────────────────
    _header("Step 1 — Ask the model to remember some facts")
    prompt1 = (
        "Please remember the following project facts using your memory tools:\n"
        "1. Project Nexus uses PostgreSQL 15, Redis 7, Python 3.11\n"
        "2. The production server IP is 10.0.1.50 (do NOT store real IPs in prod!)\n"
        "3. The team standup is every day at 09:00 UTC\n"
        "Store each fact as a separate memory under /nexus/, then confirm you stored them."
    )
    print(f"\nUser: {prompt1}")
    messages1 = [{"role": "system", "content": system}, {"role": "user", "content": prompt1}]
    reply1 = _run_tool_loop(client, args.model, handler, messages1)
    print(f"\nAssistant: {reply1}")

    # ── Step 2: Retrieve without hints ───────────────────────────────────────
    _header("Step 2 — Ask about stored facts (no hints given)")
    prompt2 = "What database does Project Nexus use, and when is the standup?"
    print(f"\nUser: {prompt2}")
    messages2 = [{"role": "system", "content": system}, {"role": "user", "content": prompt2}]
    reply2 = _run_tool_loop(client, args.model, handler, messages2)
    print(f"\nAssistant: {reply2}")

    # ── Step 3: Update a memory ───────────────────────────────────────────────
    _header("Step 3 — Update a stored fact")
    prompt3 = (
        "The standup time changed to 10:00 UTC. "
        "Please update the memory that stores the standup time to reflect this."
    )
    print(f"\nUser: {prompt3}")
    messages3 = [{"role": "system", "content": system}, {"role": "user", "content": prompt3}]
    reply3 = _run_tool_loop(client, args.model, handler, messages3)
    print(f"\nAssistant: {reply3}")

    # ── Step 4: Persistence check (new MemoryFS instance, same store) ─────────
    _header("Step 4 — Persistence check (fresh MemoryFS instance, same store)")
    mem2 = MemoryFS(path=store_path)
    handler2 = LLMFSToolHandler(mem2)
    prompt4 = "What time is the Project Nexus standup? Search your memory."
    print(f"\nUser: {prompt4}")
    messages4 = [{"role": "system", "content": system}, {"role": "user", "content": prompt4}]
    reply4 = _run_tool_loop(client, args.model, handler2, messages4)
    print(f"\nAssistant: {reply4}")

    # ── Step 5: Direct verification ───────────────────────────────────────────
    _header("Step 5 — Direct LLMFS verification (no LLM)")
    all_memories = mem2.list("/nexus/")
    print(f"\nMemories stored under /nexus/: {len(all_memories)}")
    for obj in all_memories:
        print(f"  {obj.path}")
        print(f"    tags   : {obj.tags}")
        print(f"    content: {obj.content[:100]}")

    results = mem2.search("standup meeting time")
    print(f"\nSearch 'standup meeting time' → top result:")
    if results:
        r = results[0]
        print(f"  [{r.score:.3f}] {r.path}: {r.chunk_text[:100]}")
    else:
        print("  (no results)")

    print(f"\n{'─' * 60}")
    print("  Demo complete!")
    if use_tmp:
        print(f"  Store at {store_path} — rerun with --store {store_path}")
        print("  to verify persistence across script runs.")
    print('─' * 60)


if __name__ == "__main__":
    main()
