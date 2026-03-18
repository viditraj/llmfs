"""
ollama_chat.py — Interactive chat REPL with persistent LLMFS memory.

Every conversation turn is stored in LLMFS. When you start a new session
the agent searches for relevant past context before answering, giving it
true long-term memory across restarts.

Special commands (type at the prompt):
  /list          — show all stored memories
  /search <q>    — semantic search without involving the LLM
  /forget <path> — delete a memory by path
  /status        — show memory store stats
  /clear         — wipe ALL memories (asks for confirmation)
  /quit or /exit — exit the REPL

Requirements:
  pip install openai
  ollama pull llama3.2   # or any tool-call-capable model

Run:
  python examples/ollama_chat.py
  python examples/ollama_chat.py --model mistral --store ~/.llmfs_chat
"""

from __future__ import annotations

import argparse
import os
import readline  # noqa: F401 — enables arrow-key history in input()
import sys

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.\n  pip install openai")
    sys.exit(1)

from llmfs import MemoryFS
from llmfs.integrations.openai_tools import LLMFS_TOOLS, LLMFSToolHandler

# ── ANSI colours (disabled if not a TTY) ─────────────────────────────────────
_IS_TTY = sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text

GREY   = lambda t: _c("90",   t)
CYAN   = lambda t: _c("36",   t)
GREEN  = lambda t: _c("32",   t)
YELLOW = lambda t: _c("33",   t)
RED    = lambda t: _c("31",   t)
BOLD   = lambda t: _c("1",    t)


# ── Tool loop ─────────────────────────────────────────────────────────────────

def _run_tool_loop(
    client: OpenAI,
    model: str,
    handler: LLMFSToolHandler,
    messages: list[dict],
    max_rounds: int = 8,
) -> str:
    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=LLMFS_TOOLS,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message
        messages.append(msg.model_dump(exclude_none=True))

        if choice.finish_reason != "tool_calls" or not msg.tool_calls:
            return msg.content or ""

        print(GREY(f"  [tool calls: {', '.join(tc.function.name for tc in msg.tool_calls)}]"))

        tool_results = handler.handle_batch(list(msg.tool_calls))
        for tc, result in zip(msg.tool_calls, tool_results):
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return "(max rounds reached)"


# ── Built-in slash commands ───────────────────────────────────────────────────

def _cmd_list(mem: MemoryFS) -> None:
    objects = mem.list("/")
    if not objects:
        print(YELLOW("  (no memories stored yet)"))
        return
    print(BOLD(f"  {len(objects)} memor{'y' if len(objects) == 1 else 'ies'}:"))
    for obj in objects:
        tags = f"  [{', '.join(obj.tags)}]" if obj.tags else ""
        print(f"  {CYAN(obj.path)}{GREY(tags)}")
        print(GREY(f"    {obj.content[:80]}{'…' if len(obj.content) > 80 else ''}"))


def _cmd_search(mem: MemoryFS, query: str) -> None:
    if not query.strip():
        print(RED("  Usage: /search <query>"))
        return
    results = mem.search(query, k=5)
    if not results:
        print(YELLOW("  (no results)"))
        return
    for r in results:
        print(f"  {CYAN(f'[{r.score:.2f}]')} {BOLD(r.path)}")
        print(GREY(f"    {r.chunk_text[:100]}"))


def _cmd_forget(mem: MemoryFS, path: str) -> None:
    path = path.strip()
    if not path:
        print(RED("  Usage: /forget <path>"))
        return
    result = mem.forget(path)
    print(GREEN(f"  Deleted {result['deleted']} memor{'y' if result['deleted'] == 1 else 'ies'}."))


def _cmd_status(mem: MemoryFS) -> None:
    s = mem.status()
    print(BOLD(f"  Store: {s['base_path']}"))
    print(f"  Memories : {s['total']}")
    print(f"  Chunks   : {s['chunks']}")
    print(f"  Disk     : {s['disk_mb']:.2f} MB")
    if s["layers"]:
        print("  Layers:")
        for layer, count in sorted(s["layers"].items()):
            print(f"    {layer}: {count}")


def _cmd_clear(mem: MemoryFS) -> None:
    confirm = input(RED("  Delete ALL memories? Type 'yes' to confirm: ")).strip()
    if confirm.lower() != "yes":
        print("  Cancelled.")
        return
    result = mem.forget("/", layer=None, older_than=None)
    # forget with no path deletes nothing by design — need layer sweep
    # delete per layer instead
    deleted = 0
    for layer in ["short_term", "session", "knowledge", "events"]:
        r = mem.forget(path=None, layer=layer)
        deleted += r.get("deleted", 0)
    print(GREEN(f"  Cleared {deleted} memories."))


# ── REPL ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful assistant with persistent long-term memory via LLMFS tools.

Guidelines:
- Before answering any non-trivial question, use memory_search to check if you
  have relevant stored context. This gives you memory across sessions.
- Use memory_write to store important facts the user mentions, decisions made,
  or key information that might be useful later.
- Use memory_read when you need the full content of a specific memory.
- Before calling memory_update, always call memory_search first to find the
  exact path of the memory you want to update — never guess the path.
- Prefer /knowledge layer for facts, /session layer for things relevant only to
  this conversation, /short_term for temporary notes.
- Always be transparent: tell the user when you're storing or retrieving memories.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMFS interactive chat with Ollama")
    parser.add_argument("--model", default="llama3.2",
                        help="Ollama model name (default: llama3.2)")
    parser.add_argument("--base-url", default="http://localhost:11434/v1",
                        help="Ollama base URL (default: http://localhost:11434/v1)")
    parser.add_argument("--store", default=os.path.expanduser("~/.llmfs_chat"),
                        help="Memory store path (default: ~/.llmfs_chat)")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="ollama")

    # Verify Ollama
    try:
        client.models.list()
    except Exception as exc:
        print(RED(f"Cannot reach Ollama at {args.base_url}: {exc}"))
        print("Start Ollama with:  ollama serve")
        sys.exit(1)

    mem = MemoryFS(path=args.store)
    handler = LLMFSToolHandler(mem)

    status = mem.status()
    print(BOLD("\nLLMFS Chat"))
    print(f"  Model : {CYAN(args.model)}")
    print(f"  Store : {CYAN(args.store)}  ({status['total']} memories)")
    print(GREY("  Commands: /list  /search <q>  /forget <path>  /status  /clear  /quit"))
    print(GREY("  ─" * 30))

    # Per-session message history (system prompt only — tool results accumulate)
    session_messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input(BOLD("\nYou: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                print("Goodbye.")
                break
            elif cmd == "/list":
                _cmd_list(mem)
            elif cmd == "/search":
                _cmd_search(mem, arg)
            elif cmd == "/forget":
                _cmd_forget(mem, arg)
            elif cmd == "/status":
                _cmd_status(mem)
            elif cmd == "/clear":
                _cmd_clear(mem)
            else:
                print(RED(f"  Unknown command: {cmd}"))
                print(GREY("  Known: /list /search /forget /status /clear /quit"))
            continue

        # ── LLM turn ─────────────────────────────────────────────────────────
        session_messages.append({"role": "user", "content": user_input})

        try:
            reply = _run_tool_loop(client, args.model, handler, session_messages)
        except Exception as exc:
            print(RED(f"  Error: {exc}"))
            # Remove the user message so the session stays consistent
            session_messages.pop()
            continue

        print(f"\n{BOLD('Assistant:')} {reply}")


if __name__ == "__main__":
    main()
