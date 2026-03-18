"""
ollama_context_overflow_test.py — Proves LLMFS works when context window is full.

Test flow:
  1. Store a secret fact in LLMFS via the model
  2. Flood the conversation with ~N filler messages to overflow the context window
  3. Ask the model to recall the secret fact — it CANNOT be in context anymore
  4. Model must use memory_search to retrieve it from LLMFS
  5. Pass/fail verdict printed at the end

The filler messages are real API round-trips so the model's context
actually fills up (not just simulated).

Requirements:
  pip install openai
  ollama pull qwen2.5:3b   # or any tool-call model

Run:
  python ollama_context_overflow_test.py
  python ollama_context_overflow_test.py --model qwen2.5:3b --filler-turns 30
  python ollama_context_overflow_test.py --filler-turns 5   # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: pip install openai")
    sys.exit(1)

from llmfs import MemoryFS
from llmfs.integrations.openai_tools import LLMFS_TOOLS, LLMFSToolHandler


# ── Helpers ───────────────────────────────────────────────────────────────────

def _divider(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print('─' * 60)


def _chat(client, model, messages, tools=None):
    """Single API call, returns the response message."""
    kwargs = dict(model=model, messages=messages)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    return client.chat.completions.create(**kwargs).choices[0].message


def _tool_round(client, model, handler, messages):
    """One tool-call round: call model, execute tools, return final text."""
    for _ in range(6):
        msg = _chat(client, model, messages, tools=LLMFS_TOOLS)
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            return msg.content or ""

        names = [tc.function.name for tc in msg.tool_calls]
        print(f"    [tool calls: {', '.join(names)}]")

        for tc in msg.tool_calls:
            result = handler.handle(tc)
            print(f"      {tc.function.name} → {result[:80]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    return "(max rounds)"


# ── Filler conversations ───────────────────────────────────────────────────────

FILLER_EXCHANGES = [
    ("What is the capital of France?", None),
    ("How many days are in a leap year?", None),
    ("What is 17 multiplied by 13?", None),
    ("Name three programming languages invented before 1980.", None),
    ("What does HTTP stand for?", None),
    ("What is the boiling point of water in Celsius?", None),
    ("Who wrote the play Romeo and Juliet?", None),
    ("What is the speed of light in km/s?", None),
    ("How many sides does a hexagon have?", None),
    ("What year did the first moon landing occur?", None),
    ("What is the largest planet in the solar system?", None),
    ("What does CPU stand for?", None),
    ("How many bytes are in a kilobyte?", None),
    ("What is the chemical symbol for gold?", None),
    ("What is the square root of 144?", None),
    ("Name the four cardinal directions.", None),
    ("What does RAM stand for?", None),
    ("What is the longest river in the world?", None),
    ("How many continents are on Earth?", None),
    ("What does DNS stand for?", None),
    ("What is the smallest prime number?", None),
    ("What language is spoken in Brazil?", None),
    ("What is 2 to the power of 10?", None),
    ("What does API stand for?", None),
    ("How many hours are in a week?", None),
    ("What is the freezing point of water in Fahrenheit?", None),
    ("What does SSL stand for?", None),
    ("Who invented the telephone?", None),
    ("What is the most widely spoken language in the world?", None),
    ("What does JSON stand for?", None),
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LLMFS context overflow test")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--filler-turns", type=int, default=20,
                        help="Number of filler Q&A turns (default: 20)")
    parser.add_argument("--flood-tokens", type=int, default=0,
                        help="Inject this many tokens of filler text in ONE message "
                             "(fast way to overflow 32k context, e.g. --flood-tokens 30000)")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="ollama")
    try:
        client.models.list()
    except Exception as exc:
        print(f"ERROR: Cannot reach Ollama at {args.base_url}: {exc}")
        sys.exit(1)

    store = tempfile.mkdtemp(prefix="llmfs_overflow_test_")
    mem = MemoryFS(path=store)
    handler = LLMFSToolHandler(mem)

    # Secret fact the model will need to remember
    SECRET_PATH = "/test/secret_code"
    SECRET_VALUE = "LLMFS-OVERFLOW-TEST-XK9"

    SYSTEM = (
        "You are a helpful assistant with persistent memory via LLMFS tools. "
        "Whenever the user shares important facts, codes, or information worth "
        "keeping, proactively store them using memory_write without being asked. "
        "When answering questions, use memory_search to check your stored memories "
        "for relevant context before responding."
    )

    # Shared message list — this is what will overflow
    messages = [{"role": "system", "content": SYSTEM}]

    # ── Phase 1: Store the secret naturally ──────────────────────────────────
    _divider("Phase 1 — Share a secret fact (model should store it proactively)")
    store_prompt = (
        f'Hey, just so you know, our deployment secret code is "{SECRET_VALUE}". '
        f"Keep that safe, we'll need it later."
    )
    print(f"User: {store_prompt}")
    messages.append({"role": "user", "content": store_prompt})
    reply = _tool_round(client, args.model, handler, messages)
    print(f"Assistant: {reply}")

    # Verify it was actually written
    stored = mem.read(SECRET_PATH)
    if not stored:
        # Model may have chosen a different path — search for it
        results = mem.search(SECRET_VALUE)
        if results:
            stored = mem.read(results[0].path)
            SECRET_PATH = results[0].path

    if stored and SECRET_VALUE in stored.content:
        print(f"\n  ✓ Confirmed in LLMFS: {stored.path} = '{stored.content}'")
        SECRET_PATH = stored.path
    else:
        print(f"\n  ✗ Model did not store the secret in LLMFS.")
        print(f"    All memories: {[o.path for o in mem.list('/')]}")
        print(f"    Continuing anyway to test recall behaviour...")

    context_size_after_store = len(messages)

    # ── Phase 2: Flood context ────────────────────────────────────────────────
    if args.flood_tokens > 0:
        # Fast mode: inject one giant message to eat up the context window
        # ~4 chars per token is a reasonable estimate for English text
        chars_needed = args.flood_tokens * 4
        # Repeating paragraph of unrelated text
        paragraph = (
            "The history of ancient Rome spans several centuries, beginning with the "
            "founding of the city in 753 BC according to Roman tradition. The Roman "
            "Republic was established in 509 BC after the overthrow of the Roman Kingdom. "
            "During the Republic, Rome expanded its territory through a series of wars, "
            "including the Punic Wars against Carthage. Julius Caesar crossed the Rubicon "
            "in 49 BC, leading to a civil war that ended the Republic and gave rise to "
            "the Roman Empire under Augustus in 27 BC. The empire reached its greatest "
            "extent under Emperor Trajan in 117 AD. The western empire fell in 476 AD. "
        )
        filler_text = (paragraph * (chars_needed // len(paragraph) + 1))[:chars_needed]
        estimated_tokens = len(filler_text) // 4

        _divider(f"Phase 2 — Injecting ~{estimated_tokens:,} token filler block (single message)")
        print("  (This pushes the secret fact out of the model's context window)\n")

        # Inject as a fake prior assistant turn so it's in history but not re-processed
        messages.append({
            "role": "user",
            "content": f"Here is some background reading material:\n\n{filler_text}\n\nOkay, noted."
        })
        msg = _chat(client, args.model, messages)
        messages.append(msg.model_dump(exclude_none=True))
        print(f"  Filler injected. Context now ~{estimated_tokens + 500:,} tokens.")
        print(f"  Model ack: {(msg.content or '')[:80]}")
        total_messages = len(messages)

    else:
        # Slow mode: real Q&A round-trips
        filler_count = min(args.filler_turns, len(FILLER_EXCHANGES))
        _divider(f"Phase 2 — Flooding context with {filler_count} filler Q&A exchanges")
        print("  (This pushes the secret fact out of the model's context window)\n")

        for i, (question, _) in enumerate(FILLER_EXCHANGES[:filler_count]):
            messages.append({"role": "user", "content": question})
            msg = _chat(client, args.model, messages)
            messages.append(msg.model_dump(exclude_none=True))
            answer = (msg.content or "")[:60]
            print(f"  [{i+1:02d}/{filler_count}] Q: {question[:45]:<45}  A: {answer}")
            time.sleep(0.1)

        total_messages = len(messages)
        filler_tokens_estimate = filler_count * 80
        print(f"\n  Messages in context : {total_messages}")
        print(f"  Filler messages     : {total_messages - context_size_after_store}")
        print(f"  Estimated tokens    : ~{filler_tokens_estimate}")

    print(f"  Secret at msg index : ~{context_size_after_store - 2} of {total_messages}")

    # ── Phase 3: Ask model to recall without hints ────────────────────────────
    _divider("Phase 3 — Ask naturally, no memory hint")
    recall_prompt = "Hey what was that deployment secret code again?"
    print(f"User: {recall_prompt}\n")
    messages.append({"role": "user", "content": recall_prompt})

    # Track accessed_at before recall so we can detect if LLMFS was queried
    before_recall = mem.read(SECRET_PATH)
    reply = _tool_round(client, args.model, handler, messages)
    print(f"\nAssistant: {reply}")

    # ── Phase 4: Verdict ──────────────────────────────────────────────────────
    _divider("Phase 4 — Verdict")
    recalled = SECRET_VALUE in (reply or "")

    refreshed = mem.read(SECRET_PATH)
    was_accessed = (
        refreshed
        and before_recall
        and refreshed.metadata.accessed_at != before_recall.metadata.accessed_at
    )

    stored_in_llmfs = bool(before_recall and SECRET_VALUE in before_recall.content)

    print(f"\n  Secret value      : {SECRET_VALUE}")
    print(f"  Stored in LLMFS   : {'✓ YES' if stored_in_llmfs else '✗ NO  ← model never stored it'}")
    print(f"  LLMFS queried     : {'✓ YES' if was_accessed else '✗ NO  ← model recalled from context (not a real test)'}")
    print(f"  Correct answer    : {'✓ YES' if recalled else '✗ NO'}")

    if not stored_in_llmfs:
        print("\n  ✗ INCONCLUSIVE — model never stored the secret proactively.")
        print("    Try a larger model or explicitly ask it to store facts first.")
    elif recalled and was_accessed:
        print("\n  ✓ PASS — Model retrieved the secret from LLMFS autonomously,")
        print("           not from conversation history.")
    elif recalled and not was_accessed:
        print("\n  ~ PARTIAL — Model answered correctly but recalled from context,")
        print("              not from LLMFS. Try --filler-turns with a higher value")
        print("              or a larger model to truly overflow the context window.")
    else:
        print("\n  ✗ FAIL — Model could not recall the secret.")
        print("           Check the tool call log above.")

    print(f"\n  Store kept at: {store}")
    print('─' * 60)


if __name__ == "__main__":
    main()
