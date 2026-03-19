"""
openai_agent.py — Full OpenAI function-calling loop with LLMFS tools.

Demonstrates the complete tool-use cycle:
  1. Build messages list with user input
  2. Send to (mock) OpenAI API with LLMFS_TOOLS
  3. Detect finish_reason == "tool_calls"
  4. Dispatch each tool call via LLMFSToolHandler.handle_batch()
  5. Append tool result messages back to the conversation
  6. Make a follow-up API call for the final response
  7. Repeat until no more tool calls

This example is fully runnable without the ``openai`` package.  A mock
client is used that returns canned tool_call sequences.  When ``openai``
IS installed, swap ``MockOpenAIClient`` for the real ``openai.OpenAI()``
client and remove the canned responses.

LLMFS_TOOLS exposes 7 operations as OpenAI function definitions:
  memory_write    memory_read    memory_search
  memory_update   memory_forget  memory_relate
  memory_list

Run:
    python examples/openai_agent.py
"""

from __future__ import annotations

import json
import tempfile
import textwrap
import uuid
from typing import Any

from llmfs import MemoryFS
from llmfs.integrations.openai_tools import LLMFS_TOOLS, LLMFSToolHandler


# ── Mock OpenAI SDK data structures ───────────────────────────────────────────

class _Function:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name: str, arguments: str) -> None:
        self.id = f"call_{uuid.uuid4().hex[:8]}"
        self.type = "function"
        self.function = _Function(name, arguments)


class _Message:
    def __init__(
        self,
        role: str,
        content: str | None = None,
        tool_calls: list[_ToolCall] | None = None,
    ) -> None:
        self.role = role
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Response:
    def __init__(self, choices: list[_Choice]) -> None:
        self.choices = choices


# ── Mock OpenAI client ─────────────────────────────────────────────────────────

# To use the real OpenAI client instead:
#   import openai
#   client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#
# Then call:
#   response = client.chat.completions.create(
#       model="gpt-4o",
#       messages=messages,
#       tools=LLMFS_TOOLS,
#   )

class MockOpenAIClient:
    """Simulates the OpenAI Chat Completions API with canned tool-call sequences.

    Each item in ``_scenarios`` represents one API call's response.  The mock
    cycles through them in order to simulate a realistic multi-turn exchange.
    """

    # Each tuple: (finish_reason, optional tool_calls or final content)
    _SCRIPT: list[dict[str, Any]] = [
        # Turn 1: model decides to write a memory first
        {
            "finish_reason": "tool_calls",
            "tool_calls": [
                _ToolCall(
                    "memory_write",
                    json.dumps({
                        "path": "/projects/nexus/auth_bug",
                        "content": (
                            "Bug: JWT expiry set to 0 seconds in config.py line 42. "
                            "This causes all users to be logged out immediately after login. "
                            "Fix: change JWT_EXPIRY_SECONDS to 3600."
                        ),
                        "layer": "knowledge",
                        "tags": ["bug", "auth", "jwt", "critical"],
                    }),
                )
            ],
        },
        # Turn 2: model also searches to see if there are related memories
        {
            "finish_reason": "tool_calls",
            "tool_calls": [
                _ToolCall(
                    "memory_search",
                    json.dumps({
                        "query": "authentication JWT token expiry bug",
                        "k": 3,
                    }),
                )
            ],
        },
        # Turn 3: model writes a related memory and links them
        {
            "finish_reason": "tool_calls",
            "tool_calls": [
                _ToolCall(
                    "memory_write",
                    json.dumps({
                        "path": "/projects/nexus/auth_fix_plan",
                        "content": (
                            "Fix plan for auth bug:\n"
                            "1. Change JWT_EXPIRY_SECONDS = 3600 in config.py:42\n"
                            "2. Add unit test: test_token_expiry_positive\n"
                            "3. Deploy to staging, verify login flow\n"
                            "4. Rotate JWT_SECRET while we're here"
                        ),
                        "layer": "knowledge",
                        "tags": ["fix-plan", "auth", "sprint1"],
                    }),
                ),
                _ToolCall(
                    "memory_relate",
                    json.dumps({
                        "source": "/projects/nexus/auth_fix_plan",
                        "target": "/projects/nexus/auth_bug",
                        "relationship": "fixes",
                        "strength": 0.95,
                    }),
                ),
            ],
        },
        # Turn 4: final text response (no more tool calls)
        {
            "finish_reason": "stop",
            "content": (
                "I've stored the auth bug details and fix plan in LLMFS under "
                "/projects/nexus/. The JWT expiry was set to 0 seconds — I've "
                "documented the fix (change to 3600s) and linked the bug to the "
                "fix plan in the knowledge graph. The memories will persist across "
                "sessions so any future conversation can retrieve this context."
            ),
        },
    ]

    def __init__(self) -> None:
        self._idx = 0

    class _Completions:
        def __init__(self, client: "MockOpenAIClient") -> None:
            self._client = client

        def create(
            self,
            model: str,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> _Response:
            script = MockOpenAIClient._SCRIPT
            step = script[self._client._idx % len(script)]
            self._client._idx += 1

            if step["finish_reason"] == "tool_calls":
                return _Response([
                    _Choice(
                        message=_Message(
                            role="assistant",
                            content=None,
                            tool_calls=step["tool_calls"],
                        ),
                        finish_reason="tool_calls",
                    )
                ])
            else:
                return _Response([
                    _Choice(
                        message=_Message(role="assistant", content=step["content"]),
                        finish_reason="stop",
                    )
                ])

    @property
    def chat(self) -> "_Chat":
        return self._Chat(self)

    class _Chat:
        def __init__(self, client: "MockOpenAIClient") -> None:
            self._client = client

        @property
        def completions(self) -> "MockOpenAIClient._Completions":
            return MockOpenAIClient._Completions(self._client)


# ── The agent loop ─────────────────────────────────────────────────────────────

def run_agent_loop(
    client: Any,
    handler: LLMFSToolHandler,
    initial_messages: list[dict[str, Any]],
    max_rounds: int = 10,
    verbose: bool = True,
) -> str:
    """Standard OpenAI function-calling loop with LLMFS tool dispatch.

    Args:
        client:           openai.OpenAI() or MockOpenAIClient.
        handler:          LLMFSToolHandler bound to a MemoryFS instance.
        initial_messages: Starting messages (system + user).
        max_rounds:       Safety limit on tool-call rounds.
        verbose:          Print round-by-round progress.

    Returns:
        The final text content from the assistant.
    """
    messages = list(initial_messages)
    round_num = 0

    while round_num < max_rounds:
        round_num += 1

        # ── Call the model ─────────────────────────────────────────────────────
        response = client.chat.completions.create(
            model="gpt-4o",           # ignored by mock; real client uses this
            messages=messages,
            tools=LLMFS_TOOLS,        # pass all 6 LLMFS tool definitions
        )
        choice = response.choices[0]

        if verbose:
            print(f"\nRound {round_num}: finish_reason={choice.finish_reason!r}")

        # ── No more tool calls → done ──────────────────────────────────────────
        if choice.finish_reason != "tool_calls":
            final = choice.message.content or ""
            if verbose:
                print(f"Final response:\n{textwrap.fill(final, 72, subsequent_indent='  ')}")
            return final

        # ── Dispatch tool calls via LLMFSToolHandler ───────────────────────────
        tool_calls = choice.message.tool_calls or []

        # Append the assistant's tool_call message
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        # Execute each tool call and collect results
        for tc in tool_calls:
            result_str = handler.handle(tc)
            result = json.loads(result_str)

            if verbose:
                print(f"  Tool: {tc.function.name}({tc.function.arguments[:60]}…)")
                status = result.get("status", result.get("error", "?"))
                print(f"    → {status}")

            # Append tool result as required by OpenAI API
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    return "[max_rounds exceeded]"


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Try real openai if installed; fall back to mock
    try:
        import openai as _openai
        import os
        _api_key = os.environ.get("OPENAI_API_KEY", "")
        if _api_key:
            client: Any = _openai.OpenAI(api_key=_api_key)
            print("Using real OpenAI client")
        else:
            raise ValueError("no key")
    except Exception:
        client = MockOpenAIClient()
        print("Using mock OpenAI client (set OPENAI_API_KEY to use real API)")

    with tempfile.TemporaryDirectory(prefix="llmfs_openai_") as tmp:
        mem = MemoryFS(path=tmp)
        handler = LLMFSToolHandler(mem)

        print(f"\nLLMFS_TOOLS contains {len(LLMFS_TOOLS)} tool definition(s):")
        for t in LLMFS_TOOLS:
            print(f"  • {t['function']['name']}")

        initial_messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful engineering assistant with access to LLMFS — "
                    "a persistent memory filesystem. Use memory_write to store important "
                    "findings, memory_search to retrieve context, and memory_relate to "
                    "link related memories in the knowledge graph."
                ),
            },
            {
                "role": "user",
                "content": (
                    "I just found a critical auth bug in the Nexus project. "
                    "JWT tokens expire immediately (expiry=0). Please store this "
                    "in memory, search for any related auth context, write a fix plan, "
                    "and link them together."
                ),
            },
        ]

        print("\n" + "─" * 60)
        print("Starting agent loop…")
        final = run_agent_loop(client, handler, initial_messages, verbose=True)

        # Verify memories were written
        print("\n" + "─" * 60)
        print("Verifying persisted memories:")
        all_mems = mem.list("/projects")
        for obj in all_mems:
            print(f"  {obj.path}  tags={obj.tags}")
            print(f"    {obj.content[:80]}…")

        # Show semantic search works on the stored data
        print("\nSearch 'authentication bug':")
        results = mem.search("authentication bug token expiry", k=3)
        for r in results:
            print(f"  [{r.score:.3f}] {r.path}")

        status = mem.status()
        print(f"\nMemoryFS: {status['total']} memories, {status['disk_mb']:.3f} MB")
