"""
Drop-in context middleware for LLMFS.

:class:`ContextMiddleware` wraps any callable agent (or object with a
``chat``/``invoke`` method) and provides it with effectively infinite context
by automatically offloading old turns to LLMFS.

Usage adds exactly **two lines** to existing agent code::

    from llmfs import MemoryFS
    from llmfs.context.middleware import ContextMiddleware

    agent = YourAgent(model="gpt-4o")
    agent = ContextMiddleware(agent, memory=MemoryFS())

The middleware:

1. Intercepts every turn (before + after the underlying agent call).
2. Scores importance of each new turn.
3. Auto-evicts when active tokens exceed 70 % of ``max_tokens``.
4. Extracts structured artifacts before eviction.
5. Rebuilds the memory index after eviction.
6. Injects the index into the system message before each call.
7. Exposes ``memory_search`` / ``memory_read`` as tool stubs (informational).

Supported agent interfaces:

- **Callable** — ``agent(messages) -> response``
- **chat** method — ``agent.chat(messages) -> response``
- **invoke** method — ``agent.invoke(input) -> output``

Example::

    from llmfs.context.middleware import ContextMiddleware
    from llmfs import MemoryFS

    def my_agent(messages):
        # simulate a chat model
        return {"role": "assistant", "content": "Hello!"}

    mem = MemoryFS(path="/tmp/test")
    wrapped = ContextMiddleware(my_agent, memory=mem, max_tokens=4000)
    response = wrapped.chat([{"role": "user", "content": "Hi"}])
"""
from __future__ import annotations

import logging
from typing import Any

from llmfs.context.manager import ContextManager
from llmfs.core.filesystem import MemoryFS

__all__ = ["ContextMiddleware"]

logger = logging.getLogger(__name__)

# Rough token estimate: split on whitespace
def _TOKEN_ESTIMATE(text: str) -> int:
    return max(1, len(str(text).split()))


class ContextMiddleware:
    """Wraps an agent with automatic LLMFS context management.

    Args:
        agent: Any callable, or an object with a ``chat`` or ``invoke``
            method.
        memory: :class:`~llmfs.core.filesystem.MemoryFS` instance.
        max_tokens: Context window capacity (tokens).
        evict_at: Eviction trigger threshold (fraction of ``max_tokens``).
        target_after_evict: Target tokens after eviction.
        session_id: Optional session identifier.  Random UUID hex if omitted.

    Example::

        wrapped = ContextMiddleware(agent_fn, memory=MemoryFS())
        response = wrapped.chat([{"role": "user", "content": "Hello"}])
    """

    def __init__(
        self,
        agent: Any,
        *,
        memory: MemoryFS,
        max_tokens: int = 128_000,
        evict_at: float = 0.70,
        target_after_evict: float = 0.50,
        session_id: str | None = None,
    ) -> None:
        self._agent = agent
        self._ctx = ContextManager(
            mem=memory,
            max_tokens=max_tokens,
            evict_at=evict_at,
            target_after_evict=target_after_evict,
            session_id=session_id,
        )
        self._call_count: int = 0
        self._cache_hits: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(self, messages: list[dict[str, Any]]) -> Any:
        """Pass *messages* through the middleware and call the underlying agent.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.

        Returns:
            Whatever the underlying agent returns.
        """
        self._call_count += 1

        # Track incoming turns (all non-system messages)
        for msg in messages:
            if msg.get("role") in ("user", "assistant"):
                content = msg.get("content", "")
                tokens = _TOKEN_ESTIMATE(content)
                self._ctx.on_new_turn(msg["role"], content, tokens)

        # Build augmented messages with memory index injected
        augmented = self._inject_memory_index(messages)

        # Call the underlying agent
        response = self._call_agent(augmented)

        # Track the assistant's response
        if response:
            resp_content = _extract_content(response)
            if resp_content:
                self._ctx.on_new_turn("assistant", resp_content, _TOKEN_ESTIMATE(resp_content))

        return response

    def invoke(self, input: Any) -> Any:  # noqa: A002
        """LangChain-style ``invoke`` shim.

        Wraps the input in a user message, calls :meth:`chat`, and returns
        the raw agent response.

        Args:
            input: String or dict.  If a string, it becomes a ``user`` message.

        Returns:
            Agent response.
        """
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        elif isinstance(input, dict) and "messages" in input:
            messages = input["messages"]
        else:
            messages = [{"role": "user", "content": str(input)}]
        return self.chat(messages)

    def get_context_stats(self) -> dict[str, Any]:
        """Return context usage statistics.

        Returns:
            Dict with ``call_count``, ``session_id``, plus all keys from
            :meth:`~llmfs.context.manager.ContextManager.get_stats`.
        """
        stats = self._ctx.get_stats()
        stats["call_count"] = self._call_count
        return stats

    def get_active_turns(self) -> list[dict[str, Any]]:
        """Return turns currently in the active context window."""
        return self._ctx.get_active_turns()

    def get_memory_index(self) -> str:
        """Return the current memory index string."""
        return self._ctx.get_system_prompt_addon()

    def reset_session(self) -> dict[str, Any]:
        """Reset the session (clear evicted memories and active turns)."""
        return self._ctx.reset_session()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _inject_memory_index(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Prepend or augment the system message with the memory index."""
        index = self._ctx.get_system_prompt_addon()
        if not index:
            return messages

        result = list(messages)
        system_idx = next(
            (i for i, m in enumerate(result) if m.get("role") == "system"),
            None,
        )
        if system_idx is not None:
            existing = result[system_idx].get("content", "")
            result[system_idx] = {
                "role": "system",
                "content": f"{existing}\n\n{index}".strip(),
            }
        else:
            result.insert(0, {"role": "system", "content": index})

        return result

    def _call_agent(self, messages: list[dict[str, Any]]) -> Any:
        """Dispatch to the underlying agent using the best available interface."""
        agent = self._agent
        if callable(agent) and not hasattr(agent, "chat") and not hasattr(agent, "invoke"):
            return agent(messages)
        if hasattr(agent, "chat"):
            return agent.chat(messages)
        if hasattr(agent, "invoke"):
            return agent.invoke({"messages": messages})
        # Fallback: try calling directly
        return agent(messages)

    def __repr__(self) -> str:
        stats = self._ctx.get_stats()
        return (
            f"ContextMiddleware(session={stats['session_id']!r}, "
            f"tokens={stats['total_tokens']}/{stats['max_tokens']})"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_content(response: Any) -> str:
    """Best-effort extraction of text content from an agent response."""
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return response.get("content", "") or response.get("text", "") or ""
    if hasattr(response, "content"):
        return str(response.content)
    return str(response)
