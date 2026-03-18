"""
OpenAI function-calling integration for LLMFS.

Provides:

:data:`LLMFS_TOOLS`
    A list of OpenAI-format tool definitions (JSON Schema) for all six LLMFS
    memory operations.  Pass directly to any OpenAI API call as ``tools=``.

:class:`LLMFSToolHandler`
    Dispatches OpenAI ``tool_calls`` objects to the correct
    :class:`~llmfs.core.filesystem.MemoryFS` method and returns JSON-encoded
    result strings.

Example::

    import openai
    from llmfs import MemoryFS
    from llmfs.integrations.openai_tools import LLMFS_TOOLS, LLMFSToolHandler

    mem = MemoryFS()
    handler = LLMFSToolHandler(mem)

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Remember that JWT expires in 15 min"}],
        tools=LLMFS_TOOLS,
    )

    # If the model called a tool:
    if response.choices[0].finish_reason == "tool_calls":
        tool_calls = response.choices[0].message.tool_calls
        results = handler.handle_batch(tool_calls)

Note: ``openai`` is an optional dependency.  Import this module without it
installed is fine; instantiation of :class:`LLMFSToolHandler` will succeed
even without ``openai``, since it only depends on
:class:`~llmfs.core.filesystem.MemoryFS`.
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmfs.core.filesystem import MemoryFS

__all__ = ["LLMFS_TOOLS", "LLMFSToolHandler"]

logger = logging.getLogger(__name__)


# ── OpenAI tool definitions ───────────────────────────────────────────────────

LLMFS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": (
                "Store content at a filesystem-style path in LLMFS. "
                "Use this to persist code snippets, decisions, errors, facts, "
                "or any information that may be needed in a future turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Memory path, must start with /. E.g. /knowledge/auth/bug",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to store.",
                    },
                    "layer": {
                        "type": "string",
                        "enum": ["short_term", "session", "knowledge", "events"],
                        "description": "Memory layer (affects TTL). Defaults to 'knowledge'.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of string tags for filtering.",
                    },
                    "ttl_minutes": {
                        "type": "integer",
                        "description": "Minutes until auto-expiry. Null = use layer default.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": (
                "Semantic search across stored memories. "
                "Returns the most relevant memories for a natural-language query. "
                "Use this when you need context that may have been evicted or is from a past session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural-language search query.",
                    },
                    "layer": {
                        "type": "string",
                        "enum": ["short_term", "session", "knowledge", "events"],
                        "description": "Restrict search to a specific layer.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Only return memories with all of these tags.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": (
                "Read the full content of a memory by its exact path. "
                "Use this when you know the path and need the complete content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Exact memory path.",
                    },
                    "query": {
                        "type": "string",
                        "description": "If provided, return only the most relevant chunks.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": (
                "Update an existing memory: append text, replace its content, or modify its tags."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path of the memory to update.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full replacement content.",
                    },
                    "append": {
                        "type": "string",
                        "description": "Text to append to existing content.",
                    },
                    "tags_add": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to add.",
                    },
                    "tags_remove": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags to remove.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_forget",
            "description": (
                "Delete memories by exact path, layer, or age. "
                "Provide exactly one of: path, layer, or older_than."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Delete a single memory by exact path.",
                    },
                    "layer": {
                        "type": "string",
                        "enum": ["short_term", "session", "knowledge", "events"],
                        "description": "Delete all memories in this layer.",
                    },
                    "older_than": {
                        "type": "string",
                        "description": "Delete memories older than this duration, e.g. '7 days'.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_relate",
            "description": (
                "Create a directed relationship between two memories. "
                "Useful for linking an error to the file it came from, "
                "a decision to the requirements that drove it, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source memory path.",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target memory path.",
                    },
                    "relationship": {
                        "type": "string",
                        "description": "Edge type, e.g. 'related_to', 'caused_by', 'follows'.",
                    },
                    "strength": {
                        "type": "number",
                        "description": "Edge weight in [0, 1]. Defaults to 0.8.",
                    },
                },
                "required": ["source", "target", "relationship"],
            },
        },
    },
]


# ── LLMFSToolHandler ──────────────────────────────────────────────────────────


class LLMFSToolHandler:
    """Dispatches OpenAI ``tool_calls`` to :class:`~llmfs.core.filesystem.MemoryFS`.

    Args:
        mem: :class:`~llmfs.core.filesystem.MemoryFS` instance.

    Example::

        handler = LLMFSToolHandler(mem)
        # tool_call can be an OpenAI ToolCall object or a plain dict
        result_str = handler.handle(tool_call)
        # result_str is a JSON string suitable as the tool result message
    """

    def __init__(self, mem: MemoryFS) -> None:
        self._mem = mem

    def handle(self, tool_call: Any) -> str:
        """Dispatch a single tool call and return a JSON result string.

        Args:
            tool_call: OpenAI ``ToolCall`` object or dict with ``function.name``
                and ``function.arguments`` keys.

        Returns:
            JSON-encoded result string.
        """
        name, raw_args = _extract_tool_call(tool_call)
        try:
            params = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError as exc:
            return json.dumps({"status": "error", "error": f"Invalid JSON arguments: {exc}"})

        from llmfs.mcp.tools import handle_tool_call
        result = handle_tool_call(name, params, self._mem)
        return json.dumps(result, default=str)

    def handle_batch(self, tool_calls: list[Any]) -> list[str]:
        """Dispatch a list of tool calls.

        Args:
            tool_calls: List of OpenAI ``ToolCall`` objects or dicts.

        Returns:
            List of JSON-encoded result strings, one per tool call.
        """
        return [self.handle(tc) for tc in tool_calls]

    def tool_result_messages(self, tool_calls: list[Any]) -> list[dict[str, Any]]:
        """Dispatch tool calls and return OpenAI-format tool result messages.

        Suitable for appending back to the ``messages`` list before the next
        API call.

        Args:
            tool_calls: List of OpenAI ``ToolCall`` objects or dicts.

        Returns:
            List of ``{"role": "tool", "tool_call_id": ..., "content": ...}``
            dicts.
        """
        messages: list[dict[str, Any]] = []
        for tc in tool_calls:
            call_id = _extract_call_id(tc)
            result_str = self.handle(tc)
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": result_str,
            })
        return messages


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_tool_call(tool_call: Any) -> tuple[str, str]:
    """Extract (name, arguments_str) from an OpenAI ToolCall or dict."""
    if isinstance(tool_call, dict):
        fn = tool_call.get("function", {})
        name = fn.get("name", "")
        args = fn.get("arguments", "{}")
    else:
        # OpenAI SDK object: tool_call.function.name, tool_call.function.arguments
        fn = getattr(tool_call, "function", None)
        if fn is not None:
            name = getattr(fn, "name", "")
            args = getattr(fn, "arguments", "{}")
        else:
            name = getattr(tool_call, "name", "")
            args = getattr(tool_call, "arguments", "{}")
    return name, args


def _extract_call_id(tool_call: Any) -> str:
    """Extract the tool_call_id from an OpenAI ToolCall or dict."""
    if isinstance(tool_call, dict):
        return tool_call.get("id", "")
    return getattr(tool_call, "id", "")
