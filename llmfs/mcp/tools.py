"""
MCP tool definitions for LLMFS.

Provides JSON-Schema input definitions and handler functions for the six
LLMFS MCP tools:

1. ``memory_write``   — Store content at a path
2. ``memory_search``  — Semantic search
3. ``memory_read``    — Exact-path retrieval
4. ``memory_update``  — Append/replace/re-tag
5. ``memory_forget``  — Delete memories
6. ``memory_relate``  — Create graph edges

Each handler accepts a :class:`~llmfs.core.filesystem.MemoryFS` instance and
a ``params`` dict, and returns a JSON-serialisable result dict.

Example::

    from llmfs import MemoryFS
    from llmfs.mcp.tools import TOOL_DEFINITIONS, handle_tool_call

    mem = MemoryFS(path="/tmp/test")
    result = handle_tool_call("memory_write", {"path": "/k/hello",
                                               "content": "hi"}, mem)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmfs.core.filesystem import MemoryFS

__all__ = ["TOOL_DEFINITIONS", "handle_tool_call"]

logger = logging.getLogger(__name__)


# ── JSON-Schema tool definitions ──────────────────────────────────────────────


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "memory_write",
        "description": (
            "Store content at a filesystem-style path in LLMFS. "
            "Use this to persist code, decisions, errors, facts, or any "
            "information that may be needed later."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Memory path, must start with /. "
                                   "E.g. /knowledge/auth/architecture",
                },
                "content": {
                    "type": "string",
                    "description": "Text content to store.",
                },
                "layer": {
                    "type": "string",
                    "enum": ["short_term", "session", "knowledge", "events"],
                    "default": "knowledge",
                    "description": "Memory layer (affects TTL and eviction priority).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for filtering.",
                },
                "ttl_minutes": {
                    "type": "integer",
                    "description": "Minutes until auto-expiry. Null = use layer default.",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Semantic search across stored memories. "
            "Returns the most relevant memories for a natural-language query."
        ),
        "inputSchema": {
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
                    "default": 5,
                    "description": "Number of results to return.",
                },
                "time_range": {
                    "type": "string",
                    "description": "Human time string, e.g. 'last 30 minutes', 'today'.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_read",
        "description": (
            "Read the full content of a memory by exact path. "
            "Optionally focus the response on a specific sub-query."
        ),
        "inputSchema": {
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
    {
        "name": "memory_update",
        "description": (
            "Update an existing memory: append text, replace content, or modify tags."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path of the memory to update.",
                },
                "content": {
                    "type": "string",
                    "description": "Full replacement content (mutually exclusive with append).",
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
    {
        "name": "memory_forget",
        "description": (
            "Delete memories by path, layer, or age. "
            "Provide exactly one of: path, layer, or older_than."
        ),
        "inputSchema": {
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
    {
        "name": "memory_relate",
        "description": (
            "Create a directed relationship between two memories. "
            "Useful for linking an error to the code it came from, "
            "or a decision to the requirements that drove it."
        ),
        "inputSchema": {
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
                    "description": "Relationship type, e.g. 'related_to', 'caused_by', 'follows'.",
                },
                "strength": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8,
                    "description": "Edge weight in [0, 1].",
                },
            },
            "required": ["source", "target", "relationship"],
        },
    },
]


# ── Tool handlers ──────────────────────────────────────────────────────────────


def handle_tool_call(
    name: str,
    params: dict[str, Any],
    mem: MemoryFS,
) -> dict[str, Any]:
    """Dispatch a tool call to the appropriate handler.

    Args:
        name: Tool name (must be one of the six LLMFS tools).
        params: Tool parameters dict (as received from the MCP client).
        mem: :class:`~llmfs.core.filesystem.MemoryFS` instance.

    Returns:
        JSON-serialisable result dict.  On error, returns
        ``{"status": "error", "error": "<message>"}``.
    """
    handlers: dict[str, Any] = {
        "memory_write": _handle_write,
        "memory_search": _handle_search,
        "memory_read": _handle_read,
        "memory_update": _handle_update,
        "memory_forget": _handle_forget,
        "memory_relate": _handle_relate,
    }
    handler = handlers.get(name)
    if handler is None:
        return {"status": "error", "error": f"Unknown tool: {name!r}"}
    try:
        return handler(params, mem)
    except Exception as exc:
        logger.exception("Tool %r failed: %s", name, exc)
        return {"status": "error", "error": str(exc)}


# ── Individual handlers ────────────────────────────────────────────────────────


def _handle_write(params: dict[str, Any], mem: MemoryFS) -> dict[str, Any]:
    path = params["path"]
    content = params["content"]
    layer = params.get("layer", "knowledge")
    tags = params.get("tags") or []
    ttl_minutes = params.get("ttl_minutes")
    obj = mem.write(path, content, layer=layer, tags=tags,
                    ttl_minutes=ttl_minutes, source="mcp")
    return {
        "status": "ok",
        "path": obj.path,
        "layer": obj.layer,
        "chunks": len(obj.chunks),
        "tags": obj.tags,
    }


def _handle_search(params: dict[str, Any], mem: MemoryFS) -> dict[str, Any]:
    query = params["query"]
    layer = params.get("layer")
    tags = params.get("tags")
    k = int(params.get("k", 5))
    time_range = params.get("time_range")
    results = mem.search(query, layer=layer, tags=tags, k=k, time_range=time_range)
    return {
        "status": "ok",
        "count": len(results),
        "results": [
            {
                "path": r.path,
                "score": round(r.score, 4),
                "snippet": r.chunk_text[:200],
                "tags": r.tags,
                "layer": r.metadata.get("layer", ""),
                "created_at": r.metadata.get("created_at", ""),
            }
            for r in results
        ],
    }


def _handle_read(params: dict[str, Any], mem: MemoryFS) -> dict[str, Any]:
    path = params["path"]
    query = params.get("query")
    obj = mem.read(path, query=query)
    return {
        "status": "ok",
        "path": obj.path,
        "content": obj.content,
        "layer": obj.layer,
        "tags": obj.tags,
        "created_at": obj.metadata.created_at,
        "modified_at": obj.metadata.modified_at,
        "chunks": len(obj.chunks),
    }


def _handle_update(params: dict[str, Any], mem: MemoryFS) -> dict[str, Any]:
    path = params["path"]
    content = params.get("content")
    append = params.get("append")
    tags_add = params.get("tags_add")
    tags_remove = params.get("tags_remove")
    obj = mem.update(path, content=content, append=append,
                     tags_add=tags_add, tags_remove=tags_remove)
    return {
        "status": "ok",
        "path": obj.path,
        "chunks": len(obj.chunks),
        "tags": obj.tags,
    }


def _handle_forget(params: dict[str, Any], mem: MemoryFS) -> dict[str, Any]:
    path = params.get("path")
    layer = params.get("layer")
    older_than = params.get("older_than")
    if not path and not layer and not older_than:
        return {"status": "error", "error": "Provide at least one of: path, layer, older_than"}
    result = mem.forget(path, layer=layer, older_than=older_than)
    return result


def _handle_relate(params: dict[str, Any], mem: MemoryFS) -> dict[str, Any]:
    source = params["source"]
    target = params["target"]
    relationship = params["relationship"]
    strength = float(params.get("strength", 0.8))
    return mem.relate(source, target, relationship, strength=strength)
