"""
LLMFS MCP Server.

Exposes LLMFS as a Model Context Protocol (MCP) server compatible with
Claude, Cursor, Windsurf, Continue, and any MCP-compliant LLM client.

**Transports:**

- ``stdio`` (default) — communicate over stdin/stdout; used by local MCP clients
- ``sse`` — HTTP + Server-Sent Events; used for remote/multi-client setups

**Start commands:**

.. code-block:: bash

    llmfs serve --stdio           # stdio transport
    llmfs serve --port 8765       # SSE transport on port 8765

**MCP config for auto-install (see** :func:`generate_mcp_config` **):**

.. code-block:: json

    {
      "mcpServers": {
        "llmfs": {
          "command": "llmfs",
          "args": ["serve", "--stdio"],
          "description": "AI memory filesystem"
        }
      }
    }

Example (programmatic)::

    from llmfs import MemoryFS
    from llmfs.mcp.server import LLMFSMCPServer

    mem = MemoryFS(path="~/.llmfs")
    server = LLMFSMCPServer(mem=mem)
    server.run_stdio()     # blocks; use in CLI entry-point
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from llmfs.core.filesystem import MemoryFS
from llmfs.mcp.prompts import LLMFS_SYSTEM_PROMPT
from llmfs.mcp.tools import handle_tool_call

__all__ = ["LLMFSMCPServer", "generate_mcp_config"]

logger = logging.getLogger(__name__)

_SERVER_NAME = "llmfs"
_SERVER_DESCRIPTION = "AI memory filesystem — persistent, searchable, graph-linked memory"


class LLMFSMCPServer:
    """MCP server wrapping a :class:`~llmfs.core.filesystem.MemoryFS` instance.

    Args:
        mem: :class:`~llmfs.core.filesystem.MemoryFS` instance to serve.
        name: Server name shown to MCP clients.  Defaults to ``"llmfs"``.

    Example::

        server = LLMFSMCPServer(mem=MemoryFS())
        server.run_stdio()   # blocking; run from CLI entry-point
    """

    def __init__(
        self,
        mem: MemoryFS,
        *,
        name: str = _SERVER_NAME,
    ) -> None:
        self._mem = mem
        self._name = name
        self._mcp = self._build_server()

    # ── Public API ────────────────────────────────────────────────────────────

    def run_stdio(self) -> None:
        """Start the server on stdin/stdout (blocking).

        Use this for local MCP client integrations (Claude Desktop, Cursor,
        Windsurf, Continue).
        """
        logger.info("Starting LLMFS MCP server (stdio)")
        self._mcp.run(transport="stdio")

    def run_sse(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        """Start an HTTP + SSE server (blocking).

        Args:
            host: Bind host.  Defaults to ``"127.0.0.1"``.
            port: TCP port.  Defaults to ``8765``.
        """
        logger.info("Starting LLMFS MCP server (SSE) on %s:%d", host, port)
        # FastMCP reads host/port from constructor; rebuild with custom values
        self._mcp = self._build_server(host=host, port=port)
        self._mcp.run(transport="sse")

    # ── Server construction ───────────────────────────────────────────────────

    def _build_server(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> Any:
        """Build and return a configured :class:`~mcp.server.fastmcp.FastMCP` instance."""
        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP(
            name=self._name,
            instructions=LLMFS_SYSTEM_PROMPT,
            host=host,
            port=port,
        )
        self._register_tools(mcp)
        return mcp

    def _register_tools(self, mcp: Any) -> None:
        """Register all six LLMFS tools on *mcp*."""
        mem = self._mem

        @mcp.tool(
            name="memory_write",
            description=(
                "Store content at a filesystem-style path in LLMFS. "
                "Use this to persist code, decisions, errors, facts, or any "
                "information that may be needed later."
            ),
        )
        def memory_write(
            path: str,
            content: str,
            layer: str = "knowledge",
            tags: list[str] | None = None,
            ttl_minutes: int | None = None,
        ) -> dict[str, Any]:
            """Write *content* to *path* in LLMFS.

            Args:
                path: Memory path, must start with /.
                content: Text content to store.
                layer: Memory layer (short_term/session/knowledge/events).
                tags: Optional list of string tags.
                ttl_minutes: Minutes until auto-expiry.
            """
            return handle_tool_call(
                "memory_write",
                {"path": path, "content": content, "layer": layer,
                 "tags": tags or [], "ttl_minutes": ttl_minutes},
                mem,
            )

        @mcp.tool(
            name="memory_search",
            description=(
                "Semantic search across stored memories. "
                "Returns the most relevant memories for a natural-language query."
            ),
        )
        def memory_search(
            query: str,
            layer: str | None = None,
            tags: list[str] | None = None,
            k: int = 5,
            time_range: str | None = None,
        ) -> dict[str, Any]:
            """Search LLMFS for memories matching *query*.

            Args:
                query: Natural-language search query.
                layer: Restrict search to a specific layer.
                tags: Only return memories with all of these tags.
                k: Number of results to return.
                time_range: Human time string, e.g. 'last 30 minutes'.
            """
            return handle_tool_call(
                "memory_search",
                {"query": query, "layer": layer, "tags": tags, "k": k,
                 "time_range": time_range},
                mem,
            )

        @mcp.tool(
            name="memory_read",
            description=(
                "Read the full content of a memory by exact path. "
                "Optionally focus the response on a specific sub-query."
            ),
        )
        def memory_read(
            path: str,
            query: str | None = None,
        ) -> dict[str, Any]:
            """Read the memory at *path*.

            Args:
                path: Exact memory path.
                query: If provided, return only the most relevant chunks.
            """
            return handle_tool_call("memory_read", {"path": path, "query": query}, mem)

        @mcp.tool(
            name="memory_update",
            description=(
                "Update an existing memory: append text, replace content, or modify tags."
            ),
        )
        def memory_update(
            path: str,
            content: str | None = None,
            append: str | None = None,
            tags_add: list[str] | None = None,
            tags_remove: list[str] | None = None,
        ) -> dict[str, Any]:
            """Update the memory at *path*.

            Args:
                path: Path of the memory to update.
                content: Full replacement content (mutually exclusive with append).
                append: Text to append to existing content.
                tags_add: Tags to add.
                tags_remove: Tags to remove.
            """
            return handle_tool_call(
                "memory_update",
                {"path": path, "content": content, "append": append,
                 "tags_add": tags_add, "tags_remove": tags_remove},
                mem,
            )

        @mcp.tool(
            name="memory_forget",
            description=(
                "Delete memories by path, layer, or age. "
                "Provide exactly one of: path, layer, or older_than."
            ),
        )
        def memory_forget(
            path: str | None = None,
            layer: str | None = None,
            older_than: str | None = None,
        ) -> dict[str, Any]:
            """Delete memories matching the given criteria.

            Args:
                path: Delete a single memory by exact path.
                layer: Delete all memories in this layer.
                older_than: Delete memories older than this duration, e.g. '7 days'.
            """
            return handle_tool_call(
                "memory_forget",
                {"path": path, "layer": layer, "older_than": older_than},
                mem,
            )

        @mcp.tool(
            name="memory_relate",
            description=(
                "Create a directed relationship between two memories. "
                "Useful for linking an error to the code it came from, "
                "or a decision to the requirements that drove it."
            ),
        )
        def memory_relate(
            source: str,
            target: str,
            relationship: str,
            strength: float = 0.8,
        ) -> dict[str, Any]:
            """Create a graph edge from *source* to *target*.

            Args:
                source: Source memory path.
                target: Target memory path.
                relationship: Edge type, e.g. 'related_to', 'caused_by', 'follows'.
                strength: Edge weight in [0, 1].
            """
            return handle_tool_call(
                "memory_relate",
                {"source": source, "target": target,
                 "relationship": relationship, "strength": strength},
                mem,
            )


# ── Config generation ─────────────────────────────────────────────────────────


def generate_mcp_config(
    *,
    llmfs_path: str | None = None,
) -> dict[str, Any]:
    """Generate an MCP server config dict for ``claude_desktop_config.json``.

    Args:
        llmfs_path: If provided, pass ``--llmfs-path`` to the server.

    Returns:
        Dict suitable for JSON serialisation.
    """
    args: list[str] = ["serve", "--stdio"]
    if llmfs_path:
        args += ["--llmfs-path", llmfs_path]

    return {
        "mcpServers": {
            "llmfs": {
                "command": "llmfs",
                "args": args,
                "description": _SERVER_DESCRIPTION,
            }
        }
    }


# ── MCP config installer ──────────────────────────────────────────────────────

_CLIENT_PATHS: dict[str, Path] = {
    "claude": Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
    "cursor": Path.home() / ".cursor" / "mcp.json",
    "continue": Path.home() / ".continue" / "config.json",
    "windsurf": Path.home() / ".codeium" / "windsurf" / "mcp_config.json",
}

# Linux fallback for Claude
if not (_CLIENT_PATHS["claude"]).parent.exists():
    _CLIENT_PATHS["claude"] = (
        Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    )


def install_mcp_config(
    client: str,
    *,
    llmfs_path: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write or merge the LLMFS MCP config into a client's config file.

    Args:
        client: Target client name (``"claude"``, ``"cursor"``,
            ``"continue"``, ``"windsurf"``).
        llmfs_path: Optional ``--llmfs-path`` value to embed.
        dry_run: If ``True``, return the config dict without writing.

    Returns:
        Dict with ``{"status": "ok"|"dry_run", "path": ..., "config": ...}``.

    Raises:
        ValueError: If *client* is not recognised.
    """
    if client not in _CLIENT_PATHS:
        raise ValueError(
            f"Unknown client {client!r}. "
            f"Valid clients: {sorted(_CLIENT_PATHS)}"
        )

    config_path = _CLIENT_PATHS[client]
    new_config = generate_mcp_config(llmfs_path=llmfs_path)

    if dry_run:
        return {"status": "dry_run", "path": str(config_path), "config": new_config}

    # Merge with existing config
    existing: dict[str, Any] = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    mcp_servers = existing.setdefault("mcpServers", {})
    mcp_servers.update(new_config["mcpServers"])

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("install_mcp_config: wrote %s", config_path)
    return {"status": "ok", "path": str(config_path), "config": new_config}
