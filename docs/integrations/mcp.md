# MCP Server

LLMFS ships a full [Model Context Protocol](https://modelcontextprotocol.io/) server that exposes all core tools to any MCP-compatible client. Once configured, the LLM can call `memory_write`, `memory_search`, `memory_read`, `memory_update`, `memory_forget`, and `memory_relate` natively in its tool loop.

## Auto-Install (Recommended)

```bash
pip install "llmfs[mcp]"
llmfs install-mcp --client claude    # Claude Desktop
llmfs install-mcp --client cursor    # Cursor
llmfs install-mcp --client windsurf  # Windsurf
llmfs install-mcp --client continue  # Continue
```

This writes or merges the following into your client's config file:

```json
{
  "mcpServers": {
    "llmfs": {
      "command": "llmfs",
      "args": ["serve", "--stdio"],
      "description": "AI memory filesystem — persistent, searchable, graph-linked memory"
    }
  }
}
```

### Config File Locations

| Client | Config Path |
|--------|-------------|
| Claude Desktop | **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`<br>**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`<br>**Linux**: `~/.config/Claude/claude_desktop_config.json` |
| Cursor | `~/.cursor/mcp.json` |
| Windsurf | **Windows**: `%APPDATA%\Codeium\Windsurf\mcp_config.json`<br>**macOS/Linux**: `~/.codeium/windsurf/mcp_config.json` |
| Continue | `~/.continue/config.json` |
| GitHub Copilot | **Windows**: `%APPDATA%\github-copilot\mcp_config.json`<br>**macOS/Linux**: `~/.config/github-copilot/mcp_config.json` |

## Manual Config

Print the config JSON to paste it yourself:

```bash
llmfs install-mcp --print
```

Or with a custom store path:

```bash
llmfs install-mcp --client claude --llmfs-path /my/project/.llmfs
```

## Running the Server

```bash
llmfs serve --stdio          # stdio transport (for Claude, Cursor, etc.)
llmfs serve --port 8765      # SSE transport on port 8765
```

## Programmatic Usage

```python
from llmfs import MemoryFS
from llmfs.mcp.server import LLMFSMCPServer

mem = MemoryFS(path="~/.llmfs")
server = LLMFSMCPServer(mem=mem)
server.run_stdio()    # blocking; use as CLI entry-point
# or:
server.run_sse(host="127.0.0.1", port=8765)
```

## The 6 MCP Tools

Once the server is running, the LLM has access to:

| Tool | Description |
|------|-------------|
| `memory_write` | Store content at a path with layer, tags, and optional TTL |
| `memory_search` | Semantic search with layer/tag/time filters |
| `memory_read` | Read a specific memory by exact path (with optional focused query) |
| `memory_update` | Append or replace content; add/remove tags |
| `memory_forget` | Delete by path, layer, or age |
| `memory_relate` | Create a typed, weighted graph edge between two memories |

A system prompt fragment is automatically injected that tells the LLM when and how to use each tool.
