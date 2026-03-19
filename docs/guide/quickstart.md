# Quick Start

## CLI

Initialize a store, write a memory, and search it:

```bash
# Initialize a store in the current directory
llmfs init
# Initialised LLMFS at /your/project/.llmfs

# Write your first memory
llmfs write /knowledge/hello "LLMFS stores memories at filesystem paths"

# Search it back
llmfs search "how does memory storage work"

# Check what's in the store
llmfs status
```

## Python API

Five lines to get started:

```python
from llmfs import MemoryFS

mem = MemoryFS()
mem.write("/projects/auth/bug", "JWT expiry misconfigured at auth.py:45", tags=["jwt", "bug"])
results = mem.search("authentication error", k=3)
print(results[0].path, results[0].score)
```

## Store Location

LLMFS looks for a store in this order:

1. `--llmfs-path` flag or `LLMFS_PATH` environment variable
2. `.llmfs/` in the current directory
3. `~/.llmfs` (global fallback)

```bash
# Use a project-local store
llmfs init  # creates .llmfs/ in cwd

# Use a custom path
llmfs --llmfs-path /tmp/my-store write /hello "world"
```

```python
# Default: ~/.llmfs or .llmfs/ in cwd
mem = MemoryFS()

# Custom path
mem = MemoryFS(path="/tmp/myproject-memory")
```

## What Happens Under the Hood

When you write a memory:

1. Content is split into **chunks** (code-aware AST splitting or prose-aware paragraph splitting)
2. Each chunk is **embedded** with `all-MiniLM-L6-v2` (local, CPU, ~22 MB)
3. Vectors are stored in **ChromaDB** (HNSW index for sub-linear search)
4. Metadata is stored in **SQLite** (WAL mode for concurrent reads)
5. If `auto_link=True`, similar existing memories are linked with `related_to` edges

When you search:

1. Query is embedded and searched against ChromaDB (dense/semantic)
2. Query is also searched via SQLite FTS5 (sparse/BM25 keyword)
3. Results are fused with **Reciprocal Rank Fusion**
4. Results are de-duplicated and ranked by combined score
5. Results are cached for 5 minutes

## Next Steps

- [Python API Reference](../reference/python-api.md) — all methods and options
- [CLI Reference](../reference/cli.md) — all commands
- [Memory Layers](memory-layers.md) — `short_term`, `session`, `knowledge`, `events`
- [Infinite Context](../reference/infinite-context.md) — the ContextMiddleware
- [MCP Server](../integrations/mcp.md) — use with Claude, Cursor, Windsurf
