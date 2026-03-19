# CLI Reference

All commands accept `--llmfs-path` (or `LLMFS_PATH` env var) to point to a custom store. By default LLMFS looks for `.llmfs/` in the current directory, then falls back to `~/.llmfs`.

---

## `llmfs init`

Initialize a new store in the current directory.

```bash
llmfs init
# Initialised LLMFS at /your/project/.llmfs
```

---

## `llmfs write`

Store content at a memory path.

```bash
# From inline content
llmfs write /knowledge/auth/bug "JWT expiry misconfigured at auth.py line 45"

# From a file
llmfs write /knowledge/architecture --file ARCHITECTURE.md

# With layer, tags, and TTL
llmfs write /session/plan "Refactor auth module today" \
    --layer session --tags "plan,auth" --ttl 480

# From stdin
cat report.md | llmfs write /knowledge/report
```

| Flag | Description |
|------|-------------|
| `--layer` | `short_term` \| `session` \| `knowledge` \| `events` (default: `knowledge`) |
| `--tags` | Comma-separated tags, e.g. `"jwt,bug,auth"` |
| `--ttl` | Minutes until auto-expiry |
| `--file` | Read content from a file path |

---

## `llmfs read`

Read a memory by exact path.

```bash
llmfs read /knowledge/auth/bug

# Focused read: return only chunks relevant to your query
llmfs read /knowledge/auth/bug --query "what line number"
```

---

## `llmfs search`

Semantic search across all memories.

```bash
llmfs search "authentication error"
llmfs search "bucket creation error" --layer knowledge --tags s3 --k 10
llmfs search "auth bug" --time "last 7 days"
```

| Flag | Description |
|------|-------------|
| `--layer` | Restrict to a layer |
| `--tags` | Comma-separated required tags |
| `--k` | Number of results (default: 5) |
| `--time` | Human time string: `"last 30 minutes"`, `"today"`, `"last 7 days"` |

---

## `llmfs update`

Modify an existing memory.

```bash
# Append new information
llmfs update /knowledge/auth/bug --append "Fixed in commit abc123"

# Replace content entirely
llmfs update /knowledge/auth/bug --content "Bug resolved. Root cause: missing null check."

# Manage tags
llmfs update /knowledge/auth/bug --tags-add "resolved" --tags-remove "in-progress"
```

---

## `llmfs forget`

Delete memories.

```bash
# Delete a specific memory
llmfs forget /knowledge/auth/bug

# Wipe an entire layer
llmfs forget --layer short_term

# Delete memories older than a duration
llmfs forget --older-than "30 days"

# Skip confirmation prompt
llmfs forget /session/old-task --yes
```

---

## `llmfs relate`

Link two memories in the knowledge graph.

```bash
llmfs relate /events/2026-03-15/bug /knowledge/auth/jwt-expiry caused_by
llmfs relate /knowledge/auth/jwt-expiry /knowledge/auth/architecture related_to --strength 0.95
```

---

## `llmfs query`

Run a structured MQL query.

```bash
llmfs query 'SELECT memory FROM /knowledge WHERE SIMILAR TO "auth bug" LIMIT 5'
llmfs query 'SELECT memory FROM /events WHERE TAG = "deploy" LIMIT 10' --json
```

---

## `llmfs ls`

List memories under a path prefix.

```bash
llmfs ls /knowledge
llmfs ls /session --layer session
```

---

## `llmfs status`

Show storage statistics.

```bash
llmfs status
# LLMFS Status  (/home/user/.llmfs)
#   Total memories : 142
#   Total chunks   : 891
#   Disk usage     : 45.2 MB
```

---

## `llmfs gc`

Garbage-collect expired (TTL) memories and orphaned chunks.

```bash
llmfs gc
# GC complete. Deleted 7 expired memories.
```

---

## `llmfs serve`

Start the MCP server.

```bash
llmfs serve --stdio          # stdio transport (for Claude, Cursor, etc.)
llmfs serve --port 8765      # SSE transport on port 8765
```

---

## `llmfs install-mcp`

Auto-configure LLMFS as an MCP server in a supported client.

```bash
llmfs install-mcp --client claude     # Claude Desktop
llmfs install-mcp --client cursor     # Cursor
llmfs install-mcp --client windsurf   # Windsurf
llmfs install-mcp --client continue   # Continue
llmfs install-mcp --print             # Print config JSON to stdout
```

---

## `llmfs mount` / `llmfs unmount`

Mount LLMFS as a FUSE filesystem (requires `pip install "llmfs[fuse]"`).

```bash
llmfs mount /mnt/memory
llmfs unmount /mnt/memory
```
