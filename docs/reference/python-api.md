# Python API Reference

All operations go through `MemoryFS`, the single entry point.

```python
from llmfs import MemoryFS

# Initialize — uses ~/.llmfs by default, or .llmfs/ in cwd
mem = MemoryFS()

# Or with a custom path
mem = MemoryFS(path="/tmp/myproject-memory")
```

---

## `write`

Store content at a memory path.

```python
obj = mem.write(
    path="/projects/auth/debug",
    content="User reports bucket creation failure with error: AccessDenied on s3://my-bucket",
    layer="knowledge",          # short_term | session | knowledge | events
    tags=["debug", "s3", "auth"],
    ttl_minutes=None,           # None = permanent; integer = auto-expire
    source="agent",             # manual | agent | mcp | cli
)

print(obj.path)          # /projects/auth/debug
print(obj.layer)         # knowledge
print(obj.chunks)        # list of Chunk objects (auto-chunked + embedded)
print(obj.summaries)     # level_1 (per-chunk) and level_2 (document) summaries
print(obj.metadata.created_at)
```

!!! tip "Deduplication"
    If you write to the same path with identical content, LLMFS skips re-embedding and returns the cached object immediately.

---

## `read`

Read a memory by exact path.

```python
# Full read
obj = mem.read("/projects/auth/debug")
print(obj.content)
print(obj.metadata.tags)
print(obj.relationships)   # linked memories

# Focused read — returns only the chunks most relevant to your query
obj = mem.read("/projects/auth/debug", query="what was the exact error")
print(obj.content)  # only the relevant chunk(s)
```

Raises `MemoryNotFoundError` if the path does not exist.

---

## `search`

Semantic search across all memories.

```python
# Basic semantic search
results = mem.search("bucket creation error", k=5)

# With filters
results = mem.search(
    "authentication bug",
    layer="knowledge",
    tags=["jwt"],
    path_prefix="/projects",
    time_range="last 7 days",
    k=10,
)

for r in results:
    print(f"{r.score:.2f}  {r.path}")
    print(f"  {r.chunk_text[:120]}")
    print(f"  tags={r.tags}  layer={r.metadata['layer']}")
```

Returns `list[SearchResult]` ordered by descending relevance. Results are cached for 5 minutes (configurable).

---

## `update`

Modify an existing memory.

```python
# Append new findings
mem.update("/projects/auth/debug", append="Fixed in commit abc123. Root cause: null pointer.")

# Full content replacement
mem.update("/projects/auth/debug", content="Completely new content.")

# Tag management only
mem.update("/projects/auth/debug", tags_add=["resolved"], tags_remove=["in-progress"])
```

---

## `forget`

Delete memories.

```python
# Delete a specific memory
result = mem.forget("/projects/auth/debug")
print(result)  # {"deleted": 1, "status": "ok"}

# Wipe a layer
mem.forget(layer="short_term")

# Time-based cleanup
mem.forget(older_than="30 days")
```

---

## `relate`

Create a typed, weighted graph edge between two memories.

```python
result = mem.relate(
    source="/events/2026-03-15/bug",
    target="/knowledge/auth/jwt-expiry",
    relationship="caused_by",   # related_to | follows | caused_by | contradicts
    strength=0.92,              # 0.0 to 1.0
)
print(result["relationship_id"])
```

---

## `list`

List memories under a path prefix.

```python
memories = mem.list("/knowledge", recursive=True, layer="knowledge")
for obj in memories:
    print(obj.path, obj.metadata.modified_at)
```

---

## `query` (MQL)

Run a structured MQL query.

```python
results = mem.query(
    'SELECT memory FROM /knowledge WHERE SIMILAR TO "auth bug" LIMIT 5'
)
```

See [MQL Reference](mql.md) for the full query language specification.

---

## `status`

Show storage statistics.

```python
info = mem.status()
# {
#   "total": 142,
#   "layers": {"knowledge": 98, "events": 31, "session": 11, "short_term": 2},
#   "chunks": 891,
#   "disk_mb": 45.2,
#   "base_path": "/home/user/.llmfs"
# }
```

---

## `gc`

Garbage-collect expired (TTL) memories and orphaned chunks.

```python
result = mem.gc()
# {"deleted": 7, "status": "ok"}
```

---

## Error Handling

```python
from llmfs import (
    MemoryNotFoundError,
    MemoryWriteError,
    LLMFSError,
)

try:
    obj = mem.read("/does/not/exist")
except MemoryNotFoundError as e:
    print(f"Not found: {e}")

try:
    mem.write("/bad", ...)
except MemoryWriteError as e:
    print(f"Write failed: {e}")
```

---

## AsyncMemoryFS

An async wrapper for use in async frameworks like FastAPI, LangGraph, and CrewAI:

```python
from llmfs import AsyncMemoryFS

mem = AsyncMemoryFS()
await mem.write("/hello", "world")
results = await mem.search("hello")
```

All methods delegate via `asyncio.to_thread()` for non-blocking operation.
