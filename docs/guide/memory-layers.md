# Memory Layers

Every memory in LLMFS belongs to one of four layers, each with different lifetime semantics. This mirrors how humans organize memory — ephemeral scratch notes vs. permanent knowledge.

## Layer Overview

| Layer | Purpose | Default TTL | Use When |
|-------|---------|-------------|----------|
| `short_term` | Temporary reasoning scratch space | 60 minutes | Intermediate calculations, draft thoughts |
| `session` | Current conversation context | Session-scoped | Turn-by-turn chat, in-progress task state |
| `knowledge` | Persistent facts, learnings, code | Permanent | Project knowledge, user preferences, decisions |
| `events` | Timestamped occurrences | Permanent | Bug reports, deployments, meetings, milestones |

## Usage

=== "Python"

    ```python
    mem.write("/scratch/step3", "intermediate result", layer="short_term", ttl_minutes=10)
    mem.write("/session/task", "refactoring auth module", layer="session")
    mem.write("/knowledge/auth/jwt-expiry", "JWT tokens expire after 1h", layer="knowledge")
    mem.write("/events/2026-03-15/deploy", "v2.1 deployed to prod", layer="events")
    ```

=== "CLI"

    ```bash
    llmfs write /scratch/step3 "intermediate result" --layer short_term --ttl 10
    llmfs write /session/task "refactoring auth module" --layer session
    llmfs write /knowledge/auth/jwt-expiry "JWT tokens expire after 1h" --layer knowledge
    llmfs write /events/2026-03-15/deploy "v2.1 deployed to prod" --layer events
    ```

## Garbage Collection

Expired `short_term` memories are garbage-collected automatically on each write cycle (throttled to once per minute). You can also collect manually:

=== "Python"

    ```python
    result = mem.gc()
    # {"deleted": 7, "status": "ok"}
    ```

=== "CLI"

    ```bash
    llmfs gc
    # GC complete. Deleted 7 expired memories.
    ```

## Eviction Priority

When the context manager needs to evict turns from the context window, it uses this priority order (lowest priority evicted first):

1. `short_term` — ephemeral by design
2. `session` — conversation-scoped
3. `events` — important but recoverable
4. `knowledge` — highest priority, evicted last

## Filtering by Layer

You can restrict search and list operations to a specific layer:

```python
results = mem.search("auth bug", layer="knowledge", k=5)
memories = mem.list("/", layer="events")
```

```bash
llmfs search "auth bug" --layer knowledge
llmfs ls / --layer events
```
