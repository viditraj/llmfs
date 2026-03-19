# Memory Graph

Link related memories to build a navigable knowledge graph. The graph enables relationship-aware retrieval and reasoning.

## Creating Relationships

```python
# Create typed relationships
mem.relate("/events/2026-03-15/bug",     "/knowledge/auth/jwt-expiry",  "caused_by",  strength=0.92)
mem.relate("/knowledge/auth/jwt-expiry", "/knowledge/auth/architecture", "related_to", strength=0.85)
mem.relate("/events/2026-03-14/deploy",  "/events/2026-03-15/bug",      "follows",    strength=1.0)
```

```bash
llmfs relate /events/2026-03-15/bug /knowledge/auth/jwt-expiry caused_by
llmfs relate /knowledge/auth/jwt-expiry /knowledge/auth/architecture related_to --strength 0.95
```

## Relationship Types

| Type | Meaning |
|------|---------|
| `related_to` | Generic semantic connection |
| `follows` | Temporal succession (A happened after B) |
| `caused_by` | Causal link (A was caused by B) |
| `contradicts` | Conflicting information |

## Graph Traversal

### Via MQL

```sql
SELECT memory FROM /knowledge RELATED TO "/events/2026-03-15/bug" WITHIN 2
```

### Via Python API

```python
from llmfs.graph.memory_graph import MemoryGraph

graph = MemoryGraph(mem._db)
neighbors = graph.get_neighbors("/knowledge/auth/jwt-expiry")
path      = graph.traverse("/events/2026-03-15/bug", max_depth=3)
```

Traversal supports both BFS (breadth-first) and DFS (depth-first) with configurable depth limits.

## Auto-Linking

LLMFS can automatically link semantically similar memories on every `write()`:

```python
mem = MemoryFS(auto_link=True, auto_link_threshold=0.75, auto_link_k=3)
```

After writing a memory, the top-k most similar existing memories are linked with `related_to` edges if their similarity exceeds the threshold. Disable with `auto_link=False`.

## How It Works

Graph edges are stored in the `relationships` table in SQLite:

- **source_id** / **target_id**: Memory file IDs
- **type**: One of the four relationship types
- **strength**: Float from 0.0 to 1.0
- **created_at**: Timestamp

Traversal results include visited paths, edges, and a depth map showing how far each memory is from the starting node.
