# MQL — Memory Query Language

LLMFS includes a custom query language that compiles to ChromaDB + SQLite queries. MQL provides a SQL-like syntax for expressive memory retrieval.

## Syntax

```sql
-- Semantic similarity search in a path prefix
SELECT memory FROM /knowledge WHERE SIMILAR TO "authentication bug" LIMIT 5

-- Tag filter
SELECT memory FROM /knowledge WHERE TAG = "s3" LIMIT 10

-- Combined semantic + tag filter
SELECT memory FROM /knowledge WHERE SIMILAR TO "bucket error" AND TAG = "s3" LIMIT 5

-- Time-scoped search
SELECT memory FROM /events WHERE date > 2026-01-01 AND date < 2026-04-01

-- Topic / keyword filter
SELECT memory FROM /projects WHERE topic = "authentication"

-- Order by recency
SELECT memory FROM /session ORDER BY created_at DESC LIMIT 10

-- Graph traversal (BFS, depth 2)
SELECT memory FROM /projects RELATED TO "/events/2026-03-15/bug" WITHIN 2
```

## Usage

=== "Python"

    ```python
    results = mem.query(
        'SELECT memory FROM /knowledge WHERE SIMILAR TO "JWT expiry" AND TAG = "auth" LIMIT 5'
    )
    for r in results:
        print(r.path, r.score)
    ```

=== "CLI"

    ```bash
    llmfs query 'SELECT memory FROM /knowledge WHERE SIMILAR TO "auth bug"'
    llmfs query 'SELECT memory FROM /events WHERE TAG = "deploy"' --json
    ```

## Supported Conditions

| Condition | Syntax | Backed By |
|-----------|--------|-----------|
| `SIMILAR TO` | `SIMILAR TO "query string"` | ChromaDB vector search |
| `TAG` | `TAG = "tagname"` | SQLite tag index |
| `date` | `date > 2026-01-01` | SQLite date filter |
| `topic` | `topic = "keyword"` | SQLite metadata filter |
| `RELATED TO` | `RELATED TO "/path" WITHIN N` | Graph BFS traversal |
| `AND` / `OR` | logical combinators | Merged result sets |

## How It Works

1. The **MQL parser** tokenizes the query string and builds an AST (Abstract Syntax Tree)
2. The **MQL executor** walks the AST and translates each node to ChromaDB or SQLite operations
3. Results from different conditions are fused and de-duplicated
4. The final result set is returned as `list[SearchResult]`
