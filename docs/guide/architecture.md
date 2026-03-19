# Architecture

LLMFS uses a layered architecture with a single entry point (`MemoryFS`) that delegates to specialized subsystems.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         Public Interfaces                        │
│  CLI (llmfs)   Python API   MCP Server   LangChain   OpenAI    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │    MemoryFS     │  ← Single entry point
                    │  (filesystem.py)│    write · read · search
                    │                 │    update · forget · relate
                    │                 │    list · query · gc
                    └────┬──────┬─────┘
                         │      │
          ┌──────────────▼──┐ ┌─▼──────────────────┐
          │  Storage Layer  │ │  Embedding Layer     │
          │                 │ │                      │
          │ SQLite (WAL)    │ │ all-MiniLM-L6-v2     │
          │  · file registry│ │ (local, 22 MB, CPU)  │
          │  · chunks       │ │ or OpenAI text-      │
          │  · tags         │ │ embedding-3-small    │
          │  · graph edges  │ │                      │
          │  · search cache │ └──────────────────────┘
          │                 │
          │ ChromaDB        │
          │  · chunk vectors│
          │  · metadata     │
          │  · similarity   │
          └─────────────────┘
                    │
          ┌─────────▼──────────────────────────────────┐
          │               Processing Pipeline           │
          │                                            │
          │  AdaptiveChunker   ExtractiveSummarizer    │
          │  (code-aware AST   (TF-IDF, level 1+2)    │
          │   or prose-aware)                          │
          │                                            │
          │  RetrievalEngine   MemoryGraph             │
          │  (semantic +       (BFS/DFS, relationship  │
          │   temporal +        types, strength)       │
          │   graph hybrid)                            │
          │                                            │
          │  ContextManager    MQL Parser+Executor     │
          │  (virtual memory   (custom query language  │
          │   manager for       → ChromaDB + SQLite)   │
          │   infinite ctx)                            │
          └────────────────────────────────────────────┘
```

## On-Disk Layout

```
~/.llmfs/            # default; or .llmfs/ in current directory
  metadata.db        # SQLite — file registry, chunks, tags, graph, cache
  chroma/            # ChromaDB persistence — embedding vectors
  config.json        # optional configuration overrides
```

## Storage Layer

### SQLite (metadata.db)

All structured data lives in SQLite with WAL mode enabled for concurrent reads:

| Table | Purpose |
|-------|---------|
| `files` | File registry (id, path, name, layer, size, timestamps, TTL, content hash) |
| `chunks` | Chunk text, byte offsets, embedding IDs, per-chunk summaries |
| `tags` / `file_tags` | Many-to-many tag system |
| `relationships` | Graph edges (source, target, type, strength) |
| `search_cache` | Query result cache (SHA-256 keyed, 5-min TTL) |
| `embedding_cache` | Avoids re-embedding identical strings |
| `chunks_fts` | FTS5 virtual table for BM25 keyword search |

### ChromaDB

Chunk embeddings are stored in a ChromaDB collection with an HNSW index (cosine similarity). Each chunk carries metadata: `file_id`, `path`, `layer`, `chunk_index`, `tags`.

## Embedding Layer

LLMFS lazy-loads the embedder on first use:

- **Default:** `all-MiniLM-L6-v2` via sentence-transformers (22 MB, CPU-only, 1000+ queries/sec)
- **Optional:** OpenAI `text-embedding-3-small` (higher quality for some domains, requires API key)

Embeddings are cached in SQLite keyed by `(text_hash, model_name)` to avoid redundant computation across sessions.

## Processing Pipeline

### Adaptive Chunker

Content-aware chunking with three strategies:

| Content Type | Target Size | Strategy |
|-------------|-------------|----------|
| Python code | 512 tokens | AST-based splitting at `def`/`class` boundaries |
| Markdown | 256 tokens | Header-based splitting, sub-split by paragraph |
| Plain text | 256 tokens | Sliding window with 50-token overlap |

### Retrieval Engine

Two-stage hybrid search:

1. **Dense search** — query embedded, cosine similarity via ChromaDB
2. **Sparse search** — BM25 keyword matching via SQLite FTS5
3. **Fusion** — Reciprocal Rank Fusion merges results, deduplicates by path

### Memory Graph

Typed, weighted edges between memories with BFS/DFS traversal. Relationship types: `related_to`, `follows`, `caused_by`, `contradicts`.

### Context Manager

The virtual memory subsystem that powers infinite context. See [Infinite Context](../reference/infinite-context.md) for details.
