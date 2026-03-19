# Feature Comparison

How LLMFS compares to other LLM memory solutions.

## vs. Developer Tools

| Feature | mem0 | Letta (MemGPT) | ChromaDB | **LLMFS** |
|---------|------|-----------------|----------|-----------|
| Filesystem metaphor | :x: | :x: | :x: | :white_check_mark: |
| Memory layers with TTL | Partial | :white_check_mark: | :x: | :white_check_mark: |
| Knowledge graph | :x: | :x: | :x: | :white_check_mark: |
| Custom query language (MQL) | :x: | :x: | SQL-like | **Custom MQL** |
| Auto-compression & chunking | :x: | :white_check_mark: | :x: | :white_check_mark: |
| Infinite context (VM model) | :x: | :x: | :x: | :white_check_mark: |
| CLI interface | :x: | :white_check_mark: | :x: | :white_check_mark: |
| Local-first, no server needed | :x: | :x: | :white_check_mark: | :white_check_mark: |
| Zero-config (`llmfs init`) | :x: | :x: | Partial | :white_check_mark: |
| MCP server built-in | :x: | :x: | :x: | :white_check_mark: |
| FUSE filesystem mount | :x: | :x: | :x: | :white_check_mark: |
| Drop-in agent middleware | :x: | :x: | :x: | :white_check_mark: |

## vs. ChatGPT / Claude Memory

Products like ChatGPT and Claude have built-in "memory" features, but they solve a fundamentally different problem:

| | ChatGPT / Claude Memory | LLMFS |
|---|---|---|
| **What it stores** | Short text snippets ("user prefers Python") | Full documents, code, conversations at full fidelity |
| **Storage location** | Vendor servers | Your machine, your data |
| **Structure** | Flat list of facts | Filesystem paths, layers, tags, knowledge graph |
| **Query capability** | None (opaque, injected into prompt) | Semantic search, BM25, MQL, graph traversal |
| **Programmatic access** | No API | Full Python API, CLI, MCP server |
| **Cross-agent sharing** | No | Yes — multiple agents share one LLMFS |
| **Vendor lock-in** | 100% | Zero — works with any LLM |
| **Context overflow** | Truncation or lossy summary | Virtual memory with zero information loss |

!!! info "Different audiences"
    ChatGPT/Claude memory is designed for casual users who want the model to remember their name and preferences. LLMFS is infrastructure for developers building AI agents that need structured, searchable, persistent memory at scale.

## When to Use LLMFS

LLMFS is a good fit when you need:

- **Persistent memory across sessions** — agents that remember everything
- **Structured organization** — not just a bag of embeddings, but a filesystem with paths, layers, and tags
- **Unlimited context** — conversations that span thousands of turns without losing detail
- **Multi-agent coordination** — shared memory between specialized agents
- **Local-first privacy** — no data leaves your machine
- **Vendor independence** — works with OpenAI, Anthropic, Ollama, or any LLM
