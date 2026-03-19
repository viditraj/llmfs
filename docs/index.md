---
hide:
  - navigation
---

# LLMFS — Filesystem Memory for LLMs and AI Agents

<p align="center">
  <a href="https://pypi.org/project/llmfs/"><img src="https://img.shields.io/pypi/v/llmfs.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/viditraj/llmfs/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href="https://github.com/viditraj/llmfs/actions"><img src="https://img.shields.io/github/actions/workflow/status/viditraj/llmfs/ci.yml?label=tests" alt="Tests"></a>
</p>

**LLMFS gives LLMs and AI agents persistent, searchable, structured memory — organized like a filesystem.** Instead of losing context when a conversation grows past the token limit, agents offload memories to LLMFS and retrieve exactly what they need, when they need it.

The result: **zero information loss** and an **effectively unlimited context window** — even over thousands of turns.

---

## The Problem

Every LLM agent eventually hits the same wall: **the context window fills up.**

The standard solution — lossy summarization — destroys information. When an agent summarizes 80k tokens into 5k, **94% of the detail is gone forever.** Ask it about a specific line of code from 30 turns ago, and it can only apologize.

## The Solution

LLMFS takes a different approach, borrowed directly from operating systems:

```
OS Concept     →   LLM Concept
──────────────────────────────────────────────────────────
RAM            →   Context Window (e.g. 128k tokens)
Disk / Swap    →   LLMFS  (500k+ tokens, full fidelity)
Page eviction  →   Offload old turns to LLMFS
Page fault     →   LLM calls memory_search / memory_read
Virtual addr   →   Memory path  (/session/turns/42)
MMU            →   ContextManager
```

Memories are stored at filesystem-style paths (`/projects/auth/bug`, `/events/2026-03-15_fix`) and searched semantically. They persist across sessions, support TTLs, carry metadata and tags, and can be linked in a knowledge graph.

---

## Quick Start

```bash
pip install llmfs
llmfs init
llmfs write /knowledge/hello "LLMFS stores memories at filesystem paths"
llmfs search "how does memory storage work"
```

```python
from llmfs import MemoryFS

mem = MemoryFS()
mem.write("/projects/auth/bug", "JWT expiry misconfigured at auth.py:45", tags=["jwt", "bug"])
results = mem.search("authentication error", k=3)
print(results[0].path, results[0].score)
```

[:material-rocket-launch: Get Started](guide/installation.md){ .md-button .md-button--primary }
[:material-book-open-variant: Python API](reference/python-api.md){ .md-button }
[:material-github: GitHub](https://github.com/viditraj/llmfs){ .md-button }

---

## Key Features

<div class="grid cards" markdown>

-   :material-file-tree:{ .lg .middle } **Filesystem Metaphor**

    ---

    Organize memories at intuitive paths like `/projects/auth/bug` with hierarchical structure, tags, and metadata.

-   :material-infinity:{ .lg .middle } **Unlimited Context**

    ---

    Virtual memory model evicts old turns to LLMFS and pages them back in on demand. Zero information loss.

-   :material-magnify:{ .lg .middle } **Hybrid Search**

    ---

    Semantic vector search + BM25 keyword search with reciprocal rank fusion. Sub-100ms over 10k memories.

-   :material-graph:{ .lg .middle } **Knowledge Graph**

    ---

    Link memories with typed relationships (`caused_by`, `follows`, `contradicts`) and traverse with BFS/DFS.

-   :material-connection:{ .lg .middle } **MCP Server**

    ---

    Built-in Model Context Protocol server for Claude, Cursor, Windsurf, and any MCP client.

-   :material-database-search:{ .lg .middle } **MQL Query Language**

    ---

    Custom query language: `SELECT memory FROM /knowledge WHERE SIMILAR TO "auth bug" LIMIT 5`

-   :material-puzzle:{ .lg .middle } **Framework Integrations**

    ---

    Drop-in adapters for LangChain, OpenAI function calling, and any tool-use LLM.

-   :material-harddisk:{ .lg .middle } **Local-First**

    ---

    Runs entirely on your machine. SQLite + ChromaDB. No API keys needed. 22 MB embedding model, CPU-only.

</div>

---

## Performance

| Operation | Target | Notes |
|-----------|--------|-------|
| Write (500 tokens) | < 200 ms | Includes chunking + embedding |
| Search (10k memories) | < 100 ms | Cached repeats in < 1 ms |
| Read (by path) | < 10 ms | SQLite lookup + chunk assembly |
| MQL query | < 200 ms | Parse + search |
| Context eviction (20 turns) | < 500 ms | Includes artifact extraction |
