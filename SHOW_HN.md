# Show HN Post Draft

## Title (choose one)

**Option A (recommended):**
> Show HN: LLMFS – Virtual memory for LLMs: unlimited context with zero information loss

**Option B:**
> Show HN: LLMFS – A filesystem-based memory layer that gives AI agents unlimited context

**Option C:**
> Show HN: I built an OS-style virtual memory system for LLMs

---

## Post Body

**URL to submit:** https://github.com/viditraj/llmfs

**Text (paste into the HN text field):**

---

Every LLM agent hits the same wall: the context window fills up. The standard fix — summarize and truncate — destroys 94% of the information. Ask the model about a specific line of code from 30 turns ago and it can only apologize.

LLMFS takes a different approach, borrowed from operating systems: treat the context window as RAM and LLMFS as disk/swap.

- Context fills up → old turns are evicted to LLMFS (not summarized — stored at full fidelity)
- LLM needs old info → it calls memory_search/memory_read (a "page fault")
- A compact memory index (~2k tokens) stays in the system prompt so the LLM knows what's available

The result: 500k+ tokens of memory, zero information loss, sub-100ms retrieval.

**What it actually is:** A Python library (`pip install llmfs`) that gives LLMs persistent, searchable, structured memory organized as a filesystem. Memories live at paths like `/projects/auth/bug`, are semantically embedded (ChromaDB + all-MiniLM-L6-v2), and can be linked in a knowledge graph.

**Key features:**
- 4 memory layers with TTL (short_term/session/knowledge/events)
- Hybrid search: semantic vectors + BM25 keyword + reciprocal rank fusion
- MQL query language: `SELECT memory FROM /knowledge WHERE SIMILAR TO "auth bug" LIMIT 5`
- Knowledge graph with typed relationships and BFS/DFS traversal
- Built-in MCP server (works with Claude, Cursor, Windsurf)
- Drop-in LangChain + OpenAI function-calling adapters
- FUSE mount — `cat /mnt/memory/knowledge/auth/jwt-expiry`
- Runs 100% local: SQLite + ChromaDB, no API keys, no GPU, 22 MB embedding model

**It's not:** a product like ChatGPT's memory (which stores a few text snippets). It's infrastructure for developers building agents that need to remember everything across sessions.

Built with Python, MIT licensed. Feedback welcome — especially on the retrieval pipeline and the virtual memory model.

Docs: https://viditraj.github.io/llmfs
PyPI: https://pypi.org/project/llmfs/

---

## Tips for Posting

1. **When to post:** Tuesday–Thursday, 8–10 AM US Eastern (12–14 UTC) gets the most eyeballs
2. **Don't ask for upvotes** — HN will penalize you
3. **Be in the comments** for the first 2 hours answering questions
4. **Prepare answers for likely questions:**
   - "How is this different from mem0/Letta/Zep?" → filesystem metaphor, virtual memory model, MQL, knowledge graph, fully local
   - "Why not just use a bigger context window?" → cost, latency, and models still lose info at long contexts; LLMFS retrieves precisely what's needed
   - "Does this actually work in production?" → it's beta, but the architecture is sound; benchmarks in the README
   - "Why a filesystem metaphor?" → intuitive for LLMs (they understand paths), hierarchical organization, composable with any tool that reads/writes files
5. **Cross-post to Reddit** 1–2 hours after HN if it gets traction:
   - r/LocalLLaMA (biggest impact for local-first tools)
   - r/MachineLearning
   - r/Python
