# Examples

LLMFS ships with runnable example scripts in the [`examples/`](https://github.com/viditraj/llmfs/tree/main/examples) directory.

## Quick Start

```bash
pip install -r examples/requirements.txt
ollama pull llama3.2
ollama serve
```

Then:

```bash
# Scripted demo -- proves LLMFS works end-to-end (5 automated steps)
python examples/ollama_demo.py

# Interactive chat -- long-term memory across sessions
python examples/ollama_chat.py
```

## Example Scripts

| File | Description | LLM Required? |
|------|-------------|---------------|
| `basic_usage.py` | Core MemoryFS API (write, search, read, update) | No |
| `openai_agent.py` | OpenAI function-calling loop | Yes (OpenAI) |
| `langchain_agent.py` | LangChain integration | Yes (OpenAI) |
| `agent_memory.py` | Multi-turn agent with persistent memory | Yes |
| `code_search.py` | Ingest and search a codebase | No |
| `infinite_context.py` | ContextMiddleware demo | Yes |
| `multi_agent.py` | Multiple agents sharing one LLMFS store | No |
| `ollama_demo.py` | Automated 5-step demo with Ollama | Yes (Ollama) |
| `ollama_chat.py` | Interactive chat with long-term memory | Yes (Ollama) |
| `ollama_autonomous_memory.py` | Model decides what to store autonomously | Yes (Ollama) |
| `ollama_context_overflow_test.py` | Stress-test context window overflow | Yes (Ollama) |

## Options

```bash
python examples/ollama_demo.py --model mistral
python examples/ollama_demo.py --model llama3.2 --store /tmp/my_store

python examples/ollama_chat.py --model mistral
python examples/ollama_chat.py --store ~/.my_llmfs   # persists across runs
```

## Chat Slash Commands

When using `ollama_chat.py`:

| Command | Description |
|---------|-------------|
| `/list` | Show all stored memories |
| `/search <query>` | Semantic search |
| `/forget <path>` | Delete a memory |
| `/status` | Storage stats |
| `/clear` | Wipe all memories |
| `/quit` | Exit |

## Basic Usage Example

```python
from llmfs import MemoryFS

mem = MemoryFS()

# Store a few memories
mem.write("/knowledge/db",     "We use PostgreSQL 15 with TimescaleDB extension")
mem.write("/knowledge/auth",   "JWT tokens use HS256, expire in 1 hour")
mem.write("/knowledge/stack",  "Backend: FastAPI + SQLAlchemy. Frontend: Next.js 14")

# Search
results = mem.search("database technology", k=3)
for r in results:
    print(f"[{r.score:.2f}] {r.path}: {r.chunk_text[:80]}")

# Read with a focused query
obj = mem.read("/knowledge/auth", query="what algorithm is used")
print(obj.content)

# Update
mem.update("/knowledge/auth", append="Refresh tokens last 30 days.")

# Link related memories
mem.relate("/knowledge/auth", "/knowledge/db", relationship="related_to")
```

## Multi-Agent Shared Memory

```python
from llmfs import MemoryFS

# Both agents share the same store
shared_mem = MemoryFS(path="/tmp/shared-project")

def planner_agent(task: str) -> str:
    plan = f"Plan for '{task}': 1. Analyze, 2. Design, 3. Implement"
    shared_mem.write(
        f"/session/plans/{task.replace(' ', '_')}",
        plan,
        layer="session",
        tags=["plan", "planner"],
    )
    return plan

def executor_agent(task: str) -> str:
    plans = shared_mem.search(f"plan for {task}", layer="session", k=3)
    knowledge = shared_mem.search(f"{task} patterns", layer="knowledge", k=5)
    context = "\n".join([r.chunk_text for r in plans + knowledge])
    return f"Executing with context:\n{context[:500]}..."

# Agents collaborate via shared memory
plan = planner_agent("build user authentication")
result = executor_agent("build user authentication")
```
