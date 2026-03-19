# Infinite Context — ContextMiddleware

The `ContextMiddleware` is LLMFS's flagship feature. Wrap any agent with two lines and get effectively unlimited context with zero information loss.

## The Problem

```
Turn 35: Context window hits 128k tokens
         |
  Standard approach: lossy summarization
  128k tokens -> 5k tokens = 94% information LOST FOREVER
         |
Turn 36: "What was the exact error at auth.py line 45?"
  LLM: "I don't have that detail anymore."  <-- failure
```

## The LLMFS Solution

LLMFS works like virtual memory. Old turns are evicted from the context window and stored in LLMFS at full fidelity. A compact memory index (~2k tokens) stays in the system prompt, listing what has been stored and where. When the LLM needs something, it calls `memory_read` or `memory_search` to page it back in.

```
Context Window: 128k tokens
+----------------+--------------------+----------------------------------+
| System         | Memory Index       | Active Conversation              |
| Prompt         | (~2k tokens)       | (recent 5-10 turns)             |
| (~1k)          | lists paths of     | (~20-80k tokens)                |
|                | evicted turns      |                                  |
+----------------+--------------------+----------------------------------+
                       |
               +-------v--------+
               |    LLMFS       |  500k+ tokens stored, zero lost
               |                |  Full fidelity, semantically indexed
               +----------------+
```

## Drop-In Usage

```python
from llmfs import MemoryFS
from llmfs.context import ContextMiddleware

# Wrap your existing agent with 2 lines
agent = YourExistingAgent(model="gpt-4o")
agent = ContextMiddleware(agent, memory=MemoryFS())

# Now every call transparently manages context:
# 1. Intercepts every turn (before + after)
# 2. Scores importance of each message
# 3. Auto-evicts at 70% capacity, targets 50%
# 4. Extracts artifacts (code, errors, file refs) before eviction
# 5. Rebuilds the memory index after eviction
# 6. Injects the index into the system prompt
# 7. Provides memory_search / memory_read tools to the LLM
response = agent.chat("What was the exact error from turn 15?")
```

## Importance Scoring

The middleware scores each turn before evicting the lowest-importance ones:

| Signal | Score Boost |
|--------|------------|
| Contains a code block (` ``` `) | +0.20 |
| Contains error / traceback | +0.20 |
| Contains decision keyword (`decided`, `plan`, `must`) | +0.15 |
| Role = `user` (user intent is high-value) | +0.10 |
| Very recent turn (last 3) | +0.15 |
| Very short / conversational filler | -0.20 |

## Artifact Extraction

Before a turn is evicted, the middleware automatically extracts and stores structured artifacts at dedicated sub-paths:

| Artifact | Stored At | Tags |
|----------|-----------|------|
| Code blocks | `/session/{id}/code/turn_{n}_{i}` | `["code", "<lang>"]` |
| Stack traces / errors | `/session/{id}/errors/turn_{n}` | `["error"]` |
| File paths mentioned | `/session/{id}/files/turn_{n}` | `["file_references"]` |
| Decisions | `/session/{id}/decisions/turn_{n}` | `["decision"]` |
| Full turn (always) | `/session/{id}/turns/{n}` | -- |

## Memory Index

The memory index is regenerated after each eviction cycle and injected into the system prompt:

```
## LLMFS Memory Index
You have the following memories (use memory_read / memory_search to retrieve):

- [/session/abc/turns/1]       (turn 1, 10:30) [user]      -- User asked to fix auth module bug
- [/session/abc/turns/2]       (turn 2, 10:31) [assistant]  -- Found JWT expiry at auth.py:45
- [/session/abc/code/turn_2_0] (turn 2, 10:31) [code:py]    -- Fixed auth.py token refresh logic
- [/session/abc/errors/turn_3] (turn 3, 10:32) [error]      -- TypeError: NoneType at auth.py:45
- [/session/abc/turns/5]       (turn 5, 10:35) [user]       -- Asked to also fix refresh endpoint
... (12 more -- use memory_search "topic" to find relevant ones)
```

## ContextManager API

For lower-level control:

```python
from llmfs import MemoryFS
from llmfs.context.manager import ContextManager

mem = MemoryFS()
ctx = ContextManager(
    mem=mem,
    max_tokens=128000,
    evict_at=0.70,            # start evicting at 70% capacity
    target_after_evict=0.50,  # evict down to 50%
)

# Track a new turn
ctx.on_new_turn(role="user", content="Fix the JWT bug", tokens=12)
ctx.on_new_turn(role="assistant", content="Found the issue at auth.py:45", tokens=45)

# Get the current memory index for system prompt injection
index = ctx.get_system_prompt_addon()

# Get active (in-context) turns
turns = ctx.get_active_turns()

# Reset for a new session
ctx.reset_session()
```

## Full Example with OpenAI

```python
import openai
from llmfs import MemoryFS
from llmfs.context import ContextMiddleware

mem = MemoryFS()
client = openai.OpenAI()
agent = ContextMiddleware(client, memory=mem, max_tokens=128000)

conversation = []
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    conversation.append({"role": "user", "content": user_input})
    response = agent.chat(conversation)
    assistant_message = response.choices[0].message.content
    conversation.append({"role": "assistant", "content": assistant_message})
    print(f"Assistant: {assistant_message}")

# Session statistics
stats = agent.get_context_stats()
print(f"\nTurns evicted: {stats['evicted_turns']}")
print(f"Cache hits:    {stats['cache_hits']}")
print(f"Token usage:   {stats['current_tokens']} / {stats['max_tokens']}")
```
