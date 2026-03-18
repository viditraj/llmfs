"""
System prompt fragment for LLMFS MCP server.

The prompt tells the LLM when and how to use each LLMFS tool, the layer
hierarchy, path conventions, and provides concrete usage examples.  It is
injected automatically by the MCP server and can also be retrieved
programmatically.

Example::

    from llmfs.mcp.prompts import get_prompt, LLMFS_SYSTEM_PROMPT

    print(get_prompt())
    print(get_prompt(include_index=True, mem=mem, session_id="sess1"))
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmfs.core.filesystem import MemoryFS

__all__ = ["LLMFS_SYSTEM_PROMPT", "get_prompt"]


LLMFS_SYSTEM_PROMPT: str = """\
## LLMFS — AI Memory Filesystem

You have access to a persistent, searchable memory filesystem (LLMFS).
Use it to avoid losing information when the context window fills, and to
retrieve precise details from past interactions.

### Memory Layers

| Layer       | Purpose                                  | Default TTL  |
|-------------|------------------------------------------|--------------|
| short_term  | Scratch notes, tentative ideas           | 60 minutes   |
| session     | Current conversation turns & artifacts   | Session only |
| knowledge   | Stable facts, decisions, reference docs  | Permanent    |
| events      | Timestamped log entries, observations    | Permanent    |

### Path Naming Conventions

Use meaningful, hierarchical paths:

- `/knowledge/<project>/<topic>` — stable knowledge
- `/session/<session_id>/turns/<id>` — conversation turns
- `/session/<session_id>/code/turn_<id>_<i>` — code artifacts
- `/session/<session_id>/errors/turn_<id>` — errors / tracebacks
- `/session/<session_id>/decisions/turn_<id>` — decisions made
- `/events/<YYYY-MM-DD>/<event_slug>` — timestamped events

### When to Use Each Tool

**memory_write** — Store new information:
- User shares requirements, constraints, or preferences
- You produce code, plans, or decisions
- An error or traceback occurs that may be needed later

**memory_search** — Find relevant memories:
- You need context that may have been evicted
- User asks about something from earlier in the conversation
- You are about to repeat a search you may have done before

**memory_read** — Retrieve exact content:
- You know the path and need the full content
- Verifying details of a stored decision or code snippet

**memory_update** — Append or modify:
- New information extends an existing memory
- Tags need to be updated (e.g., mark as "resolved")

**memory_forget** — Clean up:
- Remove outdated or incorrect information
- Clear session layer at end of conversation

**memory_relate** — Link memories:
- Two memories are causally or topically related
- An error trace relates to a code file
- A decision relates to a requirements document

### Example Calls

Store a bug report:
```json
{"tool": "memory_write",
 "path": "/session/sess1/errors/turn_5",
 "content": "TypeError: 'NoneType' object is not subscriptable at auth.py:45",
 "layer": "session",
 "tags": ["error", "auth"]}
```

Search for related context:
```json
{"tool": "memory_search",
 "query": "auth.py TypeError",
 "layer": "session",
 "k": 3}
```

Read a specific memory:
```json
{"tool": "memory_read",
 "path": "/knowledge/auth/architecture"}
```

### Memory Index

When a memory index is shown below, it lists what is stored in LLMFS.
Use `memory_search "<topic>"` to retrieve memories by topic, or
`memory_read "<path>"` to retrieve a specific memory by exact path.
"""


def get_prompt(
    *,
    include_index: bool = False,
    mem: MemoryFS | None = None,
    session_id: str | None = None,
    max_entries: int = 50,
) -> str:
    """Return the full system prompt, optionally with a memory index appended.

    Args:
        include_index: If ``True``, append the current memory index for
            *session_id* from *mem*.
        mem: :class:`~llmfs.core.filesystem.MemoryFS` instance (required if
            ``include_index=True``).
        session_id: Session namespace for the index (required if
            ``include_index=True``).
        max_entries: Maximum entries in the memory index.

    Returns:
        Complete prompt string.
    """
    prompt = LLMFS_SYSTEM_PROMPT

    if include_index and mem is not None and session_id:
        from llmfs.context.index_builder import IndexBuilder
        builder = IndexBuilder(max_entries=max_entries)
        index = builder.build(session_id=session_id, mem=mem)
        prompt = f"{prompt}\n\n{index}"

    return prompt
