# OpenAI Function Calling

LLMFS exports OpenAI-format tool definitions and a handler for seamless integration with the OpenAI API. Install with:

```bash
pip install "llmfs[openai]"
```

## Usage

```python
import openai
from llmfs import MemoryFS
from llmfs.integrations.openai_tools import LLMFS_TOOLS, LLMFSToolHandler

mem = MemoryFS()
handler = LLMFSToolHandler(mem)

messages = [
    {"role": "system", "content": "You are a helpful assistant with persistent memory."},
    {"role": "user",   "content": "Remember that our database is PostgreSQL 15."},
]

# Pass LLMFS tools alongside any other tools you use
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=LLMFS_TOOLS,
    tool_choice="auto",
)

# Handle all LLMFS tool calls in the response
tool_results = handler.handle_batch(response.choices[0].message.tool_calls)

# Append tool results and continue the conversation
for call, result in zip(response.choices[0].message.tool_calls, tool_results):
    messages.append({
        "role": "tool",
        "tool_call_id": call.id,
        "content": result,
    })
```

## What's in LLMFS_TOOLS

`LLMFS_TOOLS` is a plain Python list of JSON Schema dicts — pass it directly to any OpenAI-compatible API (OpenAI, Azure OpenAI, Groq, Together, etc.).

The tools match the MCP tools: `memory_write`, `memory_search`, `memory_read`, `memory_update`, `memory_forget`, `memory_relate`.

## Ollama Compatibility

LLMFS also works with Ollama via its OpenAI-compatible API. See the [examples](../examples.md) for complete Ollama demos.
