# LangChain Integration

LLMFS provides two drop-in LangChain memory adapters. Install with:

```bash
pip install "llmfs[langchain]"
```

## LLMFSChatMemory — Persistent Chat History

Stores every conversation turn in LLMFS. Memory persists across process restarts.

```python
from llmfs.integrations.langchain import LLMFSChatMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

memory = LLMFSChatMemory(memory_path="~/.llmfs")
chain = ConversationChain(llm=ChatOpenAI(model="gpt-4o"), memory=memory)

# Memory persists automatically — conversations survive process restarts
response = chain.predict(input="What was the JWT bug we discussed?")
```

## LLMFSRetrieverMemory — Semantic Context Injection

Semantically searches past conversations on every turn and injects the most relevant passages into the LLM's context:

```python
from llmfs.integrations.langchain import LLMFSRetrieverMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

memory = LLMFSRetrieverMemory(
    memory_path="~/.llmfs",
    search_k=5,                   # inject top-5 relevant memories
    layer="knowledge",
)

chain = ConversationChain(llm=ChatOpenAI(model="gpt-4o"), memory=memory)
```

## Compatibility

Both classes implement `BaseChatMessageHistory` / `BaseMemory` and work as drop-in replacements for LangChain's built-in memory classes. They work with any LangChain chain, agent, or runnable.
