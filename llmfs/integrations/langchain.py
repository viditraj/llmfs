"""
LangChain integration for LLMFS.

Provides two drop-in memory classes that back LangChain conversations with
LLMFS persistent storage:

:class:`LLMFSChatMemory`
    Implements ``BaseChatMessageHistory``.  Stores every message as a
    ``/session/chat/<timestamp>`` memory and rehydrates them on demand.
    Drop-in replacement for ``ChatMessageHistory``.

:class:`LLMFSRetrieverMemory`
    Implements ``BaseMemory``.  On every chain call it semantically searches
    LLMFS for context relevant to the current input and injects it as a
    ``memory`` variable.  Saves each input/output pair as a turn.

**Install:** ``pip install llmfs[langchain]``

Example::

    from langchain.chains import ConversationChain
    from langchain.chat_models import ChatOpenAI
    from llmfs.integrations.langchain import LLMFSChatMemory

    memory = LLMFSChatMemory(memory_path="~/.llmfs")
    chain = ConversationChain(llm=ChatOpenAI(), memory=memory)
    chain.predict(input="What was the auth bug we discussed?")

Note: When ``langchain`` is not installed the classes are still importable
but will raise ``ImportError`` on instantiation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmfs.core.filesystem import MemoryFS

__all__ = ["LLMFSChatMemory", "LLMFSRetrieverMemory"]

logger = logging.getLogger(__name__)


def _require_langchain() -> None:
    try:
        import langchain  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "langchain is required for this integration. "
            "Install it with: pip install llmfs[langchain]"
        ) from exc


def _get_mem(memory_path: str) -> MemoryFS:
    from llmfs import MemoryFS
    return MemoryFS(path=memory_path)


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


# ── LLMFSChatMemory ───────────────────────────────────────────────────────────


class LLMFSChatMemory:
    """LangChain ``BaseChatMessageHistory`` backed by LLMFS.

    Stores every chat message as a separate LLMFS memory under
    ``/session/chat/<timestamp>``, tagged with the message role.

    Args:
        memory_path: Path to the LLMFS storage directory.
        session_prefix: Path prefix to store messages under.  Defaults to
            ``/session/chat``.
        mem: Optional pre-existing :class:`~llmfs.core.filesystem.MemoryFS`
            instance (skips creating a new one).

    Example::

        from llmfs.integrations.langchain import LLMFSChatMemory
        mem_obj = LLMFSChatMemory(memory_path="/tmp/llmfs")
        mem_obj.add_user_message("Hello")
        mem_obj.add_ai_message("Hi there!")
        print(len(mem_obj.messages))  # 2
    """

    def __init__(
        self,
        memory_path: str = "~/.llmfs",
        *,
        session_prefix: str = "/session/chat",
        mem: MemoryFS | None = None,
    ) -> None:
        _require_langchain()
        self._mem = mem or _get_mem(memory_path)
        self._prefix = session_prefix.rstrip("/")

    # ── BaseChatMessageHistory interface ──────────────────────────────────────

    @property
    def messages(self) -> list[Any]:
        """Return all stored messages as LangChain message objects."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        objs = self._mem.list(self._prefix, recursive=True)
        objs.sort(key=lambda o: o.metadata.created_at or "")

        result: list[Any] = []
        for obj in objs:
            role = "human"
            for tag in obj.tags:
                if tag in ("human", "user"):
                    role = "human"
                    break
                if tag in ("ai", "assistant"):
                    role = "ai"
                    break
                if tag == "system":
                    role = "system"
                    break

            if role == "human":
                result.append(HumanMessage(content=obj.content))
            elif role == "ai":
                result.append(AIMessage(content=obj.content))
            else:
                result.append(SystemMessage(content=obj.content))

        return result

    def add_message(self, message: Any) -> None:
        """Store a LangChain message object.

        Args:
            message: Any LangChain message (``HumanMessage``, ``AIMessage``, etc.).
        """
        # Detect role from class name
        cls_name = type(message).__name__.lower()
        if "human" in cls_name or "user" in cls_name:
            role = "human"
        elif "ai" in cls_name or "assistant" in cls_name:
            role = "ai"
        elif "system" in cls_name:
            role = "system"
        else:
            role = "human"

        path = f"{self._prefix}/{_ts()}"
        content = message.content if hasattr(message, "content") else str(message)
        self._mem.write(path, content, layer="session", tags=[role], source="langchain")

    def add_user_message(self, message: str) -> None:
        """Convenience: store a human/user message."""
        path = f"{self._prefix}/{_ts()}"
        self._mem.write(path, message, layer="session", tags=["human"], source="langchain")

    def add_ai_message(self, message: str) -> None:
        """Convenience: store an AI/assistant message."""
        path = f"{self._prefix}/{_ts()}"
        self._mem.write(path, message, layer="session", tags=["ai"], source="langchain")

    def clear(self) -> None:
        """Delete all messages in this session."""
        self._mem.forget(layer="session")

    def __len__(self) -> int:
        return len(self._mem.list(self._prefix, recursive=True))


# ── LLMFSRetrieverMemory ──────────────────────────────────────────────────────


class LLMFSRetrieverMemory:
    """LangChain ``BaseMemory`` backed by LLMFS semantic retrieval.

    On every chain call it:

    1. Searches LLMFS for memories semantically relevant to the current input
       and injects them as the ``memory_key`` variable.
    2. Saves the input/output pair to LLMFS for future retrieval.

    Args:
        memory_path: Path to the LLMFS storage directory.
        memory_key: The chain variable name to inject context into.
            Defaults to ``"memory"``.
        search_k: Number of memories to retrieve on each call.
        layer: Layer to search within.  ``None`` means search all layers.
        mem: Optional pre-existing :class:`~llmfs.core.filesystem.MemoryFS`.
        input_key: Which chain input key to use as the search query.
            Defaults to ``"input"``.
        session_prefix: Path prefix for saving conversation turns.

    Example::

        from langchain.chains import ConversationChain
        from llmfs.integrations.langchain import LLMFSRetrieverMemory

        memory = LLMFSRetrieverMemory(memory_path="/tmp/llmfs", search_k=3)
        chain = ConversationChain(llm=llm, memory=memory)
    """

    def __init__(
        self,
        memory_path: str = "~/.llmfs",
        *,
        memory_key: str = "memory",
        search_k: int = 3,
        layer: str | None = None,
        mem: MemoryFS | None = None,
        input_key: str = "input",
        session_prefix: str = "/session/turns",
    ) -> None:
        _require_langchain()
        self._mem = mem or _get_mem(memory_path)
        self._memory_key = memory_key
        self._search_k = search_k
        self._layer = layer
        self._input_key = input_key
        self._prefix = session_prefix.rstrip("/")

    # ── BaseMemory interface ───────────────────────────────────────────────────

    @property
    def memory_variables(self) -> list[str]:
        """LangChain requires this to know which variables we inject."""
        return [self._memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Search LLMFS and return retrieved context.

        Args:
            inputs: Chain inputs dict.  The value at ``input_key`` is used as
                the search query.

        Returns:
            Dict mapping ``memory_key`` to a formatted string of retrieved
            memories.
        """
        query = inputs.get(self._input_key, "") or ""
        if not query.strip():
            return {self._memory_key: ""}

        results = self._mem.search(query, k=self._search_k, layer=self._layer)
        if not results:
            return {self._memory_key: ""}

        lines = ["## Relevant memories from LLMFS:"]
        for r in results:
            snippet = r.chunk_text[:300].replace("\n", " ")
            lines.append(f"- [{r.path}] (score={r.score:.2f}): {snippet}")

        return {self._memory_key: "\n".join(lines)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Save the current input/output turn to LLMFS.

        Args:
            inputs: Chain inputs dict.
            outputs: Chain outputs dict.
        """
        human_text = inputs.get(self._input_key, "") or ""
        # Prefer "response" key, then "output", then fallback to str only if
        # neither key is present at all.
        if "response" in outputs:
            ai_text = outputs["response"] or ""
        elif "output" in outputs:
            ai_text = outputs["output"] or ""
        else:
            ai_text = str(outputs)

        ts = _ts()
        if human_text:
            self._mem.write(
                f"{self._prefix}/{ts}_human",
                human_text,
                layer="session",
                tags=["human", "turn"],
                source="langchain",
            )
        if ai_text:
            self._mem.write(
                f"{self._prefix}/{ts}_ai",
                ai_text,
                layer="session",
                tags=["ai", "turn"],
                source="langchain",
            )

    def clear(self) -> None:
        """Clear the session layer."""
        self._mem.forget(layer="session")
