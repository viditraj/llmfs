"""
LLMFS — filesystem-metaphor memory layer for LLMs and AI agents.

Public API::

    from llmfs import MemoryFS

    mem = MemoryFS()
    mem.write("/projects/auth", content="JWT bug at line 45")
    results = mem.search("authentication")
    obj = mem.read("/projects/auth")
"""
from llmfs.core.async_fs import AsyncMemoryFS
from llmfs.core.exceptions import (
    EmbedderError,
    LLMFSError,
    MemoryDeleteError,
    MemoryNotFoundError,
    MemoryWriteError,
    StorageError,
)
from llmfs.core.filesystem import MemoryFS
from llmfs.core.memory_layers import MemoryLayer
from llmfs.core.memory_object import MemoryObject, SearchResult

__all__ = [
    "MemoryFS",
    "AsyncMemoryFS",
    "MemoryObject",
    "SearchResult",
    "MemoryLayer",
    "LLMFSError",
    "MemoryNotFoundError",
    "MemoryWriteError",
    "MemoryDeleteError",
    "EmbedderError",
    "StorageError",
]

__version__ = "0.1.11"
