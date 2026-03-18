"""
Custom exceptions for LLMFS.

All public-facing errors raised by MemoryFS and its subsystems.
"""
from __future__ import annotations

__all__ = [
    "LLMFSError",
    "MemoryNotFoundError",
    "MemoryWriteError",
    "MemoryDeleteError",
    "EmbedderError",
    "StorageError",
    "ChunkerError",
    "MQLParseError",
    "MQLExecutionError",
    "ConfigError",
]


class LLMFSError(Exception):
    """Base class for all LLMFS errors."""


class MemoryNotFoundError(LLMFSError):
    """Raised when a memory path does not exist."""

    def __init__(self, path: str) -> None:
        self.path = path
        super().__init__(f"Memory not found: {path!r}")


class MemoryWriteError(LLMFSError):
    """Raised when writing a memory fails."""

    def __init__(self, path: str, reason: str = "") -> None:
        self.path = path
        msg = f"Failed to write memory at {path!r}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class MemoryDeleteError(LLMFSError):
    """Raised when deleting a memory fails."""

    def __init__(self, path: str, reason: str = "") -> None:
        self.path = path
        msg = f"Failed to delete memory at {path!r}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class EmbedderError(LLMFSError):
    """Raised when the embedding model fails."""

    def __init__(self, reason: str = "") -> None:
        msg = "Embedder error"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class StorageError(LLMFSError):
    """Raised when the storage backend (SQLite or ChromaDB) fails."""

    def __init__(self, reason: str = "") -> None:
        msg = "Storage error"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ChunkerError(LLMFSError):
    """Raised when content chunking fails."""

    def __init__(self, reason: str = "") -> None:
        msg = "Chunker error"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class MQLParseError(LLMFSError):
    """Raised when an MQL query cannot be parsed."""

    def __init__(self, query: str, reason: str = "", position: int = -1) -> None:
        self.query = query
        self.position = position
        msg = "MQL parse error"
        if position >= 0:
            msg += f" at position {position}"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class MQLExecutionError(LLMFSError):
    """Raised when an MQL query fails during execution."""

    def __init__(self, reason: str = "") -> None:
        msg = "MQL execution error"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class ConfigError(LLMFSError):
    """Raised for invalid or missing configuration."""

    def __init__(self, reason: str = "") -> None:
        msg = "Configuration error"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)
