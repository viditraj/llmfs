"""
MemoryObject — the core data structure representing a single memory in LLMFS.

Every piece of knowledge stored in LLMFS is a MemoryObject. It holds the raw
content, its chunked + embedded form, hierarchical summaries, rich metadata,
and graph relationships to other memories.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

__all__ = ["Chunk", "Summaries", "MemoryMetadata", "Relationship", "MemoryObject", "SearchResult"]


# ── Value objects ────────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A single text chunk of a memory, backed by an embedding in ChromaDB."""

    index: int
    text: str
    start_offset: int = 0
    end_offset: int = 0
    embedding_id: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "embedding_id": self.embedding_id,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Chunk:
        return cls(
            index=d["index"],
            text=d["text"],
            start_offset=d.get("start_offset", 0),
            end_offset=d.get("end_offset", 0),
            embedding_id=d.get("embedding_id", ""),
            summary=d.get("summary", ""),
        )


@dataclass
class Summaries:
    """Hierarchical extractive summaries of the memory content."""

    level_1: list[str] = field(default_factory=list)  # per-chunk summaries
    level_2: str = ""  # document-level summary

    def to_dict(self) -> dict[str, Any]:
        return {"level_1": self.level_1, "level_2": self.level_2}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Summaries:
        return cls(level_1=d.get("level_1", []), level_2=d.get("level_2", ""))


@dataclass
class MemoryMetadata:
    """Timestamps, tags, TTL and provenance for a memory."""

    created_at: str = ""
    modified_at: str = ""
    accessed_at: str = ""
    tags: list[str] = field(default_factory=list)
    ttl: int | None = None          # minutes until expiry; None = permanent
    source: str = "manual"          # manual | agent | mcp | cli

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "accessed_at": self.accessed_at,
            "tags": self.tags,
            "ttl": self.ttl,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryMetadata:
        return cls(
            created_at=d.get("created_at", ""),
            modified_at=d.get("modified_at", ""),
            accessed_at=d.get("accessed_at", ""),
            tags=d.get("tags", []),
            ttl=d.get("ttl"),
            source=d.get("source", "manual"),
        )


@dataclass
class Relationship:
    """A directed edge in the memory knowledge graph."""

    target: str                     # target memory path
    type: str                       # related_to | follows | caused_by | contradicts
    strength: float = 0.8           # 0.0–1.0

    def to_dict(self) -> dict[str, Any]:
        return {"target": self.target, "type": self.type, "strength": self.strength}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Relationship:
        return cls(
            target=d["target"],
            type=d["type"],
            strength=d.get("strength", 0.8),
        )


# ── Main dataclass ───────────────────────────────────────────────────────────


@dataclass
class MemoryObject:
    """The central data structure for a single LLMFS memory.

    A MemoryObject is identified by its filesystem-style path (e.g.
    ``/projects/auth/debug``). It stores the raw content, chunked embeddings,
    multi-level summaries, metadata, and outgoing relationships.

    Example::

        obj = MemoryObject(
            id="abc123",
            path="/projects/auth/debug",
            content="JWT token expiry bug found at auth.py:45",
            layer="events",
        )
        d = obj.to_dict()
        restored = MemoryObject.from_dict(d)
    """

    id: str
    path: str
    content: str
    layer: str                                          # short_term|session|knowledge|events
    chunks: list[Chunk] = field(default_factory=list)
    summaries: Summaries = field(default_factory=Summaries)
    metadata: MemoryMetadata = field(default_factory=MemoryMetadata)
    relationships: list[Relationship] = field(default_factory=list)

    # ── Computed properties ──────────────────────────────────────────────────

    @property
    def content_hash(self) -> str:
        """SHA-256 hex digest of the UTF-8 encoded content."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def tags(self) -> list[str]:
        """Shortcut to ``metadata.tags``."""
        return self.metadata.tags

    @property
    def name(self) -> str:
        """Last path component, e.g. ``debug`` for ``/projects/auth/debug``."""
        return self.path.rstrip("/").rsplit("/", 1)[-1]

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "path": self.path,
            "content": self.content,
            "layer": self.layer,
            "chunks": [c.to_dict() for c in self.chunks],
            "summaries": self.summaries.to_dict(),
            "metadata": self.metadata.to_dict(),
            "relationships": [r.to_dict() for r in self.relationships],
        }

    def to_json(self) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MemoryObject:
        """Deserialize from a dictionary produced by :meth:`to_dict`."""
        return cls(
            id=d["id"],
            path=d["path"],
            content=d["content"],
            layer=d["layer"],
            chunks=[Chunk.from_dict(c) for c in d.get("chunks", [])],
            summaries=Summaries.from_dict(d.get("summaries", {})),
            metadata=MemoryMetadata.from_dict(d.get("metadata", {})),
            relationships=[Relationship.from_dict(r) for r in d.get("relationships", [])],
        )

    @classmethod
    def from_json(cls, s: str) -> MemoryObject:
        """Deserialize from a JSON string."""
        return cls.from_dict(json.loads(s))

    # ── Validation ───────────────────────────────────────────────────────────

    def validate(self) -> None:
        """Raise ``ValueError`` if the object is in an invalid state.

        Raises:
            ValueError: If ``path`` doesn't start with ``/``, ``layer`` is
                unknown, or ``id`` is empty.
        """
        from llmfs.core.memory_layers import VALID_LAYERS  # avoid circular at module level

        if not self.path.startswith("/"):
            raise ValueError(f"Memory path must start with '/': {self.path!r}")
        if not self.id:
            raise ValueError("Memory id must not be empty")
        if self.layer not in VALID_LAYERS:
            raise ValueError(
                f"Unknown layer {self.layer!r}. Valid: {sorted(VALID_LAYERS)}"
            )

    def __repr__(self) -> str:
        return (
            f"MemoryObject(id={self.id!r}, path={self.path!r}, "
            f"layer={self.layer!r}, chunks={len(self.chunks)})"
        )


# ── Search result ─────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single result returned by :meth:`MemoryFS.search`.

    Attributes:
        path: The memory path.
        content: Full memory content (or best-matching chunk text).
        score: Relevance score in [0, 1]; higher is better.
        metadata: The memory's metadata dict.
        tags: Convenience alias for ``metadata["tags"]``.
        chunk_text: The specific chunk that matched (may differ from full content).
    """

    path: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    chunk_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "tags": self.tags,
            "chunk_text": self.chunk_text,
        }


def _utcnow() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
