"""
AsyncMemoryFS — async wrapper for MemoryFS.

Wraps every :class:`~llmfs.core.filesystem.MemoryFS` method with
:func:`asyncio.to_thread` so that async agent frameworks (LangGraph,
CrewAI, FastAPI endpoints) can call LLMFS without blocking their event
loop.

All arguments and return types mirror :class:`MemoryFS` exactly.

Example::

    from llmfs.core.async_fs import AsyncMemoryFS

    async def main():
        mem = AsyncMemoryFS(path="/tmp/test")
        await mem.write("/k/hello", "world")
        results = await mem.search("hello", k=3)
        obj = await mem.read("/k/hello")
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from llmfs.core.filesystem import MemoryFS
from llmfs.core.memory_object import MemoryObject, SearchResult
from llmfs.embeddings.base import EmbedderBase

__all__ = ["AsyncMemoryFS"]

# Alias so type annotations inside AsyncMemoryFS still refer to the builtin
# ``list`` even though the class defines a method with the same name.
_list = list


class AsyncMemoryFS:
    """Async wrapper around :class:`~llmfs.core.filesystem.MemoryFS`.

    Every public method delegates to the synchronous ``MemoryFS`` via
    :func:`asyncio.to_thread`, making it safe to use in ``async`` contexts
    without blocking the event loop.

    The constructor arguments are identical to :class:`MemoryFS`.

    Args:
        path: Directory where LLMFS stores its data.
        embedder: Optional embedder instance.
        auto_link: Enable auto-linking on write.
        auto_link_threshold: Similarity threshold for auto-link.
        auto_link_k: Max auto-link edges per write.

    Example::

        mem = AsyncMemoryFS()
        await mem.write("/knowledge/auth", "JWT tokens expire in 1h")
        results = await mem.search("authentication")
    """

    def __init__(
        self,
        path: str | Path | None = None,
        embedder: EmbedderBase | None = None,
        *,
        auto_link: bool = True,
        auto_link_threshold: float = 0.75,
        auto_link_k: int = 3,
    ) -> None:
        self._sync = MemoryFS(
            path=path,
            embedder=embedder,
            auto_link=auto_link,
            auto_link_threshold=auto_link_threshold,
            auto_link_k=auto_link_k,
        )

    @property
    def sync(self) -> MemoryFS:
        """Access the underlying synchronous :class:`MemoryFS` instance."""
        return self._sync

    # ── Warmup ────────────────────────────────────────────────────────────────

    async def warmup(self) -> None:
        """Eagerly load the embedder model and VectorStore.

        Call at process start so the first ``write`` / ``search`` is fast.
        """
        await asyncio.to_thread(self._sync.warmup)

    # ── Write ─────────────────────────────────────────────────────────────────

    async def write(
        self,
        path: str,
        content: str,
        layer: str = "knowledge",
        tags: list[str] | None = None,
        ttl_minutes: int | None = None,
        source: str = "manual",
        content_type: str | None = None,
    ) -> MemoryObject:
        """Async version of :meth:`MemoryFS.write`."""
        return await asyncio.to_thread(
            self._sync.write,
            path, content,
            layer=layer, tags=tags, ttl_minutes=ttl_minutes,
            source=source, content_type=content_type,
        )

    # ── Read ──────────────────────────────────────────────────────────────────

    async def read(self, path: str, query: str | None = None) -> MemoryObject:
        """Async version of :meth:`MemoryFS.read`."""
        return await asyncio.to_thread(self._sync.read, path, query)

    # ── Search ────────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        layer: str | None = None,
        tags: list[str] | None = None,
        path_prefix: str | None = None,
        time_range: str | None = None,
        k: int = 5,
    ) -> list[SearchResult]:
        """Async version of :meth:`MemoryFS.search`."""
        return await asyncio.to_thread(
            self._sync.search,
            query,
            layer=layer, tags=tags, path_prefix=path_prefix,
            time_range=time_range, k=k,
        )

    # ── Update ────────────────────────────────────────────────────────────────

    async def update(
        self,
        path: str,
        *,
        content: str | None = None,
        append: str | None = None,
        tags_add: list[str] | None = None,
        tags_remove: list[str] | None = None,
    ) -> MemoryObject:
        """Async version of :meth:`MemoryFS.update`."""
        return await asyncio.to_thread(
            self._sync.update,
            path,
            content=content, append=append,
            tags_add=tags_add, tags_remove=tags_remove,
        )

    # ── Forget ────────────────────────────────────────────────────────────────

    async def forget(
        self,
        path: str | None = None,
        *,
        layer: str | None = None,
        older_than: str | None = None,
    ) -> dict[str, Any]:
        """Async version of :meth:`MemoryFS.forget`."""
        return await asyncio.to_thread(
            self._sync.forget, path, layer=layer, older_than=older_than,
        )

    # ── Relate ────────────────────────────────────────────────────────────────

    async def relate(
        self,
        source: str,
        target: str,
        relationship: str,
        strength: float = 0.8,
    ) -> dict[str, Any]:
        """Async version of :meth:`MemoryFS.relate`."""
        return await asyncio.to_thread(
            self._sync.relate, source, target, relationship, strength=strength,
        )

    # ── List ──────────────────────────────────────────────────────────────────

    async def list(
        self,
        path_prefix: str = "/",
        *,
        layer: str | None = None,
        recursive: bool = True,
    ) -> _list[MemoryObject]:
        """Async version of :meth:`MemoryFS.list`."""
        return await asyncio.to_thread(
            self._sync.list, path_prefix, layer=layer, recursive=recursive,
        )

    # ── Status ────────────────────────────────────────────────────────────────

    async def status(self) -> dict[str, Any]:
        """Async version of :meth:`MemoryFS.status`."""
        return await asyncio.to_thread(self._sync.status)

    # ── Query ─────────────────────────────────────────────────────────────────

    async def query(self, mql: str) -> _list[Any]:
        """Async version of :meth:`MemoryFS.query`."""
        return await asyncio.to_thread(self._sync.query, mql)

    # ── GC ────────────────────────────────────────────────────────────────────

    async def gc(self) -> dict[str, Any]:
        """Async version of :meth:`MemoryFS.gc`."""
        return await asyncio.to_thread(self._sync.gc)

    def __repr__(self) -> str:
        return f"AsyncMemoryFS(sync={self._sync!r})"
