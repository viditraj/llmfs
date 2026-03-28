"""
MemoryFS — the main entry point for LLMFS.

Provides a filesystem-metaphor API for storing, retrieving, and searching
memories.  All state is persisted to a local directory (default ``~/.llmfs``).

Usage::

    from llmfs import MemoryFS

    mem = MemoryFS()
    mem.write("/projects/auth/debug", content="JWT expiry bug at line 45")
    results = mem.search("authentication bug")
    obj = mem.read("/projects/auth/debug")
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmfs.compression.chunker import AdaptiveChunker
from llmfs.compression.chunker import Chunk as RawChunk
from llmfs.compression.summarizer import ExtractiveSummarizer
from llmfs.core.exceptions import MemoryNotFoundError, MemoryWriteError, StorageError
from llmfs.core.memory_layers import VALID_LAYERS, ttl_expires_at
from llmfs.core.memory_object import (
    Chunk,
    MemoryMetadata,
    MemoryObject,
    Relationship,
    SearchResult,
    Summaries,
    _utcnow,
)
from llmfs.embeddings.base import EmbedderBase
from llmfs.storage.metadata_db import MetadataDB
from llmfs.storage.vector_store import VectorStore

__all__ = ["MemoryFS"]

logger = logging.getLogger(__name__)

_DEFAULT_BASE = Path.home() / ".llmfs"
_GC_INTERVAL_SECONDS = 60


class MemoryFS:
    """Filesystem-metaphor persistent memory for LLMs and AI agents.

    Args:
        path: Directory where LLMFS stores its data.  Created if it doesn't
            exist.  Defaults to ``~/.llmfs``.
        embedder: Embedder instance to use.  If ``None``, a
            :class:`~llmfs.embeddings.local.LocalEmbedder` is created lazily.
        auto_link: If ``True`` (the default), automatically create
            ``related_to`` graph edges to semantically similar memories on
            each :meth:`write` call.
        auto_link_threshold: Minimum cosine similarity to auto-create an
            edge.  Defaults to ``0.75``.
        auto_link_k: Maximum number of auto-link edges per write.
            Defaults to ``3``.

    Example::

        mem = MemoryFS(path="/tmp/myproject")
        mem.write("/knowledge/stack", content="We use PostgreSQL 15")
        results = mem.search("database", k=3)
        obj = mem.read("/knowledge/stack")
        mem.forget("/knowledge/stack")
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
        self._base = Path(path) if path else _DEFAULT_BASE
        self._base.mkdir(parents=True, exist_ok=True)

        self._db = MetadataDB(self._base / "metadata.db")
        self._vs: VectorStore | None = None  # lazy-loaded on first use
        self._vs_dir = self._base / "chroma"
        self._chunker = AdaptiveChunker()
        self._summarizer = ExtractiveSummarizer()
        self._embedder: EmbedderBase | None = embedder  # lazy if None

        self._auto_link = auto_link
        self._auto_link_threshold = auto_link_threshold
        self._auto_link_k = auto_link_k

        self._last_gc: float = 0.0
        self._gc_if_due()
        logger.info("MemoryFS ready at %s", self._base)

    # ── Embedder (lazy) ───────────────────────────────────────────────────────

    def _get_embedder(self) -> EmbedderBase:
        if self._embedder is None:
            from llmfs.embeddings.local import LocalEmbedder
            self._embedder = LocalEmbedder(cache_db=self._db)
        return self._embedder

    # ── VectorStore (lazy) ────────────────────────────────────────────────────

    def _get_vs(self) -> VectorStore:
        if self._vs is None:
            self._vs = VectorStore(self._vs_dir)
        return self._vs

    # ── GC ────────────────────────────────────────────────────────────────────

    def _gc_if_due(self) -> None:
        import time
        now = time.monotonic()
        if now - self._last_gc >= _GC_INTERVAL_SECONDS:
            self._db.expire_ttl()
            self._last_gc = now

    # ── Write ─────────────────────────────────────────────────────────────────

    def write(
        self,
        path: str,
        content: str,
        layer: str = "knowledge",
        tags: list[str] | None = None,
        ttl_minutes: int | None = None,
        source: str = "manual",
        content_type: str | None = None,
    ) -> MemoryObject:
        """Store content at *path*.

        If a memory already exists at *path* and the content hash is
        unchanged, the existing object is returned without re-embedding.

        Args:
            path: Filesystem-style path, must start with ``/``.
            content: Text to store.
            layer: One of ``short_term``, ``session``, ``knowledge``, ``events``.
            tags: Optional list of string tags.
            ttl_minutes: Minutes until auto-expiry.  ``None`` uses the layer
                default.
            source: Origin label (``manual``, ``agent``, ``mcp``, ``cli``).
            content_type: Hint for the chunker (``python``, ``markdown``, etc.).

        Returns:
            The stored :class:`~llmfs.core.memory_object.MemoryObject`.

        Raises:
            ValueError: If *path* doesn't start with ``/`` or *layer* is invalid.
            MemoryWriteError: If the write fails.
        """
        self._validate_path(path)
        self._validate_layer(layer)
        tags = tags or []
        self._gc_if_due()

        content_hash = hashlib.sha256(content.encode()).hexdigest()
        existing = self._db.get_file(path)

        if existing and existing.get("content_hash") == content_hash:
            # Content unchanged — update accessed_at and tags if they changed
            self._db.touch_accessed(path)
            if tags != existing.get("tags", []):
                self._db.set_tags(existing["id"], tags)
            refreshed = self._db.get_file(path)  # re-fetch with updated tags
            assert refreshed is not None
            obj = self._load_object(refreshed)
            return obj

        # Generate or reuse ID
        mem_id = existing["id"] if existing else str(uuid.uuid4())
        now = _utcnow()
        ttl_exp = ttl_expires_at(layer, ttl_minutes)

        # Chunk + embed
        raw_chunks: list[RawChunk] = self._chunker.chunk(content, content_type)
        texts = [c.text for c in raw_chunks]
        embeddings = self._get_embedder().embed_batch(texts)

        # Summarize (level 1 = per-chunk, level 2 = document)
        chunk_summaries, doc_summary = self._summarizer.summarize_all(content, texts)

        # Build Chunk objects with unique embedding IDs and chunk-level summaries
        chunk_objs: list[Chunk] = []
        for rc, _emb, chunk_sum in zip(raw_chunks, embeddings, chunk_summaries, strict=False):
            emb_id = f"{mem_id}_{rc.index}"
            chunk_objs.append(Chunk(
                index=rc.index,
                text=rc.text,
                start_offset=rc.start_offset,
                end_offset=rc.end_offset,
                embedding_id=emb_id,
                summary=chunk_sum,
            ))

        # Upsert into vector store
        try:
            self._get_vs().upsert_batch(
                embedding_ids=[c.embedding_id for c in chunk_objs],
                embeddings=embeddings,
                metadatas=[
                    {
                        "file_id": mem_id,
                        "path": path,
                        "layer": layer,
                        "chunk_index": c.index,
                        "tags": ",".join(tags),
                    }
                    for c in chunk_objs
                ],
                texts=texts,
            )
        except StorageError as exc:
            raise MemoryWriteError(path, str(exc)) from exc

        # Upsert into SQLite
        try:
            if existing:
                self._db.delete_chunks(mem_id)
                self._db.update_file(
                    path,
                    size=len(content.encode()),
                    modified_at=now,
                    content_hash=content_hash,
                    ttl_expires=ttl_exp or "",
                )
                self._db.set_tags(mem_id, tags)
            else:
                self._db.insert_file(
                    id=mem_id,
                    path=path,
                    name=path.rstrip("/").rsplit("/", 1)[-1],
                    layer=layer,
                    size=len(content.encode()),
                    created_at=now,
                    modified_at=now,
                    content_hash=content_hash,
                    ttl_expires=ttl_exp,
                    source=source,
                )
                self._db.set_tags(mem_id, tags)

            for c in chunk_objs:
                self._db.insert_chunk(
                    id=f"{mem_id}_chunk_{c.index}",
                    file_id=mem_id,
                    chunk_index=c.index,
                    start_offset=c.start_offset,
                    end_offset=c.end_offset,
                    text=c.text,
                    embedding_id=c.embedding_id,
                    summary=c.summary,
                )
        except StorageError as exc:
            raise MemoryWriteError(path, str(exc)) from exc

        # Invalidate search cache for this path
        self._db.cache_invalidate(path)

        mem_obj = MemoryObject(
            id=mem_id,
            path=path,
            content=content,
            layer=layer,
            chunks=chunk_objs,
            summaries=Summaries(
                level_1=[c.summary for c in chunk_objs],
                level_2=doc_summary,
            ),
            metadata=MemoryMetadata(
                created_at=existing["created_at"] if existing else now,
                modified_at=now,
                tags=tags,
                ttl=ttl_minutes,
                source=source,
            ),
        )
        logger.debug("write: path=%s chunks=%d summary_len=%d", path, len(chunk_objs), len(doc_summary))

        # Auto-link: create edges to semantically similar memories
        if self._auto_link and embeddings:
            self._auto_link_memory(mem_id, path, embeddings[0])

        return mem_obj

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self, path: str, query: str | None = None) -> MemoryObject:
        """Read a memory by exact path.

        Args:
            path: Exact path to retrieve.
            query: If provided, only the most relevant chunks are included in
                the returned ``content`` (focused read).

        Returns:
            The :class:`~llmfs.core.memory_object.MemoryObject`.

        Raises:
            MemoryNotFoundError: If *path* doesn't exist.
        """
        self._validate_path(path)
        row = self._db.get_file(path)
        if not row:
            raise MemoryNotFoundError(path)

        self._db.touch_accessed(path)
        obj = self._load_object(row)

        if query and obj.chunks:
            # Focused read: retrieve only matching chunks
            q_emb = self._get_embedder().embed(query)
            results = self._get_vs().query(q_emb, k=3, where={"path": {"$eq": path}})
            if results:
                obj.content = "\n\n".join(r["text"] for r in results)

        return obj

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        layer: str | None = None,
        tags: list[str] | None = None,
        path_prefix: str | None = None,
        time_range: str | None = None,
        k: int = 5,
    ) -> list[SearchResult]:
        """Hybrid semantic + BM25 search across stored memories.

        Combines dense vector similarity from ChromaDB with BM25 keyword
        scoring from SQLite FTS5, fused via Reciprocal Rank Fusion for the
        best of both worlds.

        Args:
            query: Natural-language search query.
            layer: Filter to a specific layer.
            tags: Filter to memories having all supplied tags.
            path_prefix: Restrict to paths starting with this prefix.
            time_range: Human time string, e.g. ``"last 30 minutes"``,
                ``"today"``, ``"last 7 days"``.
            k: Number of results to return.

        Returns:
            List of :class:`~llmfs.core.memory_object.SearchResult` ordered
            by descending relevance score.
        """
        cache_key = hashlib.sha256(
            json.dumps(
                {"q": query, "layer": layer, "tags": tags,
                 "prefix": path_prefix, "tr": time_range, "k": k},
                sort_keys=True,
            ).encode()
        ).hexdigest()

        cached = self._db.cache_get(cache_key)
        if cached is not None:
            logger.debug("search cache hit: %s", cache_key[:8])
            return [SearchResult(**r) for r in cached]

        created_after = _parse_time_range(time_range) if time_range else None

        # ── Dense (semantic) retrieval ────────────────────────────────────
        q_emb = self._get_embedder().embed(query)
        where = self._build_where(layer=layer, path_prefix=path_prefix)
        raw_dense = self._get_vs().query(q_emb, k=k * 3, where=where or None)

        dense_results = self._raw_hits_to_results(
            raw_dense, tags=tags, created_after=created_after,
        )

        # ── BM25 (keyword) retrieval ──────────────────────────────────────
        bm25_results = self._bm25_search(
            query, layer=layer, path_prefix=path_prefix,
            tags=tags, created_after=created_after, limit=k * 3,
        )

        # ── Fuse via Reciprocal Rank Fusion ───────────────────────────────
        from llmfs.retrieval.ranker import Ranker
        ranker = Ranker()
        results = ranker.fuse(dense_results, bm25_results, top_k=k)

        self._db.cache_set(cache_key, [r.to_dict() for r in results])
        return results

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        path: str,
        *,
        content: str | None = None,
        append: str | None = None,
        tags_add: list[str] | None = None,
        tags_remove: list[str] | None = None,
    ) -> MemoryObject:
        """Update an existing memory.

        Args:
            path: Path of the memory to update.
            content: Full replacement content (mutually exclusive with *append*).
            append: Text to append to existing content.
            tags_add: Tags to add.
            tags_remove: Tags to remove.

        Returns:
            Updated :class:`~llmfs.core.memory_object.MemoryObject`.

        Raises:
            MemoryNotFoundError: If *path* doesn't exist.
            ValueError: If both *content* and *append* are provided.
        """
        if content and append:
            raise ValueError("Provide either 'content' or 'append', not both")

        row = self._db.get_file(path)
        if not row:
            raise MemoryNotFoundError(path)

        existing_obj = self._load_object(row)

        if append:
            new_content = existing_obj.content + "\n" + append
        elif content:
            new_content = content
        else:
            new_content = existing_obj.content

        # Handle tag mutations
        current_tags = list(existing_obj.tags)
        if tags_add:
            current_tags = list(set(current_tags) | set(tags_add))
        if tags_remove:
            current_tags = [t for t in current_tags if t not in tags_remove]

        return self.write(
            path,
            new_content,
            layer=existing_obj.layer,
            tags=current_tags,
            source=existing_obj.metadata.source,
        )

    # ── Forget ────────────────────────────────────────────────────────────────

    def forget(
        self,
        path: str | None = None,
        *,
        layer: str | None = None,
        older_than: str | None = None,
    ) -> dict[str, Any]:
        """Delete memories matching the given criteria.

        At least one of *path*, *layer*, or *older_than* must be provided.

        Args:
            path: Delete a single memory by exact path.
            layer: Delete all memories in a layer.
            older_than: Human duration string, e.g. ``"7 days"``, ``"1 hour"``.
                Deletes memories created before ``now - duration``.

        Returns:
            ``{"deleted": N, "status": "ok"}``.

        Raises:
            ValueError: If no criteria are provided.
            MemoryNotFoundError: If *path* is given but doesn't exist.
        """
        if path is None and layer is None and older_than is None:
            raise ValueError("Provide at least one of: path, layer, older_than")

        deleted = 0

        if path:
            row = self._db.get_file(path)
            if not row:
                raise MemoryNotFoundError(path)
            self._get_vs().delete_by_file_id(row["id"])
            deleted += self._db.delete_file(path)
            self._db.cache_invalidate(path)

        elif layer:
            files = self._db.list_files(layer=layer)
            for f in files:
                self._get_vs().delete_by_file_id(f["id"])
                self._db.delete_file(f["path"])
                deleted += 1
            self._db.cache_invalidate()

        elif older_than:
            cutoff = _parse_older_than(older_than)
            if cutoff:
                files = self._db.list_files(created_before=cutoff)
                for f in files:
                    self._get_vs().delete_by_file_id(f["id"])
                    self._db.delete_file(f["path"])
                    deleted += 1
                self._db.cache_invalidate()

        logger.info("forget: deleted %d memories", deleted)
        return {"deleted": deleted, "status": "ok"}

    # ── Relate ────────────────────────────────────────────────────────────────

    def relate(
        self,
        source: str,
        target: str,
        relationship: str,
        strength: float = 0.8,
    ) -> dict[str, Any]:
        """Create a directed relationship between two memories.

        Args:
            source: Source memory path.
            target: Target memory path.
            relationship: Relationship type string (e.g. ``"related_to"``,
                ``"follows"``, ``"caused_by"``).
            strength: Edge weight in [0, 1].

        Returns:
            ``{"relationship_id": "...", "status": "ok"}``.

        Raises:
            MemoryNotFoundError: If either path doesn't exist.
        """
        src_row = self._db.get_file(source)
        if not src_row:
            raise MemoryNotFoundError(source)
        tgt_row = self._db.get_file(target)
        if not tgt_row:
            raise MemoryNotFoundError(target)

        rel_id = str(uuid.uuid4())
        self._db.insert_relationship(
            id=rel_id,
            source_id=src_row["id"],
            target_id=tgt_row["id"],
            rel_type=relationship,
            strength=strength,
        )
        return {"relationship_id": rel_id, "status": "ok"}

    # ── List ──────────────────────────────────────────────────────────────────

    def list(
        self,
        path_prefix: str = "/",
        *,
        layer: str | None = None,
        recursive: bool = True,
    ) -> list[MemoryObject]:
        """List memories under *path_prefix*.

        Args:
            path_prefix: Only return memories whose path starts here.
            layer: Optional layer filter.
            recursive: If ``False``, only return direct children (not
                implemented yet — currently always returns all matches).

        Returns:
            List of :class:`~llmfs.core.memory_object.MemoryObject`.
        """
        rows = self._db.list_files(layer=layer, path_prefix=path_prefix)
        return [self._load_object(r) for r in rows]

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        """Return a summary of the LLMFS instance.

        Returns:
            Dict with ``total``, ``layers``, ``chunks``, ``disk_mb`` keys.
        """
        all_files = self._db.list_files()
        layers: dict[str, int] = {}
        for f in all_files:
            layers[f["layer"]] = layers.get(f["layer"], 0) + 1

        disk_mb = sum(
            p.stat().st_size for p in self._base.rglob("*") if p.is_file()
        ) / (1024 * 1024)

        return {
            "total": len(all_files),
            "layers": layers,
            "chunks": self._get_vs().count(),
            "disk_mb": round(disk_mb, 2),
            "base_path": str(self._base),
        }

    # ── GC (public) ───────────────────────────────────────────────────────────

    def query(self, mql: str) -> list[Any]:  # type: ignore[valid-type]
        """Execute an MQL query string and return matching memories.

        This is a convenience wrapper around
        :class:`~llmfs.query.executor.MQLExecutor`.

        Args:
            mql: MQL query string, e.g.
                ``'SELECT memory FROM /knowledge WHERE SIMILAR TO "auth bug"'``

        Returns:
            List of :class:`~llmfs.core.memory_object.SearchResult`.

        Raises:
            MQLParseError: On syntax errors.
            MQLExecutionError: On runtime errors.

        Example::

            results = mem.query(
                'SELECT memory FROM / WHERE TAG = "bug" LIMIT 10'
            )
        """
        from llmfs.query.executor import execute_mql
        return execute_mql(mql, self)

    # ── GC (public) ───────────────────────────────────────────────────────────

    def gc(self) -> dict[str, Any]:
        """Garbage-collect expired (TTL) memories.

        Deletes all memories whose ``ttl_expires`` timestamp has passed from
        both SQLite and ChromaDB.

        Returns:
            ``{"deleted": N, "status": "ok"}`` where *N* is the number of
            memories removed.
        """
        # Identify expired file IDs before deleting (for vector store cleanup)
        expired = self._db.list_expired()
        for row in expired:
            self._get_vs().delete_by_file_id(row["id"])
        deleted = self._db.expire_ttl()
        logger.info("gc: deleted %d expired memories", deleted)
        return {"deleted": deleted, "status": "ok"}

    # ── Internals ─────────────────────────────────────────────────────────────

    def _raw_hits_to_results(
        self,
        raw_hits: list[dict],
        *,
        tags: list[str] | None = None,
        created_after: str | None = None,
    ) -> list[SearchResult]:
        """Convert ChromaDB raw hits to SearchResult, applying post-filters."""
        seen: dict[str, SearchResult] = {}
        for item in raw_hits:
            item_path: str = item["metadata"].get("path", "")
            if not item_path:
                continue
            if item_path in seen and item["score"] <= seen[item_path].score:
                continue

            file_row = self._db.get_file(item_path)
            if not file_row:
                continue

            if tags:
                file_tags = set(file_row.get("tags", []))
                if not all(t in file_tags for t in tags):
                    continue

            if created_after:
                created = file_row.get("created_at", "")
                if created and created < created_after:
                    continue

            seen[item_path] = SearchResult(
                path=item_path,
                content=item["text"],
                score=item["score"],
                metadata={
                    "layer": file_row.get("layer", ""),
                    "created_at": file_row.get("created_at", ""),
                    "modified_at": file_row.get("modified_at", ""),
                    "source": file_row.get("source", ""),
                },
                tags=file_row.get("tags", []),
                chunk_text=item["text"],
            )
        return sorted(seen.values(), key=lambda r: r.score, reverse=True)

    def _bm25_search(
        self,
        query: str,
        *,
        layer: str | None = None,
        path_prefix: str | None = None,
        tags: list[str] | None = None,
        created_after: str | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        """BM25 keyword search via SQLite FTS5, returning SearchResults."""
        fts_hits = self._db.fts_search(
            query, limit=limit, layer=layer, path_prefix=path_prefix,
        )
        if not fts_hits:
            return []

        # Normalize BM25 scores to [0, 1]
        max_score = max(h["bm25_score"] for h in fts_hits) if fts_hits else 1.0
        max_score = max(max_score, 0.001)  # avoid division by zero

        seen: dict[str, SearchResult] = {}
        for hit in fts_hits:
            path = hit.get("path", "")
            if not path:
                continue
            norm_score = hit["bm25_score"] / max_score

            if path in seen and norm_score <= seen[path].score:
                continue

            file_row = self._db.get_file(path)
            if not file_row:
                continue

            if tags:
                file_tags = set(file_row.get("tags", []))
                if not all(t in file_tags for t in tags):
                    continue

            if created_after:
                created = file_row.get("created_at", "")
                if created and created < created_after:
                    continue

            seen[path] = SearchResult(
                path=path,
                content=hit.get("text", ""),
                score=norm_score,
                metadata={
                    "layer": file_row.get("layer", ""),
                    "created_at": file_row.get("created_at", ""),
                    "modified_at": file_row.get("modified_at", ""),
                    "source": file_row.get("source", ""),
                },
                tags=file_row.get("tags", []),
                chunk_text=hit.get("text", ""),
            )
        return sorted(seen.values(), key=lambda r: r.score, reverse=True)

    def _auto_link_memory(
        self,
        mem_id: str,
        path: str,
        embedding: list[float],
    ) -> None:
        """Create ``related_to`` edges to semantically similar memories.

        Uses the first chunk embedding to find neighbours via ChromaDB,
        then creates graph edges for any that exceed the similarity threshold.
        Runs silently — never raises.
        """
        try:
            # Query for neighbours (oversample to filter self)
            hits = self._get_vs().query(
                embedding,
                k=self._auto_link_k + 5,
            )
            src_row = self._db.get_file(path)
            if not src_row:
                return

            linked = 0
            for hit in hits:
                hit_path = hit["metadata"].get("path", "")
                if not hit_path or hit_path == path:
                    continue
                if hit["score"] < self._auto_link_threshold:
                    continue
                tgt_row = self._db.get_file(hit_path)
                if not tgt_row:
                    continue
                # Check if relationship already exists
                existing_rels = self._db.get_relationships(src_row["id"])
                already_linked = any(
                    r["target_id"] == tgt_row["id"] for r in existing_rels
                )
                if already_linked:
                    continue
                rel_id = str(uuid.uuid4())
                self._db.insert_relationship(
                    id=rel_id,
                    source_id=src_row["id"],
                    target_id=tgt_row["id"],
                    rel_type="related_to",
                    strength=round(hit["score"], 3),
                )
                linked += 1
                if linked >= self._auto_link_k:
                    break
            if linked:
                logger.debug("auto_link: %s linked to %d neighbours", path, linked)
        except Exception as exc:
            logger.debug("auto_link failed (non-fatal): %s", exc)

    def _load_object(self, row: dict[str, Any]) -> MemoryObject:
        """Reconstruct a MemoryObject from a SQLite file row."""
        chunks_data = self._db.get_chunks(row["id"])
        chunks = [
            Chunk(
                index=c["chunk_index"],
                text=c["text"],
                start_offset=c["start_offset"],
                end_offset=c["end_offset"],
                embedding_id=c["embedding_id"],
            )
            for c in chunks_data
        ]
        rels_data = self._db.get_relationships(row["id"])
        rels = []
        for r in rels_data:
            tgt_row = self._db.get_file_by_id(r["target_id"])
            if tgt_row:
                rels.append(Relationship(
                    target=tgt_row["path"],
                    type=r["type"],
                    strength=r["strength"],
                ))

        content = " ".join(c.text for c in chunks) if chunks else ""
        return MemoryObject(
            id=row["id"],
            path=row["path"],
            content=content,
            layer=row["layer"],
            chunks=chunks,
            metadata=MemoryMetadata(
                created_at=row.get("created_at", ""),
                modified_at=row.get("modified_at", ""),
                accessed_at=row.get("accessed_at", "") or "",
                tags=row.get("tags", []),
                ttl=row.get("ttl_expires"),
                source=row.get("source", "manual"),
            ),
            relationships=rels,
        )

    @staticmethod
    def _validate_path(path: str) -> None:
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError(f"Path must start with '/': {path!r}")

    @staticmethod
    def _validate_layer(layer: str) -> None:
        if layer not in VALID_LAYERS:
            raise ValueError(
                f"Invalid layer {layer!r}. Valid layers: {sorted(VALID_LAYERS)}"
            )

    @staticmethod
    def _build_where(
        layer: str | None,
        path_prefix: str | None,
    ) -> dict[str, Any]:
        """Build a ChromaDB ``where`` filter dict."""
        conditions = []
        if layer:
            conditions.append({"layer": {"$eq": layer}})
        if path_prefix:
            # ChromaDB doesn't support LIKE; we do prefix filtering post-query
            pass
        if len(conditions) == 1:
            return conditions[0]
        if len(conditions) > 1:
            return {"$and": conditions}
        return {}

    def __repr__(self) -> str:
        return f"MemoryFS(path={self._base!r})"


# ── Time helpers ──────────────────────────────────────────────────────────────


def _parse_time_range(time_range: str) -> str | None:
    """Convert a human time range to an ISO-8601 cutoff (lower bound)."""
    from datetime import timedelta
    tr = time_range.lower().strip()
    now = datetime.now(timezone.utc)

    patterns = [
        (r"last (\d+) minutes?", lambda m: timedelta(minutes=int(m.group(1)))),
        (r"last (\d+) hours?", lambda m: timedelta(hours=int(m.group(1)))),
        (r"last (\d+) days?", lambda m: timedelta(days=int(m.group(1)))),
        (r"last (\d+) weeks?", lambda m: timedelta(weeks=int(m.group(1)))),
        (r"today", lambda m: timedelta(hours=now.hour, minutes=now.minute, seconds=now.second)),
        (r"yesterday", lambda m: timedelta(days=1)),
        (r"this week", lambda m: timedelta(days=now.weekday())),
    ]
    import re
    for pattern, delta_fn in patterns:
        m = re.fullmatch(pattern, tr)
        if m:
            cutoff = now - delta_fn(m)
            return cutoff.isoformat()
    return None


def _parse_older_than(older_than: str) -> str | None:
    """Convert ``"7 days"`` style string to an ISO-8601 cutoff (upper bound)."""
    import re
    from datetime import timedelta
    m = re.fullmatch(r"(\d+)\s*(minute|hour|day|week)s?", older_than.lower().strip())
    if not m:
        return None
    amount, unit = int(m.group(1)), m.group(2)
    delta_map = {"minute": timedelta(minutes=amount), "hour": timedelta(hours=amount),
                 "day": timedelta(days=amount), "week": timedelta(weeks=amount)}
    cutoff = datetime.now(timezone.utc) - delta_map[unit]
    return cutoff.isoformat()
