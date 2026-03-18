"""
ChromaDB vector store wrapper for LLMFS.

Stores chunk embeddings and exposes semantic search with optional metadata
filters (layer, path prefix, tags).  One ChromaDB collection is created per
LLMFS instance, named ``llmfs_<hex8>`` where the suffix is the first 8 chars
of the SHA-256 of the storage path.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from llmfs.core.exceptions import StorageError

__all__ = ["VectorStore"]

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB wrapper that manages one collection per LLMFS instance.

    Args:
        chroma_dir: Directory where ChromaDB persists its data.

    Example::

        vs = VectorStore("/tmp/test/chroma")
        vs.upsert("id-1", [0.1, 0.2, ...], {"file_id": "abc", "layer": "knowledge"}, "text")
        results = vs.query([0.1, 0.2, ...], k=5)
    """

    def __init__(self, chroma_dir: str | Path) -> None:
        self._dir = Path(chroma_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = "llmfs_" + hashlib.sha256(str(self._dir).encode()).hexdigest()[:8]
        self._client = self._make_client()
        self._collection = self._get_or_create_collection()
        logger.debug(
            "VectorStore ready: collection=%s dir=%s",
            self._collection_name, self._dir,
        )

    # ── Init helpers ──────────────────────────────────────────────────────────

    def _make_client(self):  # type: ignore[return]
        try:
            import chromadb
            return chromadb.PersistentClient(path=str(self._dir))
        except ImportError as exc:
            raise StorageError("chromadb is not installed. Run: pip install chromadb") from exc

    def _get_or_create_collection(self):  # type: ignore[return]
        try:
            return self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise StorageError(f"Failed to create ChromaDB collection: {exc}") from exc

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert(
        self,
        embedding_id: str,
        embedding: list[float],
        metadata: dict[str, Any],
        text: str,
    ) -> None:
        """Insert or update a single chunk embedding.

        Args:
            embedding_id: Unique ID for this chunk (stored in ``chunks`` table).
            embedding: Float vector from the embedder.
            metadata: Arbitrary key/value pairs stored alongside the vector.
                Recommended keys: ``file_id``, ``path``, ``layer``, ``chunk_index``,
                ``tags`` (comma-joined string).
            text: The raw text of the chunk (stored for retrieval).

        Raises:
            StorageError: On ChromaDB error.
        """
        # ChromaDB metadata values must be str | int | float | bool
        safe_meta = {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                     for k, v in metadata.items()}
        try:
            self._collection.upsert(
                ids=[embedding_id],
                embeddings=[embedding],
                metadatas=[safe_meta],
                documents=[text],
            )
        except Exception as exc:
            raise StorageError(f"ChromaDB upsert failed: {exc}") from exc

    def upsert_batch(
        self,
        embedding_ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        texts: list[str],
    ) -> None:
        """Batch upsert for efficiency.

        Args:
            embedding_ids: List of chunk IDs.
            embeddings: Parallel list of float vectors.
            metadatas: Parallel list of metadata dicts.
            texts: Parallel list of raw chunk texts.

        Raises:
            StorageError: On ChromaDB error.
        """
        if not embedding_ids:
            return
        safe_metas = [
            {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
             for k, v in m.items()}
            for m in metadatas
        ]
        try:
            self._collection.upsert(
                ids=embedding_ids,
                embeddings=embeddings,
                metadatas=safe_metas,
                documents=texts,
            )
        except Exception as exc:
            raise StorageError(f"ChromaDB batch upsert failed: {exc}") from exc

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Semantic search by vector similarity.

        Args:
            embedding: Query vector from the embedder.
            k: Number of results to return.
            where: Optional ChromaDB ``where`` filter dict, e.g.
                ``{"layer": {"$eq": "knowledge"}}``.

        Returns:
            List of dicts with keys ``id``, ``text``, ``score``, ``metadata``.
            Results are ordered by descending relevance.

        Raises:
            StorageError: On ChromaDB error.
        """
        try:
            kwargs: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": min(k, max(self._collection.count(), 1)),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where
            res = self._collection.query(**kwargs)
        except Exception as exc:
            raise StorageError(f"ChromaDB query failed: {exc}") from exc

        results = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        for eid, doc, meta, dist in zip(ids, docs, metas, dists, strict=False):
            # ChromaDB cosine distance is in [0, 2]; convert to similarity [0, 1]
            score = max(0.0, 1.0 - dist / 2.0)
            results.append({
                "id": eid,
                "text": doc,
                "score": score,
                "metadata": meta or {},
            })
        return results

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, embedding_id: str) -> None:
        """Delete a single chunk by its embedding ID."""
        try:
            self._collection.delete(ids=[embedding_id])
        except Exception as exc:
            raise StorageError(f"ChromaDB delete failed: {exc}") from exc

    def delete_by_file_id(self, file_id: str) -> None:
        """Delete all chunks belonging to a file.

        Args:
            file_id: The UUID of the parent file.
        """
        try:
            self._collection.delete(where={"file_id": {"$eq": file_id}})
        except Exception as exc:
            raise StorageError(f"ChromaDB delete_by_file_id failed: {exc}") from exc

    # ── Info ──────────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return the total number of chunks stored."""
        return self._collection.count()

    def __repr__(self) -> str:
        return (
            f"VectorStore(collection={self._collection_name!r}, "
            f"dir={self._dir!r}, count={self.count()})"
        )
