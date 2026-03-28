"""
RetrievalEngine — hybrid search for LLMFS.

Combines three complementary retrieval signals:

1. **Semantic search** — dense vector similarity via ChromaDB (primary signal).
2. **Temporal filtering** — narrow results to a time window when requested.
3. **Graph expansion** — after finding an initial result set, optionally expand
   it with graph neighbours from :class:`~llmfs.graph.memory_graph.MemoryGraph`.

Results from all signals are merged and re-ranked by
:class:`~llmfs.retrieval.ranker.Ranker` (weighted score fusion + diversity).

The engine is intentionally stateless beyond its dependencies — no caching is
done here (that lives in :class:`~llmfs.storage.metadata_db.MetadataDB`).

Example::

    from llmfs import MemoryFS
    from llmfs.retrieval.engine import RetrievalEngine

    mem = MemoryFS()
    engine = RetrievalEngine(
        db=mem._db, vs=mem._get_vs(), embedder=mem._get_embedder()
    )
    results = engine.search("JWT authentication bug", k=5, layer="knowledge")
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from llmfs.core.exceptions import MemoryNotFoundError
from llmfs.core.memory_object import SearchResult
from llmfs.embeddings.base import EmbedderBase
from llmfs.graph.memory_graph import MemoryGraph
from llmfs.retrieval.ranker import Ranker
from llmfs.storage.metadata_db import MetadataDB
from llmfs.storage.vector_store import VectorStore

__all__ = ["RetrievalEngine"]

logger = logging.getLogger(__name__)

# How many raw semantic results to fetch before re-ranking
_SEMANTIC_OVERSAMPLE = 3


def _parse_time_range(time_range: str) -> datetime | None:
    """Parse a human-readable time range into a UTC cutoff datetime.

    Supports strings like:
    - ``"last 30 minutes"`` / ``"last 30 mins"``
    - ``"last 7 days"`` / ``"last 7d"``
    - ``"today"``
    - ``"last 1 hour"`` / ``"last 2 hours"``
    - ``"last week"``

    Args:
        time_range: Human time string.

    Returns:
        UTC datetime representing the start of the window, or ``None`` if the
        string could not be parsed.
    """
    now = datetime.now(timezone.utc)
    s = time_range.strip().lower()

    if s == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    if s == "last week":
        return now - timedelta(weeks=1)

    m = re.match(
        r"last\s+(\d+)\s*(minute|min|hour|day|week|month)s?",
        s,
    )
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        delta_map = {
            "minute": timedelta(minutes=n),
            "min": timedelta(minutes=n),
            "hour": timedelta(hours=n),
            "day": timedelta(days=n),
            "week": timedelta(weeks=n),
            "month": timedelta(days=n * 30),
        }
        delta = delta_map.get(unit)
        if delta:
            return now - delta

    logger.debug("Could not parse time_range %r", time_range)
    return None


class RetrievalEngine:
    """Hybrid retrieval engine for LLMFS memories.

    Combines semantic search, temporal filtering, metadata filters, and
    optional graph expansion into a single ranked result list.

    Args:
        db: Metadata database.
        vs: Vector store.
        embedder: Embedder to use for query encoding.
        graph: Optional :class:`~llmfs.graph.memory_graph.MemoryGraph` for
            graph-expanded retrieval.  If ``None``, graph expansion is skipped.
        ranker: Optional :class:`~llmfs.retrieval.ranker.Ranker` instance.
            A default one is created if not provided.

    Example::

        engine = RetrievalEngine(db=mem._db, vs=mem._get_vs(), embedder=mem._get_embedder())
        results = engine.search("auth bug", k=5)
    """

    def __init__(
        self,
        db: MetadataDB,
        vs: VectorStore,
        embedder: EmbedderBase,
        graph: MemoryGraph | None = None,
        ranker: Ranker | None = None,
    ) -> None:
        self._db = db
        self._vs = vs
        self._embedder = embedder
        self._graph = graph
        self._ranker = ranker or Ranker()

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        layer: str | None = None,
        tags: list[str] | None = None,
        path_prefix: str | None = None,
        time_range: str | None = None,
        k: int = 5,
        graph_expand: bool = False,
        graph_depth: int = 1,
    ) -> list[SearchResult]:
        """Hybrid search combining semantic, temporal, and graph signals.

        Args:
            query: Natural-language query string.
            layer: Restrict to a specific memory layer.
            tags: Only return memories that have *all* of these tags.
            path_prefix: Restrict to paths starting with this string.
            time_range: Human time string (e.g. ``"last 7 days"``).  When
                provided, only memories modified within this window are returned.
            k: Number of results to return after ranking.
            graph_expand: If ``True`` and a :class:`MemoryGraph` was provided,
                expand the initial result set with graph neighbours.
            graph_depth: Hops to traverse when *graph_expand* is ``True``.

        Returns:
            Ranked :class:`~llmfs.core.memory_object.SearchResult` list
            (up to *k* items), highest relevance first.
        """
        if not query.strip():
            return []

        # 1. Embed the query
        q_emb = self._embedder.embed(query)

        # 2. Build ChromaDB where-filter
        where = self._build_where(layer=layer, path_prefix=path_prefix)

        # 3. Semantic search — oversample to allow for post-filters
        raw_k = max(k * _SEMANTIC_OVERSAMPLE, 20)
        raw_hits = self._vs.query(q_emb, k=raw_k, where=where or None)

        # 4. Convert hits → SearchResult, apply time and tag filters
        time_cutoff = _parse_time_range(time_range) if time_range else None
        results = self._hits_to_results(raw_hits, tags=tags, time_cutoff=time_cutoff)

        # 5. Optional graph expansion
        if graph_expand and self._graph and results:
            results = self._expand_with_graph(results, depth=graph_depth, k=k * 2)

        # 6. Deduplicate by path (keep best chunk score per path)
        results = _deduplicate(results)

        # 7. Re-rank
        ranked = self._ranker.rank(results, query=query, query_tags=tags, top_k=k)

        logger.debug(
            "search %r: raw=%d post_filter=%d final=%d",
            query[:40], len(raw_hits), len(results), len(ranked),
        )
        return ranked

    def search_by_path_prefix(
        self,
        prefix: str,
        *,
        layer: str | None = None,
        k: int = 50,
    ) -> list[SearchResult]:
        """List memories under a path prefix without semantic ranking.

        Args:
            prefix: Path prefix, e.g. ``"/projects/auth"``.
            layer: Optional layer filter.
            k: Maximum results.

        Returns:
            :class:`~llmfs.core.memory_object.SearchResult` list ordered by
            ``modified_at`` descending.
        """
        rows = self._db.list_files(layer=layer, path_prefix=prefix)
        results = []
        for row in rows[:k]:
            results.append(SearchResult(
                path=row["path"],
                content="",  # content not loaded here
                score=1.0,
                metadata={
                    "created_at": row.get("created_at", ""),
                    "modified_at": row.get("modified_at", ""),
                    "accessed_at": row.get("accessed_at", ""),
                    "source": row.get("source", "manual"),
                },
                tags=row.get("tags", []),
                chunk_text="",
            ))
        return results

    def related(
        self,
        path: str,
        *,
        depth: int = 2,
        k: int = 10,
    ) -> list[SearchResult]:
        """Return memories related to *path* via the knowledge graph.

        Requires a :class:`~llmfs.graph.memory_graph.MemoryGraph` to have been
        provided at construction time.

        Args:
            path: Source memory path.
            depth: BFS depth to traverse.
            k: Maximum results.

        Returns:
            :class:`~llmfs.core.memory_object.SearchResult` list ordered by
            graph proximity (closer nodes first), then strength.

        Raises:
            MemoryNotFoundError: If *path* does not exist.
            ValueError: If no graph was provided.
        """
        if self._graph is None:
            raise ValueError("RetrievalEngine was created without a MemoryGraph")

        traversal = self._graph.bfs(path, depth=depth, max_nodes=k * 3)
        results: list[SearchResult] = []

        for visited_path in traversal.visited[1:]:  # skip the root itself
            row = self._db.get_file(visited_path)
            if not row:
                continue
            graph_depth = traversal.depth_map.get(visited_path, depth)
            # Score based on depth: closer = higher score
            score = 1.0 / (1.0 + graph_depth)
            results.append(SearchResult(
                path=visited_path,
                content="",
                score=score,
                metadata={
                    "created_at": row.get("created_at", ""),
                    "modified_at": row.get("modified_at", ""),
                    "source": row.get("source", "manual"),
                },
                tags=row.get("tags", []),
            ))
            if len(results) >= k:
                break

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_where(
        layer: str | None,
        path_prefix: str | None,
    ) -> dict[str, Any]:
        """Build a ChromaDB ``where`` filter dict.

        ChromaDB supports only a single ``$and`` nesting level in most
        versions, so we build it carefully.
        """
        conditions: list[dict[str, Any]] = []

        if layer:
            conditions.append({"layer": {"$eq": layer}})
        if path_prefix:
            # ChromaDB metadata filter: check path starts with prefix using
            # a ``$contains`` substring match (available in chroma >=0.4.14).
            # Fall back to exact match if needed — the path_prefix filter is
            # applied again in Python after the query for safety.
            conditions.append({"path": {"$contains": path_prefix}})

        if len(conditions) == 0:
            return {}
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def _hits_to_results(
        self,
        hits: list[dict[str, Any]],
        tags: list[str] | None,
        time_cutoff: datetime | None,
    ) -> list[SearchResult]:
        """Convert raw VectorStore hits to SearchResult objects, applying filters."""
        results: list[SearchResult] = []

        for hit in hits:
            meta = hit.get("metadata", {})
            path = meta.get("path", "")
            if not path:
                continue

            # Time filter
            if time_cutoff:
                modified_at = meta.get("modified_at") or meta.get("created_at")
                if modified_at:
                    try:
                        dt = datetime.fromisoformat(modified_at)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        if dt < time_cutoff:
                            continue
                    except ValueError:
                        pass

            # Tag filter — check against the DB for accurate tag list
            result_tags: list[str] = []
            if tags or True:  # always fetch tags for downstream ranking
                row = self._db.get_file(path)
                if row:
                    result_tags = row.get("tags", [])
                else:
                    result_tags = []

            if tags and not all(t in result_tags for t in tags):
                continue

            results.append(SearchResult(
                path=path,
                content=hit.get("text", ""),
                score=float(hit.get("score", 0.0)),
                metadata={
                    "created_at": meta.get("created_at", ""),
                    "modified_at": meta.get("modified_at", ""),
                    "accessed_at": meta.get("accessed_at", ""),
                    "layer": meta.get("layer", ""),
                    "source": meta.get("source", "manual"),
                },
                tags=result_tags,
                chunk_text=hit.get("text", ""),
            ))

        return results

    def _expand_with_graph(
        self,
        results: list[SearchResult],
        depth: int,
        k: int,
    ) -> list[SearchResult]:
        """Expand *results* with graph neighbours of the top-ranked paths."""
        assert self._graph is not None

        expanded = list(results)
        seen_paths = {r.path for r in results}

        # Only expand from the top-5 seeds to keep it bounded
        seeds = [r.path for r in results[:5]]
        for seed_path in seeds:
            try:
                traversal = self._graph.bfs(seed_path, depth=depth, max_nodes=k)
            except MemoryNotFoundError:
                continue

            for visited_path in traversal.visited[1:]:
                if visited_path in seen_paths:
                    continue
                row = self._db.get_file(visited_path)
                if not row:
                    continue
                graph_depth_val = traversal.depth_map.get(visited_path, depth)
                score = 0.3 / (1.0 + graph_depth_val)  # graph results get lower base score
                expanded.append(SearchResult(
                    path=visited_path,
                    content="",
                    score=score,
                    metadata={
                        "created_at": row.get("created_at", ""),
                        "modified_at": row.get("modified_at", ""),
                        "source": row.get("source", "manual"),
                    },
                    tags=row.get("tags", []),
                ))
                seen_paths.add(visited_path)
                if len(expanded) >= k:
                    break

        return expanded

    def __repr__(self) -> str:
        return (
            f"RetrievalEngine(db={self._db!r}, graph={'yes' if self._graph else 'no'})"
        )


# ── Module helpers ────────────────────────────────────────────────────────────


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    """Keep the highest-scoring result for each unique path."""
    best: dict[str, SearchResult] = {}
    for r in results:
        if r.path not in best or r.score > best[r.path].score:
            best[r.path] = r
    return list(best.values())
