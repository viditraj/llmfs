"""
MQL Executor — runs a parsed MQL AST against a MemoryFS instance.

:class:`MQLExecutor` walks the :class:`~llmfs.query.parser.SelectStatement`
AST produced by :class:`~llmfs.query.parser.MQLParser` and dispatches each
condition type to the appropriate LLMFS retrieval call:

- :class:`~llmfs.query.parser.SimilarCondition`   → ``mem.search(query)``
- :class:`~llmfs.query.parser.TagCondition`       → SQLite tag filter
- :class:`~llmfs.query.parser.DateCondition`      → SQLite date range
- :class:`~llmfs.query.parser.TopicCondition`     → keyword search via ``mem.search``
- :class:`~llmfs.query.parser.RelatedToCondition` → ``MemoryGraph.bfs`` traversal
- :class:`~llmfs.query.parser.AndCondition`       → intersection of result sets
- :class:`~llmfs.query.parser.OrCondition`        → union of result sets

The top-level :func:`execute_mql` helper (and ``MemoryFS.query()``) accept a
raw MQL string, parse it, execute it, and return the result list.

Example::

    from llmfs import MemoryFS
    from llmfs.query.executor import MQLExecutor

    mem = MemoryFS(path="/tmp/test")
    mem.write("/knowledge/auth", "JWT expiry bug at auth.py:45",
              tags=["auth", "bug"])

    executor = MQLExecutor(mem)
    results = executor.execute_mql(
        'SELECT memory FROM /knowledge WHERE SIMILAR TO "JWT bug"'
    )
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from llmfs.core.exceptions import MQLExecutionError
from llmfs.core.memory_object import SearchResult
from llmfs.query.parser import (
    AndCondition,
    DateCondition,
    MQLParser,
    OrCondition,
    RelatedToCondition,
    SelectStatement,
    SimilarCondition,
    TagCondition,
    TopicCondition,
)

if TYPE_CHECKING:
    from llmfs.core.filesystem import MemoryFS

__all__ = ["MQLExecutor", "execute_mql"]

logger = logging.getLogger(__name__)


class MQLExecutor:
    """Executes a parsed MQL :class:`~llmfs.query.parser.SelectStatement`.

    Args:
        mem: :class:`~llmfs.core.filesystem.MemoryFS` instance.

    Example::

        executor = MQLExecutor(mem)
        results = executor.execute_mql('SELECT memory FROM / LIMIT 5')
    """

    def __init__(self, mem: MemoryFS) -> None:
        self._mem = mem
        self._parser = MQLParser()

    # ── Public API ────────────────────────────────────────────────────────────

    def execute_mql(self, mql: str) -> list[SearchResult]:
        """Parse *mql* and execute it.

        Args:
            mql: Raw MQL query string.

        Returns:
            List of :class:`~llmfs.core.memory_object.SearchResult`.

        Raises:
            MQLParseError: On syntax errors.
            MQLExecutionError: On runtime errors during execution.
        """
        stmt = self._parser.parse(mql)
        return self.execute(stmt)

    def execute(self, stmt: SelectStatement) -> list[SearchResult]:
        """Execute a pre-parsed :class:`~llmfs.query.parser.SelectStatement`.

        Args:
            stmt: Parsed MQL statement.

        Returns:
            List of :class:`~llmfs.core.memory_object.SearchResult`.

        Raises:
            MQLExecutionError: On runtime errors.
        """
        try:
            results = self._execute_statement(stmt)
        except Exception as exc:
            if "MQL" in type(exc).__name__:
                raise
            raise MQLExecutionError(str(exc)) from exc

        logger.debug("execute: stmt=%r results=%d", stmt, len(results))
        return results

    # ── Statement execution ───────────────────────────────────────────────────

    def _execute_statement(self, stmt: SelectStatement) -> list[SearchResult]:
        path = stmt.path or "/"
        limit = stmt.limit
        order_by = stmt.order_by
        order_dir = stmt.order_dir

        if stmt.conditions is None:
            # No WHERE clause — list everything under path
            results = self._list_under_path(path)
        else:
            results = self._eval_condition(stmt.conditions, path_prefix=path)

        # Optionally re-sort
        results = self._sort_results(results, order_by=order_by, order_dir=order_dir)

        # Apply LIMIT
        if limit is not None:
            results = results[:limit]

        return results

    # ── Condition evaluation ──────────────────────────────────────────────────

    def _eval_condition(
        self,
        cond: Any,
        *,
        path_prefix: str,
    ) -> list[SearchResult]:
        """Recursively evaluate a condition node."""
        if isinstance(cond, SimilarCondition):
            return self._eval_similar(cond, path_prefix=path_prefix)
        if isinstance(cond, TagCondition):
            return self._eval_tag(cond, path_prefix=path_prefix)
        if isinstance(cond, DateCondition):
            return self._eval_date(cond, path_prefix=path_prefix)
        if isinstance(cond, TopicCondition):
            return self._eval_topic(cond, path_prefix=path_prefix)
        if isinstance(cond, RelatedToCondition):
            return self._eval_related(cond)
        if isinstance(cond, AndCondition):
            left = self._eval_condition(cond.left, path_prefix=path_prefix)
            right = self._eval_condition(cond.right, path_prefix=path_prefix)
            return _intersect(left, right)
        if isinstance(cond, OrCondition):
            left = self._eval_condition(cond.left, path_prefix=path_prefix)
            right = self._eval_condition(cond.right, path_prefix=path_prefix)
            return _union(left, right)
        raise MQLExecutionError(f"Unknown condition type: {type(cond).__name__}")

    def _eval_similar(
        self,
        cond: SimilarCondition,
        *,
        path_prefix: str,
    ) -> list[SearchResult]:
        """Semantic similarity search."""
        return self._mem.search(
            cond.query_str,
            path_prefix=path_prefix if path_prefix != "/" else None,
            k=50,
        )

    def _eval_tag(
        self,
        cond: TagCondition,
        *,
        path_prefix: str,
    ) -> list[SearchResult]:
        """Tag equality / inequality / IN filter via SQLite."""
        rows = self._mem._db.list_files(path_prefix=path_prefix if path_prefix != "/" else None)
        results: list[SearchResult] = []
        for row in rows:
            row_tags: set[str] = set(row.get("tags", []))
            if cond.op == "=" and cond.tag in row_tags:
                pass
            elif cond.op == "!=" and cond.tag not in row_tags:
                pass
            elif cond.op == "in" and any(v in row_tags for v in cond.values):
                pass
            else:
                continue
            results.append(_row_to_search_result(row))
        return results

    def _eval_date(
        self,
        cond: DateCondition,
        *,
        path_prefix: str,
    ) -> list[SearchResult]:
        """Date comparison filter via SQLite."""
        rows = self._mem._db.list_files(path_prefix=path_prefix if path_prefix != "/" else None)
        results: list[SearchResult] = []
        try:
            # Parse the value — support "2026-01-01" and full ISO-8601
            if len(cond.value) == 10:  # YYYY-MM-DD
                cmp_dt = datetime.fromisoformat(cond.value + "T00:00:00+00:00")
            else:
                cmp_dt = datetime.fromisoformat(cond.value)
            if cmp_dt.tzinfo is None:
                cmp_dt = cmp_dt.replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise MQLExecutionError(f"Invalid date value {cond.value!r}: {exc}") from exc

        field_map = {
            "date": "created_at",
            "created": "created_at",
            "modified": "modified_at",
        }
        db_field = field_map.get(cond.field, "created_at")

        for row in rows:
            raw_ts = row.get(db_field, "")
            if not raw_ts:
                continue
            try:
                row_dt = datetime.fromisoformat(raw_ts)
                if row_dt.tzinfo is None:
                    row_dt = row_dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue

            if cond.op == ">" and row_dt > cmp_dt:
                pass
            elif cond.op == ">=" and row_dt >= cmp_dt:
                pass
            elif cond.op == "<" and row_dt < cmp_dt:
                pass
            elif cond.op == "<=" and row_dt <= cmp_dt:
                pass
            elif cond.op == "=" and row_dt.date() == cmp_dt.date():
                pass
            else:
                continue
            results.append(_row_to_search_result(row))
        return results

    def _eval_topic(
        self,
        cond: TopicCondition,
        *,
        path_prefix: str,
    ) -> list[SearchResult]:
        """Keyword / topic search — delegates to semantic search."""
        return self._mem.search(
            cond.topic_str,
            path_prefix=path_prefix if path_prefix != "/" else None,
            k=50,
        )

    def _eval_related(self, cond: RelatedToCondition) -> list[SearchResult]:
        """Graph traversal from anchor path."""
        from llmfs.graph.memory_graph import MemoryGraph
        graph = MemoryGraph(self._mem._db)
        try:
            traversal = graph.bfs(cond.anchor_path, depth=cond.depth, max_nodes=100)
        except Exception as exc:
            raise MQLExecutionError(f"Graph traversal failed: {exc}") from exc

        results: list[SearchResult] = []
        for visited_path in traversal.visited[1:]:  # skip anchor itself
            row = self._mem._db.get_file(visited_path)
            if not row:
                continue
            depth_val = traversal.depth_map.get(visited_path, cond.depth)
            score = 1.0 / (1.0 + depth_val)
            results.append(SearchResult(
                path=visited_path,
                content="",
                score=score,
                metadata={
                    "created_at": row.get("created_at", ""),
                    "modified_at": row.get("modified_at", ""),
                    "layer": row.get("layer", ""),
                    "source": row.get("source", "manual"),
                },
                tags=row.get("tags", []),
                chunk_text="",
            ))
        return results

    def _list_under_path(self, path_prefix: str) -> list[SearchResult]:
        """List all memories under *path_prefix* (no WHERE clause)."""
        prefix = path_prefix if path_prefix != "/" else None
        rows = self._mem._db.list_files(path_prefix=prefix)
        return [_row_to_search_result(row) for row in rows]

    @staticmethod
    def _sort_results(
        results: list[SearchResult],
        *,
        order_by: str | None,
        order_dir: str,
    ) -> list[SearchResult]:
        reverse = order_dir != "asc"
        if order_by in ("date", "created"):
            return sorted(
                results,
                key=lambda r: r.metadata.get("created_at", ""),
                reverse=reverse,
            )
        if order_by == "modified":
            return sorted(
                results,
                key=lambda r: r.metadata.get("modified_at", ""),
                reverse=reverse,
            )
        if order_by == "score":
            return sorted(results, key=lambda r: r.score, reverse=reverse)
        # Default: sort by score descending
        return sorted(results, key=lambda r: r.score, reverse=True)


# ── Module-level convenience ──────────────────────────────────────────────────


def execute_mql(mql: str, mem: MemoryFS) -> list[SearchResult]:
    """Parse and execute an MQL query string against *mem*.

    This is the function wired into ``MemoryFS.query()``.

    Args:
        mql: Raw MQL string.
        mem: :class:`~llmfs.core.filesystem.MemoryFS` instance.

    Returns:
        List of :class:`~llmfs.core.memory_object.SearchResult`.

    Raises:
        MQLParseError: On syntax errors.
        MQLExecutionError: On runtime execution errors.
    """
    return MQLExecutor(mem).execute_mql(mql)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _row_to_search_result(row: dict[str, Any]) -> SearchResult:
    """Convert a SQLite file row to a SearchResult."""
    return SearchResult(
        path=row["path"],
        content="",
        score=1.0,
        metadata={
            "created_at": row.get("created_at", ""),
            "modified_at": row.get("modified_at", ""),
            "layer": row.get("layer", ""),
            "source": row.get("source", "manual"),
        },
        tags=row.get("tags", []),
        chunk_text="",
    )


def _intersect(a: list[SearchResult], b: list[SearchResult]) -> list[SearchResult]:
    """Return results whose paths appear in both *a* and *b*."""
    b_paths = {r.path for r in b}
    # Preserve scores from *a* (the left / primary side)
    return [r for r in a if r.path in b_paths]


def _union(a: list[SearchResult], b: list[SearchResult]) -> list[SearchResult]:
    """Return merged results, keeping the higher score when a path appears in both."""
    merged: dict[str, SearchResult] = {}
    for r in a:
        merged[r.path] = r
    for r in b:
        if r.path not in merged or r.score > merged[r.path].score:
            merged[r.path] = r
    return list(merged.values())
