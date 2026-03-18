"""
SQLite metadata database for LLMFS.

Stores all structured metadata about memories: paths, layers, tags, chunk
references, relationships, and a search result cache.  ChromaDB stores the
actual embedding vectors; this module stores everything else.

WAL mode is enabled for concurrent reads alongside writes.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmfs.core.exceptions import StorageError

__all__ = ["MetadataDB"]

logger = logging.getLogger(__name__)

# ── SQL DDL ───────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    id            TEXT PRIMARY KEY,
    path          TEXT UNIQUE NOT NULL,
    name          TEXT NOT NULL,
    layer         TEXT NOT NULL,
    size          INTEGER DEFAULT 0,
    created_at    TEXT NOT NULL,
    modified_at   TEXT NOT NULL,
    accessed_at   TEXT,
    content_hash  TEXT,
    ttl_expires   TEXT,
    source        TEXT DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS chunks (
    id            TEXT PRIMARY KEY,
    file_id       TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunk_index   INTEGER NOT NULL,
    start_offset  INTEGER DEFAULT 0,
    end_offset    INTEGER DEFAULT 0,
    text          TEXT NOT NULL,
    embedding_id  TEXT NOT NULL,
    summary       TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS tags (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name  TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS file_tags (
    file_id  TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    tag_id   INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (file_id, tag_id)
);

CREATE TABLE IF NOT EXISTS relationships (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    target_id   TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    type        TEXT NOT NULL,
    strength    REAL DEFAULT 0.5,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_cache (
    query_hash    TEXT PRIMARY KEY,
    results_json  TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    expires_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_files_layer ON files(layer);
CREATE INDEX IF NOT EXISTS idx_files_path  ON files(path);
CREATE INDEX IF NOT EXISTS idx_files_ttl   ON files(ttl_expires);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_rel_source  ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target  ON relationships(target_id);
"""


class MetadataDB:
    """SQLite-backed metadata store for LLMFS.

    Thread-safe via an internal :class:`threading.Lock` on write operations.
    Reads use ``check_same_thread=False`` so multiple threads can query.

    Args:
        db_path: Absolute path to the ``.db`` file.  The parent directory
            must exist.

    Example::

        db = MetadataDB("/tmp/test/metadata.db")
        db.insert_file(id="abc", path="/k/test", name="test",
                       layer="knowledge", ...)
        row = db.get_file("/k/test")
    """

    def __init__(self, db_path: str | Path) -> None:
        self._path = Path(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._initialize()
        logger.debug("MetadataDB initialised at %s", self._path)

    # ── Schema ────────────────────────────────────────────────────────────────

    def _initialize(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _utcnow() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Execute a write statement inside the lock."""
        with self._lock:
            try:
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur
            except sqlite3.Error as exc:
                raise StorageError(str(exc)) from exc

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        """Execute a read statement (no lock needed with WAL)."""
        try:
            return self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(str(exc)) from exc

    # ── File CRUD ─────────────────────────────────────────────────────────────

    def insert_file(
        self,
        *,
        id: str,
        path: str,
        name: str,
        layer: str,
        size: int = 0,
        created_at: str = "",
        modified_at: str = "",
        content_hash: str = "",
        ttl_expires: str | None = None,
        source: str = "manual",
    ) -> None:
        """Insert a new file row.

        Args:
            id: UUID for the memory.
            path: Filesystem-style path (must start with ``/``).
            name: Last path component.
            layer: Memory layer string.
            size: Content size in bytes.
            created_at: ISO-8601 UTC creation timestamp.
            modified_at: ISO-8601 UTC modification timestamp.
            content_hash: SHA-256 of content.
            ttl_expires: ISO-8601 expiry timestamp or ``None``.
            source: Origin of the memory (``manual``, ``agent``, ``mcp``, ``cli``).

        Raises:
            StorageError: On SQLite error.
        """
        now = self._utcnow()
        self._exec(
            """
            INSERT INTO files
              (id, path, name, layer, size, created_at, modified_at,
               accessed_at, content_hash, ttl_expires, source)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                id, path, name, layer, size,
                created_at or now, modified_at or now,
                None, content_hash, ttl_expires, source,
            ),
        )

    def get_file(self, path: str) -> dict[str, Any] | None:
        """Return a file row as a dict, or ``None`` if not found.

        Args:
            path: Exact path to look up.

        Returns:
            Dict with all column values, or ``None``.
        """
        rows = self._query("SELECT * FROM files WHERE path = ?", (path,))
        if not rows:
            return None
        row = rows[0]
        result = dict(row)
        result["tags"] = self.get_tags_for_file(result["id"])
        return result

    def get_file_by_id(self, file_id: str) -> dict[str, Any] | None:
        """Return a file row by UUID."""
        rows = self._query("SELECT * FROM files WHERE id = ?", (file_id,))
        if not rows:
            return None
        result = dict(rows[0])
        result["tags"] = self.get_tags_for_file(file_id)
        return result

    def update_file(
        self,
        path: str,
        *,
        size: int | None = None,
        modified_at: str | None = None,
        content_hash: str | None = None,
        ttl_expires: str | None = None,
    ) -> None:
        """Update mutable fields of a file row.

        Args:
            path: Path of the file to update.
            size: New size in bytes (optional).
            modified_at: New modification timestamp (optional).
            content_hash: New content hash (optional).
            ttl_expires: New expiry timestamp or ``None`` to clear (optional).

        Raises:
            StorageError: On SQLite error.
        """
        sets: list[str] = []
        params: list[Any] = []
        now = self._utcnow()
        sets.append("modified_at = ?")
        params.append(modified_at or now)
        if size is not None:
            sets.append("size = ?")
            params.append(size)
        if content_hash is not None:
            sets.append("content_hash = ?")
            params.append(content_hash)
        if ttl_expires is not None:
            sets.append("ttl_expires = ?")
            params.append(ttl_expires)
        params.append(path)
        self._exec(f"UPDATE files SET {', '.join(sets)} WHERE path = ?", tuple(params))

    def touch_accessed(self, path: str) -> None:
        """Update the ``accessed_at`` timestamp for a file."""
        self._exec(
            "UPDATE files SET accessed_at = ? WHERE path = ?",
            (self._utcnow(), path),
        )

    def delete_file(self, path: str) -> int:
        """Delete a file (and cascade to chunks, tags, relationships).

        Returns:
            Number of rows deleted (0 if path didn't exist).
        """
        cur = self._exec("DELETE FROM files WHERE path = ?", (path,))
        return cur.rowcount

    def list_files(
        self,
        *,
        layer: str | None = None,
        path_prefix: str | None = None,
        tags: list[str] | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List files matching optional filters.

        Args:
            layer: Filter to a specific layer.
            path_prefix: Filter to paths starting with this prefix.
            tags: All supplied tags must be present on the file.
            created_after: ISO-8601; only files created after this time.
            created_before: ISO-8601; only files created before this time.
            limit: Maximum number of results.

        Returns:
            List of file row dicts (with ``tags`` key populated).
        """
        sql = "SELECT f.* FROM files f"
        where, params = [], []

        if tags:
            # Require ALL supplied tags (one JOIN per tag)
            for i, tag in enumerate(tags):
                alias = f"ft{i}"
                tag_alias = f"t{i}"
                sql += (
                    f" JOIN file_tags {alias} ON {alias}.file_id = f.id"
                    f" JOIN tags {tag_alias} ON {tag_alias}.id = {alias}.tag_id"
                    f" AND {tag_alias}.name = ?"
                )
                params.append(tag)

        if layer:
            where.append("f.layer = ?")
            params.append(layer)
        if path_prefix:
            where.append("f.path LIKE ?")
            params.append(path_prefix.rstrip("/") + "%")
        if created_after:
            where.append("f.created_at > ?")
            params.append(created_after)
        if created_before:
            where.append("f.created_at < ?")
            params.append(created_before)

        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY f.created_at DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"

        rows = self._query(sql, tuple(params))
        results = []
        for row in rows:
            d = dict(row)
            d["tags"] = self.get_tags_for_file(d["id"])
            results.append(d)
        return results

    # ── Chunks ────────────────────────────────────────────────────────────────

    def insert_chunk(
        self,
        *,
        id: str,
        file_id: str,
        chunk_index: int,
        start_offset: int,
        end_offset: int,
        text: str,
        embedding_id: str,
        summary: str = "",
    ) -> None:
        """Insert a chunk row."""
        self._exec(
            """
            INSERT OR REPLACE INTO chunks
              (id, file_id, chunk_index, start_offset, end_offset,
               text, embedding_id, summary)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (id, file_id, chunk_index, start_offset, end_offset,
             text, embedding_id, summary),
        )

    def get_chunks(self, file_id: str) -> list[dict[str, Any]]:
        """Return all chunks for a file, ordered by chunk_index."""
        rows = self._query(
            "SELECT * FROM chunks WHERE file_id = ? ORDER BY chunk_index",
            (file_id,),
        )
        return [dict(r) for r in rows]

    def delete_chunks(self, file_id: str) -> None:
        """Delete all chunks belonging to a file."""
        self._exec("DELETE FROM chunks WHERE file_id = ?", (file_id,))

    # ── Tags ──────────────────────────────────────────────────────────────────

    def get_or_create_tag(self, name: str) -> int:
        """Return the tag id, creating the tag if it doesn't exist."""
        rows = self._query("SELECT id FROM tags WHERE name = ?", (name,))
        if rows:
            return rows[0]["id"]
        with self._lock:
            cur = self._conn.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
            self._conn.commit()
            if cur.lastrowid:
                return cur.lastrowid
        # Race — another thread inserted it
        return self._query("SELECT id FROM tags WHERE name = ?", (name,))[0]["id"]

    def tag_file(self, file_id: str, tag_name: str) -> None:
        """Associate *tag_name* with *file_id*."""
        tag_id = self.get_or_create_tag(tag_name)
        self._exec(
            "INSERT OR IGNORE INTO file_tags (file_id, tag_id) VALUES (?,?)",
            (file_id, tag_id),
        )

    def untag_file(self, file_id: str, tag_name: str) -> None:
        """Remove *tag_name* association from *file_id*."""
        rows = self._query("SELECT id FROM tags WHERE name = ?", (tag_name,))
        if not rows:
            return
        self._exec(
            "DELETE FROM file_tags WHERE file_id = ? AND tag_id = ?",
            (file_id, rows[0]["id"]),
        )

    def set_tags(self, file_id: str, tags: list[str]) -> None:
        """Replace all tags on a file with *tags*."""
        self._exec("DELETE FROM file_tags WHERE file_id = ?", (file_id,))
        for tag in tags:
            self.tag_file(file_id, tag)

    def get_tags_for_file(self, file_id: str) -> list[str]:
        """Return sorted list of tag names for a file."""
        rows = self._query(
            """
            SELECT t.name FROM tags t
            JOIN file_tags ft ON ft.tag_id = t.id
            WHERE ft.file_id = ?
            ORDER BY t.name
            """,
            (file_id,),
        )
        return [r["name"] for r in rows]

    # ── Relationships ─────────────────────────────────────────────────────────

    def insert_relationship(
        self,
        *,
        id: str,
        source_id: str,
        target_id: str,
        rel_type: str,
        strength: float = 0.8,
    ) -> None:
        """Insert a directed relationship edge."""
        self._exec(
            """
            INSERT OR REPLACE INTO relationships
              (id, source_id, target_id, type, strength, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            (id, source_id, target_id, rel_type, strength, self._utcnow()),
        )

    def get_relationships(self, file_id: str) -> list[dict[str, Any]]:
        """Return all outgoing relationships for a file."""
        rows = self._query(
            "SELECT * FROM relationships WHERE source_id = ? ORDER BY strength DESC",
            (file_id,),
        )
        return [dict(r) for r in rows]

    def get_incoming_relationships(self, file_id: str) -> list[dict[str, Any]]:
        """Return all incoming relationships for a file (edges *to* this node)."""
        rows = self._query(
            "SELECT * FROM relationships WHERE target_id = ? ORDER BY strength DESC",
            (file_id,),
        )
        return [dict(r) for r in rows]

    def delete_relationship(self, rel_id: str) -> None:
        """Delete a single relationship by id."""
        self._exec("DELETE FROM relationships WHERE id = ?", (rel_id,))

    # ── TTL / GC ──────────────────────────────────────────────────────────────

    def expire_ttl(self) -> int:
        """Delete all memories whose TTL has passed.

        Returns:
            Number of files deleted.
        """
        now = self._utcnow()
        cur = self._exec(
            "DELETE FROM files WHERE ttl_expires IS NOT NULL AND ttl_expires < ?",
            (now,),
        )
        deleted = cur.rowcount
        if deleted:
            logger.info("TTL GC: deleted %d expired memories", deleted)
        return deleted

    def list_expired(self) -> list[dict[str, Any]]:
        """Return all files whose TTL has passed (without deleting them).

        Returns:
            List of file row dicts for expired memories.
        """
        now = self._utcnow()
        rows = self._query(
            "SELECT * FROM files WHERE ttl_expires IS NOT NULL AND ttl_expires < ?",
            (now,),
        )
        return [dict(r) for r in rows]

    # ── Search cache ──────────────────────────────────────────────────────────

    def cache_get(self, query_hash: str) -> list[dict[str, Any]] | None:
        """Return cached search results, or ``None`` if missing/expired."""
        now = self._utcnow()
        rows = self._query(
            "SELECT results_json FROM search_cache WHERE query_hash = ? AND expires_at > ?",
            (query_hash, now),
        )
        if not rows:
            return None
        try:
            return json.loads(rows[0]["results_json"])
        except json.JSONDecodeError:
            return None

    def cache_set(
        self,
        query_hash: str,
        results: list[dict[str, Any]],
        ttl_seconds: int = 300,
    ) -> None:
        """Store search results in the cache.

        Args:
            query_hash: SHA-256 of the query + parameter string.
            results: Serialisable list of result dicts.
            ttl_seconds: Cache validity window in seconds.
        """
        from datetime import timedelta
        now_dt = datetime.now(timezone.utc)
        expires_dt = now_dt + timedelta(seconds=ttl_seconds)
        self._exec(
            """
            INSERT OR REPLACE INTO search_cache
              (query_hash, results_json, created_at, expires_at)
            VALUES (?,?,?,?)
            """,
            (query_hash, json.dumps(results), now_dt.isoformat(), expires_dt.isoformat()),
        )

    def cache_invalidate(self, path: str | None = None) -> None:
        """Invalidate cache entries (all, or those mentioning *path*)."""
        if path is None:
            self._exec("DELETE FROM search_cache")
        else:
            self._exec(
                "DELETE FROM search_cache WHERE results_json LIKE ?",
                (f"%{path}%",),
            )

    # ── Close ─────────────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __repr__(self) -> str:
        return f"MetadataDB(path={self._path!r})"
