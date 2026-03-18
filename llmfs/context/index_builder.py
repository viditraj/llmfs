"""
Memory index builder for LLMFS context management.

:class:`IndexBuilder` generates a compact (~2 k-token) textual index of all
memories stored under a session prefix.  The index is injected into the LLM's
system prompt so it always knows *what* is available in memory, even when the
full content has been evicted from the active context window.

Example output::

    ## LLMFS Memory Index
    You have the following memories (use memory_read/memory_search to retrieve):

    - [/session/abc/turns/1]       (turn 1, 10:30) [user]      — Asked to fix auth module bug
    - [/session/abc/turns/2]       (turn 2, 10:31) [assistant] — Found JWT expiry at auth.py:45
    - [/session/abc/code/turn_2_0] (turn 2, 10:31) [code:py]   — Fixed auth.py token refresh
    ... (17 more — use memory_search "topic" to find relevant ones)

Example::

    from llmfs.context.index_builder import IndexBuilder
    from llmfs import MemoryFS

    mem = MemoryFS(path="/tmp/test_llmfs")
    builder = IndexBuilder()
    index_text = builder.build(session_id="abc123", mem=mem)
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmfs.core.filesystem import MemoryFS

__all__ = ["IndexBuilder"]

logger = logging.getLogger(__name__)

# Approximate chars per token for estimation
_CHARS_PER_TOKEN = 4

# Default display limits
_DEFAULT_MAX_ENTRIES = 50
_SHOW_HEAD = 10
_SHOW_TAIL = 10

# Summary truncation length (chars)
_SUMMARY_WIDTH = 60


class IndexBuilder:
    """Builds the memory index string for system prompt injection.

    Args:
        max_entries: Maximum number of index entries to show before
            truncating to head + tail.

    Example::

        builder = IndexBuilder(max_entries=50)
        text = builder.build(session_id="sess1", mem=mem)
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._max_entries = max_entries

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        session_id: str,
        mem: MemoryFS,
        *,
        max_entries: int | None = None,
    ) -> str:
        """Build the memory index for *session_id*.

        Args:
            session_id: Session namespace (memories under
                ``/session/{session_id}/``).
            mem: :class:`~llmfs.core.filesystem.MemoryFS` instance.
            max_entries: Override the instance default.

        Returns:
            Formatted index string, approximately 2 k tokens.
        """
        limit = max_entries if max_entries is not None else self._max_entries
        prefix = f"/session/{session_id}"
        objects = mem.list(prefix, recursive=True)

        if not objects:
            return (
                "## LLMFS Memory Index\n"
                "No memories stored for this session yet."
            )

        # Sort by turn_id embedded in path, then by path lexicographically
        objects.sort(key=lambda o: (_extract_turn_id(o.path), o.path))

        lines = self._format_entries(objects, limit)

        header = (
            "## LLMFS Memory Index\n"
            "You have the following memories "
            "(use memory_read/memory_search to retrieve):\n"
        )
        return header + "\n" + "\n".join(lines)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token count estimate (len // 4).

        Args:
            text: Any string.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // _CHARS_PER_TOKEN)

    # ── Private ───────────────────────────────────────────────────────────────

    def _format_entries(
        self,
        objects: list[Any],
        limit: int,
    ) -> list[str]:
        total = len(objects)

        if total <= limit:
            display = objects
            omitted = 0
        else:
            # Show up to limit//2 from head and limit//2 from tail
            half = max(1, limit // 2)
            head = objects[:half]
            tail = objects[max(half, total - half):]
            display = head + tail
            omitted = total - len(head) - len(tail)

        lines: list[str] = []
        for obj in display:
            lines.append(_format_entry(obj))

        if omitted > 0:
            lines.append(
                f"... ({omitted} more — use memory_search \"<topic>\" to find relevant ones)"
            )

        return lines


# ── Formatting helpers ────────────────────────────────────────────────────────


def _format_entry(obj: Any) -> str:
    """Format a single MemoryObject as a one-line index entry."""
    path = obj.path
    turn_id = _extract_turn_id(path)

    # Label for the type column (role / artifact type)
    label = _artifact_label(path, obj)

    # Timestamp from metadata
    ts = _format_timestamp(obj.metadata.modified_at or obj.metadata.created_at)

    # Summary: prefer level_2, fall back to first chunk text
    summary = _best_summary(obj)

    turn_part = f"turn {turn_id}, " if turn_id >= 0 else ""
    path_col = f"[{path}]"
    meta_col = f"({turn_part}{ts})"

    line = f"- {path_col:<45} {meta_col:<20} [{label:<12}] — {summary}"
    return line


def _extract_turn_id(path: str) -> int:
    """Extract a numeric turn id from a session path, or -1 if absent.

    Matches both ``/turns/3`` and ``turn_7_0`` style patterns.
    """
    # Match /turns/<number> (strict turns directory)
    m = re.search(r"/turns/(\d+)", path)
    if m:
        return int(m.group(1))
    # Match turn_<number> (artifact paths like /code/turn_2_0)
    m = re.search(r"turn[_/](\d+)", path)
    return int(m.group(1)) if m else -1


def _artifact_label(path: str, obj: Any) -> str:
    """Return a short label describing the artifact type."""
    tags = obj.tags if hasattr(obj, "tags") else []

    if "/code/" in path:
        lang = next((t for t in tags if t not in ("code", "assistant", "user")), "")
        return f"code:{lang}" if lang else "code"
    if "/errors/" in path:
        return "error"
    if "/files/" in path:
        return "file_refs"
    if "/decisions/" in path:
        return "decision"
    if "/turns/" in path:
        role = next((t for t in tags if t in ("user", "assistant")), "turn")
        return role
    return "memory"


def _format_timestamp(ts: str | None) -> str:
    """Return HH:MM from an ISO-8601 timestamp, or empty string."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%H:%M")
    except (ValueError, TypeError):
        return ""


def _best_summary(obj: Any) -> str:
    """Return the best available summary, truncated to _SUMMARY_WIDTH chars."""
    text = ""
    # Try level_2 document summary
    if obj.summaries and obj.summaries.level_2:
        text = obj.summaries.level_2
    # Fall back to first chunk summary
    elif obj.summaries and obj.summaries.level_1:
        text = obj.summaries.level_1[0] if obj.summaries.level_1 else ""
    # Fall back to raw content
    if not text and obj.content:
        text = obj.content

    text = text.replace("\n", " ").strip()
    if len(text) > _SUMMARY_WIDTH:
        text = text[:_SUMMARY_WIDTH - 1] + "…"
    return text
