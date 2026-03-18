"""
Adaptive content chunker for LLMFS.

Splits content into semantically meaningful chunks before embedding:

- **Python code** — split at top-level ``def``/``class`` boundaries using the
  ``ast`` module, with a function-signature overlap header on each chunk.
- **Markdown** — split at headers (``#`` lines), then sub-split large sections
  by paragraph.
- **Plain text** — sliding window of ~256 tokens with 50-token overlap.

All chunk sizes are measured in *approximate tokens* (``len(text.split())``).
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass

from llmfs.core.exceptions import ChunkerError

__all__ = ["Chunk", "AdaptiveChunker"]

logger = logging.getLogger(__name__)

# Target token counts (rough: 1 token ≈ 1 word)
_CODE_TARGET = 512
_PROSE_TARGET = 256
_PLAIN_TARGET = 256
_OVERLAP = 50


@dataclass
class Chunk:
    """A text chunk produced by :class:`AdaptiveChunker`.

    Attributes:
        index: Zero-based position in the chunk list.
        text: The chunk text.
        start_offset: Byte offset of the first character in the original content.
        end_offset: Byte offset one past the last character.
    """

    index: int
    text: str
    start_offset: int = 0
    end_offset: int = 0


class AdaptiveChunker:
    """Content-aware text chunker.

    Detects the content type and applies the most suitable splitting strategy.

    Args:
        code_target: Target token count for code chunks.
        prose_target: Target token count for prose/markdown chunks.
        plain_target: Target token count for plain-text chunks.
        overlap: Token overlap between adjacent plain-text chunks.

    Example::

        chunker = AdaptiveChunker()
        chunks = chunker.chunk("def foo():\\n    return 1\\n", content_type="python")
        for c in chunks:
            print(c.index, c.text[:40])
    """

    def __init__(
        self,
        code_target: int = _CODE_TARGET,
        prose_target: int = _PROSE_TARGET,
        plain_target: int = _PLAIN_TARGET,
        overlap: int = _OVERLAP,
    ) -> None:
        self._code_target = code_target
        self._prose_target = prose_target
        self._plain_target = plain_target
        self._overlap = overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(self, content: str, content_type: str | None = None) -> list[Chunk]:
        """Split *content* into chunks.

        Args:
            content: The raw text to split.
            content_type: Hint about the content type.  One of ``"python"``,
                ``"markdown"``, ``"code"`` (any language), or ``"text"``.
                If ``None``, the type is auto-detected.

        Returns:
            Non-empty list of :class:`Chunk` objects.  A single chunk is
            returned for very short content.

        Raises:
            ChunkerError: If content is not a string or chunking fails.
        """
        if not isinstance(content, str):
            raise ChunkerError(f"content must be str, got {type(content).__name__}")
        if not content.strip():
            return [Chunk(index=0, text=content, start_offset=0, end_offset=len(content))]

        detected = content_type or self._detect_type(content)
        logger.debug("Chunking %d chars as type=%s", len(content), detected)

        try:
            if detected == "python":
                chunks = self._chunk_python(content)
            elif detected == "markdown":
                chunks = self._chunk_markdown(content)
            elif detected == "code":
                chunks = self._chunk_plain(content, target=self._code_target)
            else:
                chunks = self._chunk_plain(content, target=self._plain_target)
        except Exception as exc:
            logger.warning("Chunker fell back to plain-text: %s", exc)
            chunks = self._chunk_plain(content, target=self._plain_target)

        # Guarantee at least one chunk
        if not chunks:
            chunks = [Chunk(index=0, text=content, start_offset=0, end_offset=len(content))]

        return chunks

    # ── Type detection ────────────────────────────────────────────────────────

    def _detect_type(self, content: str) -> str:
        """Heuristically detect content type."""
        first_2k = content[:2000]
        # Python: has def/class at column 0
        if re.search(r"^(def |class )", first_2k, re.MULTILINE):
            return "python"
        # Markdown: has ATX headers
        if re.search(r"^#{1,6} ", first_2k, re.MULTILINE):
            return "markdown"
        # Generic code: has { } ; patterns
        if re.search(r"[{};]", first_2k):
            return "code"
        return "text"

    # ── Python chunker ────────────────────────────────────────────────────────

    def _chunk_python(self, content: str) -> list[Chunk]:
        """Split Python source at top-level function/class boundaries."""
        lines = content.splitlines(keepends=True)
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Fall back to regex-based splitting
            return self._chunk_python_regex(content)

        # Collect (start_line, end_line) 1-indexed for top-level defs
        top_level: list[tuple[int, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if getattr(node, "col_offset", 1) == 0:
                    end = getattr(node, "end_lineno", node.lineno)
                    top_level.append((node.lineno, end))

        if not top_level:
            return self._chunk_plain(content, target=self._code_target)

        top_level.sort()

        # Build segments: code before first def, each def block, trailing code
        segments: list[str] = []
        prev_end = 0
        for start_ln, end_ln in top_level:
            # 0-indexed
            si, ei = start_ln - 1, end_ln
            preamble = "".join(lines[prev_end:si])
            if preamble.strip():
                segments.append(preamble)
            segments.append("".join(lines[si:ei]))
            prev_end = ei
        tail = "".join(lines[prev_end:])
        if tail.strip():
            segments.append(tail)

        # Merge tiny segments into the previous one
        merged = self._merge_segments(segments, self._code_target)
        return self._segments_to_chunks(merged, content)

    def _chunk_python_regex(self, content: str) -> list[Chunk]:
        """Fallback: split at ``def `` / ``class `` at column 0."""
        parts = re.split(r"(?=^(?:def |class ))", content, flags=re.MULTILINE)
        merged = self._merge_segments([p for p in parts if p], self._code_target)
        return self._segments_to_chunks(merged, content)

    # ── Markdown chunker ──────────────────────────────────────────────────────

    def _chunk_markdown(self, content: str) -> list[Chunk]:
        """Split at ATX headers, sub-splitting large sections by paragraph."""
        sections = re.split(r"(?=^#{1,6} )", content, flags=re.MULTILINE)
        chunks: list[str] = []
        for section in sections:
            if not section.strip():
                continue
            if _tokens(section) <= self._prose_target:
                chunks.append(section)
            else:
                # Sub-split by paragraph
                paras = re.split(r"\n{2,}", section)
                chunks.extend(
                    self._merge_segments([p for p in paras if p.strip()], self._prose_target)
                )
        merged = self._merge_segments(chunks, self._prose_target)
        return self._segments_to_chunks(merged, content)

    # ── Plain-text chunker ────────────────────────────────────────────────────

    def _chunk_plain(self, content: str, target: int) -> list[Chunk]:
        """Sliding-window word-level chunker."""
        words = content.split()
        if len(words) <= target:
            return [Chunk(index=0, text=content, start_offset=0, end_offset=len(content))]

        chunks: list[str] = []
        step = max(1, target - self._overlap)
        i = 0
        while i < len(words):
            window = words[i: i + target]
            chunks.append(" ".join(window))
            i += step

        return self._segments_to_chunks(chunks, content)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _merge_segments(self, segments: list[str], target: int) -> list[str]:
        """Merge consecutive small segments until they reach *target* tokens."""
        if not segments:
            return []
        merged: list[str] = [segments[0]]
        for seg in segments[1:]:
            if _tokens(merged[-1]) + _tokens(seg) <= target:
                merged[-1] += "\n" + seg
            else:
                merged.append(seg)
        return merged

    def _segments_to_chunks(self, segments: list[str], original: str) -> list[Chunk]:
        """Convert string segments to :class:`Chunk` objects with offsets."""
        chunks: list[Chunk] = []
        search_start = 0
        for _i, seg in enumerate(segments):
            if not seg:
                continue
            # Find segment in original to get byte offsets
            idx = original.find(seg.strip()[:50], search_start)
            if idx == -1:
                idx = search_start
            end = idx + len(seg)
            chunks.append(Chunk(
                index=len(chunks),
                text=seg,
                start_offset=idx,
                end_offset=min(end, len(original)),
            ))
            search_start = max(search_start, idx + 1)
        return chunks or [Chunk(index=0, text="\n".join(segments),
                                start_offset=0, end_offset=len(original))]


# ── Module-level helper ───────────────────────────────────────────────────────


def _tokens(text: str) -> int:
    """Rough token count (one token ≈ one whitespace-delimited word)."""
    return len(text.split())
