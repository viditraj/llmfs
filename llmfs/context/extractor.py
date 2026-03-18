"""
Artifact extraction for LLMFS context management.

Before a conversation turn is evicted from the active context window,
:class:`ArtifactExtractor` parses it for structured artifacts and writes
them individually to LLMFS.  This enables precise, targeted retrieval of
specific details (code, errors, decisions) even after the full turn has
been evicted.

Artifacts stored:

- **Code blocks** — ``/session/{session_id}/code/turn_{id}_{i}``
- **Stack traces / errors** — ``/session/{session_id}/errors/turn_{id}``
- **File paths mentioned** — ``/session/{session_id}/files/turn_{id}``
- **Decisions / plans** — ``/session/{session_id}/decisions/turn_{id}``

Example::

    from llmfs.context.extractor import ArtifactExtractor
    from llmfs import MemoryFS

    mem = MemoryFS(path="/tmp/test_llmfs")
    extractor = ArtifactExtractor()
    paths = extractor.extract(
        turn={"id": "1", "role": "assistant",
              "content": "```python\\nprint('hi')\\n```"},
        mem=mem,
        session_id="abc123",
    )
    # paths == ["/session/abc123/code/turn_1_0"]
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmfs.core.filesystem import MemoryFS

__all__ = ["ArtifactExtractor"]

logger = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

# Fenced code block: ```lang\n...\n```
_CODE_BLOCK_RE = re.compile(
    r"```(?P<lang>\w*)\s*\n(?P<code>.*?)```",
    re.DOTALL,
)

# Error / traceback indicators
_ERROR_INDICATOR_RE = re.compile(
    r"(?:Traceback \(most recent call last\)|"
    r"^\s*[\w.]*(?:Error|Exception|Warning)\s*:)",
    re.MULTILINE | re.IGNORECASE,
)

# Source file paths (common extensions)
_FILE_PATH_RE = re.compile(
    r"(?:^|[\s\"'`(])(?P<path>(?:\.{0,2}/)?[\w./\-]+\."
    r"(?:py|js|ts|jsx|tsx|go|rs|java|cpp|c|h|json|yaml|yml|toml|md|txt|sh|rb|cs|php))"
    r"(?:[:\s\"'`)]|$)",
    re.MULTILINE,
)

# Decision keywords (sentence-level)
_DECISION_RE = re.compile(
    r"(?:^|\.\s+|!\s+|\?\s+)(?P<sentence>[^.!?]*?"
    r"(?:decided|will use|going with|approach:|plan:|we should|we will|"
    r"i will|final decision|resolved|chosen|concluded|agreed)[^.!?]*[.!?]?)",
    re.IGNORECASE | re.MULTILINE,
)


class ArtifactExtractor:
    """Extracts structured artifacts from a conversation turn and stores them.

    Args:
        layer: Memory layer to use when writing artifacts.  Defaults to
            ``"session"``.

    Example::

        extractor = ArtifactExtractor()
        paths = extractor.extract(turn={"id": "5", "role": "user",
                                        "content": "Fix auth.py:45"},
                                  mem=mem, session_id="sess1")
    """

    def __init__(self, layer: str = "session") -> None:
        self._layer = layer

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        turn: dict[str, Any],
        mem: MemoryFS,
        session_id: str,
    ) -> list[str]:
        """Extract artifacts from *turn* and write them to *mem*.

        Args:
            turn: Dict with at least ``"id"``, ``"role"``, and ``"content"``.
            mem: :class:`~llmfs.core.filesystem.MemoryFS` instance to write to.
            session_id: Session identifier used to namespace paths.

        Returns:
            List of paths that were written to *mem* (may be empty).
        """
        turn_id = str(turn.get("id", "unknown"))
        content = turn.get("content", "")
        role = turn.get("role", "assistant")
        written: list[str] = []

        written.extend(self._extract_code_blocks(content, turn_id, role, mem, session_id))
        written.extend(self._extract_errors(content, turn_id, role, mem, session_id))
        written.extend(self._extract_file_refs(content, turn_id, role, mem, session_id))
        written.extend(self._extract_decisions(content, turn_id, role, mem, session_id))

        logger.debug(
            "extract: turn=%s session=%s artifacts=%d",
            turn_id, session_id, len(written),
        )
        return written

    def extract_and_store_full_turn(
        self,
        turn: dict[str, Any],
        mem: MemoryFS,
        session_id: str,
    ) -> str:
        """Write the *full* turn content to LLMFS as a fallback archive.

        Args:
            turn: Dict with ``"id"``, ``"role"``, and ``"content"``.
            mem: :class:`~llmfs.core.filesystem.MemoryFS` instance.
            session_id: Session identifier.

        Returns:
            The path written.
        """
        turn_id = str(turn.get("id", "unknown"))
        role = turn.get("role", "assistant")
        content = turn.get("content", "")
        path = f"/session/{session_id}/turns/{turn_id}"

        # Metadata stored inline via tags for easy filtering
        tags = ["turn", role]
        try:
            mem.write(path, content, layer=self._layer, tags=tags, source="context_manager")
        except Exception:
            logger.exception("Failed to archive full turn %s", turn_id)
        return path

    # ── Private extraction helpers ────────────────────────────────────────────

    def _extract_code_blocks(
        self,
        content: str,
        turn_id: str,
        role: str,
        mem: MemoryFS,
        session_id: str,
    ) -> list[str]:
        paths: list[str] = []
        for i, m in enumerate(_CODE_BLOCK_RE.finditer(content)):
            lang = m.group("lang") or "text"
            code = m.group("code").strip()
            if not code:
                continue
            path = f"/session/{session_id}/code/turn_{turn_id}_{i}"
            tags = ["code", lang, role]
            try:
                mem.write(path, code, layer=self._layer, tags=tags, source="context_manager")
                paths.append(path)
            except Exception:
                logger.warning("Failed to store code block at %s", path)
        return paths

    def _extract_errors(
        self,
        content: str,
        turn_id: str,
        role: str,
        mem: MemoryFS,
        session_id: str,
    ) -> list[str]:
        if not _ERROR_INDICATOR_RE.search(content):
            return []

        # Extract the error block: from first indicator line to blank line or end
        lines = content.splitlines()
        error_lines: list[str] = []
        capturing = False
        for line in lines:
            if _ERROR_INDICATOR_RE.search(line):
                capturing = True
            if capturing:
                error_lines.append(line)
                # Stop at a blank line after we've started
                if capturing and len(error_lines) > 1 and not line.strip():
                    break

        error_text = "\n".join(error_lines).strip() or content
        path = f"/session/{session_id}/errors/turn_{turn_id}"
        tags = ["error", role]
        try:
            mem.write(path, error_text, layer=self._layer, tags=tags, source="context_manager")
            return [path]
        except Exception:
            logger.warning("Failed to store error at %s", path)
            return []

    def _extract_file_refs(
        self,
        content: str,
        turn_id: str,
        role: str,
        mem: MemoryFS,
        session_id: str,
    ) -> list[str]:
        refs = list({m.group("path") for m in _FILE_PATH_RE.finditer(content)})
        if not refs:
            return []

        path = f"/session/{session_id}/files/turn_{turn_id}"
        tags = ["file_references", role]
        try:
            mem.write(
                path,
                json.dumps(refs, indent=2),
                layer=self._layer,
                tags=tags,
                source="context_manager",
            )
            return [path]
        except Exception:
            logger.warning("Failed to store file refs at %s", path)
            return []

    def _extract_decisions(
        self,
        content: str,
        turn_id: str,
        role: str,
        mem: MemoryFS,
        session_id: str,
    ) -> list[str]:
        sentences = [m.group("sentence").strip() for m in _DECISION_RE.finditer(content)]
        if not sentences:
            return []

        path = f"/session/{session_id}/decisions/turn_{turn_id}"
        tags = ["decision", role]
        decision_text = "\n".join(f"- {s}" for s in sentences if s)
        if not decision_text:
            return []
        try:
            mem.write(path, decision_text, layer=self._layer, tags=tags, source="context_manager")
            return [path]
        except Exception:
            logger.warning("Failed to store decisions at %s", path)
            return []
