"""
Importance scoring for LLM conversation turns.

:class:`ImportanceScorer` assigns a float score in [0, 1] to each turn,
indicating how important it is to keep in the active context window versus
evicting to LLMFS storage.

Scoring is additive: a base score of 0.5 is adjusted by several signal
boosts/penalties, then clamped to [0, 1].

Example::

    from llmfs.context.importance import ImportanceScorer

    scorer = ImportanceScorer()
    score = scorer.score(
        content="```python\\ndef fix(): ...\\n```",
        role="assistant",
        turn_index=10,
        total_turns=15,
    )
    # score ≈ 0.70  (code block boost applied)
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["ImportanceWeights", "ImportanceScorer"]


# ── Weight configuration ───────────────────────────────────────────────────────


@dataclass
class ImportanceWeights:
    """Configurable weights for the :class:`ImportanceScorer`.

    All boosts are additive onto a base of 0.5.  The final value is clamped
    to [0, 1].

    Attributes:
        base: Starting score before any adjustments.
        code_block_boost: Applied when content contains a fenced code block.
        error_boost: Applied when content contains an error/traceback.
        decision_boost: Applied when content contains decision keywords.
        user_role_boost: Applied when ``role == "user"``.
        recency_boost: Applied to the most recent ``recency_window`` turns.
        recency_window: Number of trailing turns that get the recency boost.
        short_content_penalty: Applied to very short filler content.
        filler_penalty: Applied to single-word filler responses.
        short_content_threshold: Token count below which content is "short".
    """

    base: float = 0.5
    code_block_boost: float = 0.20
    error_boost: float = 0.20
    decision_boost: float = 0.15
    user_role_boost: float = 0.10
    recency_boost: float = 0.15
    recency_window: int = 3
    short_content_penalty: float = 0.10
    filler_penalty: float = 0.20
    short_content_threshold: int = 50


# ── Regex constants ────────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"```", re.MULTILINE)

_ERROR_PATTERNS = re.compile(
    r"(Traceback \(most recent call last\)|"
    r"\bError\s*:|"
    r"\bException\s*:|"
    r"\bTypeError\b|"
    r"\bValueError\b|"
    r"\bKeyError\b|"
    r"\bAttributeError\b|"
    r"\bRuntimeError\b|"
    r"\bImportError\b|"
    r"\bIndexError\b)",
    re.IGNORECASE,
)

_DECISION_KEYWORDS = re.compile(
    r"\b(decided|will use|going with|approach:|plan:|must|important|"
    r"we should|we will|i will|final decision|resolved|chosen|"
    r"concluded|agreed)\b",
    re.IGNORECASE,
)

_FILLER_WORDS = frozenset(
    {
        "ok", "okay", "sure", "thanks", "thank you", "got it", "noted",
        "understood", "alright", "yep", "yup", "yes", "no", "fine",
        "great", "good", "cool", "nice", "awesome", "perfect",
    }
)


# ── Scorer ─────────────────────────────────────────────────────────────────────


class ImportanceScorer:
    """Assigns importance scores to conversation turns.

    Args:
        weights: :class:`ImportanceWeights` instance.  Defaults to plan
            defaults.

    Example::

        scorer = ImportanceScorer()
        score = scorer.score("Traceback: ...", role="assistant",
                             turn_index=5, total_turns=10)
        # ≈ 0.70 (error boost)
    """

    def __init__(self, weights: ImportanceWeights | None = None) -> None:
        self._w = weights or ImportanceWeights()

    def score(
        self,
        content: str,
        *,
        role: str = "assistant",
        turn_index: int = 0,
        total_turns: int = 1,
    ) -> float:
        """Score a single turn.

        Args:
            content: The turn's text content.
            role: Either ``"user"`` or ``"assistant"``.
            turn_index: 0-based index of this turn in the conversation.
            total_turns: Total number of turns so far (inclusive).

        Returns:
            Float in [0, 1].  Higher = keep in active context longer.
        """
        w = self._w
        score = w.base

        tokens = _token_count(content)
        lower = content.lower().strip()

        # Positive boosts
        if _CODE_BLOCK_RE.search(content):
            score += w.code_block_boost

        if _ERROR_PATTERNS.search(content):
            score += w.error_boost

        if _DECISION_KEYWORDS.search(content):
            score += w.decision_boost

        if role == "user":
            score += w.user_role_boost

        # Recency boost: last N turns
        if total_turns > 0 and turn_index >= total_turns - w.recency_window:
            score += w.recency_boost

        # Penalties
        if tokens < w.short_content_threshold:
            score -= w.short_content_penalty

        if _is_filler(lower):
            score -= w.filler_penalty

        return max(0.0, min(1.0, score))

    def score_batch(
        self,
        turns: Sequence[dict],
    ) -> list[float]:
        """Score a list of turn dicts.

        Each dict must have ``"content"`` and ``"role"`` keys.

        Args:
            turns: Sequence of ``{"role": ..., "content": ...}`` dicts.

        Returns:
            List of floats, one per turn.
        """
        total = len(turns)
        return [
            self.score(
                t.get("content", ""),
                role=t.get("role", "assistant"),
                turn_index=i,
                total_turns=total,
            )
            for i, t in enumerate(turns)
        ]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _token_count(text: str) -> int:
    """Rough token estimate: split on whitespace."""
    return len(text.split())


def _is_filler(lower_stripped: str) -> bool:
    """Return True if the content looks like a conversational filler."""
    return lower_stripped in _FILLER_WORDS or (
        len(lower_stripped.split()) <= 3
        and lower_stripped.rstrip(".,!?") in _FILLER_WORDS
    )
