"""
Extractive summarizer for LLMFS.

Uses TF-IDF sentence scoring to produce two levels of summary:

- **Level 1** — per-chunk summary: the top-N most informative sentences
  from each chunk (default N=2).
- **Level 2** — document summary: the top-N most informative sentences
  across the full content (default N=3).

The implementation is intentionally lightweight — pure stdlib + scikit-learn
(already a declared dependency) — so it works offline with no GPU.

Example::

    from llmfs.compression.summarizer import ExtractiveSummarizer

    s = ExtractiveSummarizer()
    level1 = s.summarize_chunks(["Chunk A text...", "Chunk B text..."])
    level2 = s.summarize_document("Full document text...", max_sentences=3)
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence

__all__ = ["ExtractiveSummarizer"]

logger = logging.getLogger(__name__)

# Minimum sentence length to be eligible as a summary sentence (in chars)
_MIN_SENTENCE_CHARS = 20
# Maximum number of sentences to consider for TF-IDF (avoid memory pressure)
_MAX_SENTENCES_FOR_TFIDF = 512


def _split_sentences(text: str) -> list[str]:
    """Split *text* into sentences using a simple rule-based splitter.

    Handles abbreviations (e.g. ``Mr.``, ``U.S.A.``) by requiring that the
    character after the period is an uppercase letter or digit.

    Args:
        text: Raw text to split.

    Returns:
        List of non-empty sentence strings.
    """
    # Replace common abbreviations' periods with a placeholder to avoid splitting
    cleaned = re.sub(r"\b([A-Z]\.)+", lambda m: m.group(0).replace(".", "▪"), text)
    # Split on sentence-ending punctuation followed by whitespace + uppercase
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\d\"])", cleaned)
    sentences = []
    for p in parts:
        sentence = p.replace("▪", ".").strip()
        if len(sentence) >= _MIN_SENTENCE_CHARS:
            sentences.append(sentence)
    return sentences or [text.strip()]


def _tfidf_scores(sentences: list[str]) -> list[float]:
    """Score each sentence by its mean TF-IDF weight.

    Args:
        sentences: List of sentences to score.

    Returns:
        List of float scores, one per sentence (higher = more informative).
        Falls back to uniform scores if scikit-learn is unavailable or if
        there are fewer than 2 sentences.
    """
    if len(sentences) < 2:
        return [1.0] * len(sentences)
    try:
        import numpy as np  # type: ignore[import]
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import]

        limited = sentences[:_MAX_SENTENCES_FOR_TFIDF]
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(limited)
        # Mean TF-IDF weight per sentence
        scores_array: list[float] = np.asarray(matrix.mean(axis=1)).flatten().tolist()
        # Pad with zeros if we capped
        return scores_array + [0.0] * (len(sentences) - len(limited))
    except Exception as exc:  # pragma: no cover
        logger.warning("TF-IDF scoring failed (%s), using fallback", exc)
        return [1.0] * len(sentences)


def _top_sentences(text: str, max_sentences: int) -> str:
    """Return the *max_sentences* most informative sentences from *text*.

    Sentences are returned **in their original order** (not ranked order) so
    the summary reads naturally.

    Args:
        text: Source text.
        max_sentences: Maximum number of sentences to include.

    Returns:
        Concatenated summary string, or the original *text* if it is very
        short.
    """
    if not text.strip():
        return ""

    sentences = _split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    scores = _tfidf_scores(sentences)
    # Pair each sentence with its original index and score
    indexed = sorted(enumerate(zip(scores, sentences, strict=False)), key=lambda x: -x[1][0])
    top_indices = sorted(i for i, _ in indexed[:max_sentences])
    selected = [sentences[i] for i in top_indices]
    return " ".join(selected)


class ExtractiveSummarizer:
    """TF-IDF extractive summarizer for LLMFS memory objects.

    Produces two levels of summary for each memory:

    - **Level 1** (per-chunk): short summaries of individual text chunks to
      enable fast chunk-level retrieval and deduplication.
    - **Level 2** (document): a single summary of the full content for quick
      skimming in the memory index.

    Args:
        chunk_sentences: Maximum sentences per chunk summary (default 2).
        doc_sentences: Maximum sentences for the document summary (default 3).

    Example::

        s = ExtractiveSummarizer(chunk_sentences=2, doc_sentences=3)
        chunk_summaries = s.summarize_chunks(["First chunk ...", "Second chunk ..."])
        doc_summary     = s.summarize_document("Full content ...")
    """

    def __init__(
        self,
        chunk_sentences: int = 2,
        doc_sentences: int = 3,
    ) -> None:
        if chunk_sentences < 1:
            raise ValueError("chunk_sentences must be >= 1")
        if doc_sentences < 1:
            raise ValueError("doc_sentences must be >= 1")
        self._chunk_sentences = chunk_sentences
        self._doc_sentences = doc_sentences

    # ── Public API ────────────────────────────────────────────────────────────

    def summarize_chunks(self, chunk_texts: Sequence[str]) -> list[str]:
        """Produce a per-chunk extractive summary for each chunk text.

        Args:
            chunk_texts: Sequence of raw chunk strings (in chunk order).

        Returns:
            List of summary strings, one per chunk.  Empty chunks yield ``""``.
        """
        return [_top_sentences(t, self._chunk_sentences) for t in chunk_texts]

    def summarize_document(
        self,
        content: str,
        max_sentences: int | None = None,
    ) -> str:
        """Produce a document-level extractive summary.

        Args:
            content: Full raw content of the memory.
            max_sentences: Override the instance default.  Must be >= 1 if
                provided.

        Returns:
            Summary string (may be shorter than requested if the content has
            fewer sentences).
        """
        n = max_sentences if max_sentences is not None else self._doc_sentences
        if n < 1:
            raise ValueError("max_sentences must be >= 1")
        return _top_sentences(content, n)

    def summarize_all(
        self,
        content: str,
        chunk_texts: Sequence[str],
    ) -> tuple[list[str], str]:
        """Convenience wrapper that returns both summary levels at once.

        Args:
            content: Full raw content for the document-level summary.
            chunk_texts: Individual chunk texts for per-chunk summaries.

        Returns:
            ``(level_1_summaries, level_2_summary)`` tuple.
        """
        level1 = self.summarize_chunks(chunk_texts)
        level2 = self.summarize_document(content)
        return level1, level2

    def __repr__(self) -> str:
        return (
            f"ExtractiveSummarizer("
            f"chunk_sentences={self._chunk_sentences}, "
            f"doc_sentences={self._doc_sentences})"
        )
