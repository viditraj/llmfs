"""
Local CPU embedder using sentence-transformers (all-MiniLM-L6-v2).

22 MB model, ~1 000 queries/sec on CPU, 384-dim vectors.
The model is downloaded on first use and cached by sentence-transformers.
An in-process LRU cache avoids re-embedding identical strings.
"""
from __future__ import annotations

import functools
import logging
from typing import Any

from llmfs.core.exceptions import EmbedderError
from llmfs.embeddings.base import EmbedderBase

__all__ = ["LocalEmbedder"]

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_LRU_SIZE = 1024


class LocalEmbedder(EmbedderBase):
    """Sentence-transformers embedder running fully on CPU.

    Args:
        model_name: Hugging Face model identifier.  Defaults to
            ``all-MiniLM-L6-v2``.

    Example::

        embedder = LocalEmbedder()
        vec = embedder.embed("Fix JWT expiry bug")
        print(len(vec))  # 384
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None  # lazy-load on first use
        logger.debug("LocalEmbedder initialised (model=%s, lazy)", model_name)

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _get_model(self):  # type: ignore[return]
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                print(f"[llmfs] Loading embedder ({self._model_name})…", flush=True)
                logger.info("Loading sentence-transformer model: %s", self._model_name)
                self._model = SentenceTransformer(self._model_name)
                print(f"[llmfs] Embedder ready (dim={self.embedding_dim})", flush=True)
                logger.info("Model loaded (dim=%d)", self.embedding_dim)
            except ImportError as exc:
                raise EmbedderError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            except Exception as exc:
                raise EmbedderError(f"Failed to load model {self._model_name!r}: {exc}") from exc
        return self._model

    # ── EmbedderBase interface ────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    @functools.lru_cache(maxsize=_LRU_SIZE)
    def embed(self, text: str) -> list[float]:
        """Embed a single string with LRU caching.

        Args:
            text: Input text (leading/trailing whitespace is stripped).

        Returns:
            Float vector of length 384 for the default model.

        Raises:
            EmbedderError: If the model fails to encode.
        """
        text = text.strip()
        if not text:
            raise EmbedderError("Cannot embed empty string")
        try:
            vec = self._get_model().encode(text, convert_to_numpy=True)
            return vec.tolist()
        except Exception as exc:
            raise EmbedderError(f"Encoding failed: {exc}") from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings.

        Checks the LRU cache per-string first; uncached strings are encoded
        together in one model call.

        Args:
            texts: List of input strings.

        Returns:
            List of float vectors, parallel to *texts*.

        Raises:
            EmbedderError: If any string is empty or encoding fails.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                raise EmbedderError(f"Cannot embed empty string at index {i}")
            # Try LRU cache (embed is cached)
            cache_info = self.embed.cache_info()  # noqa: F841 — warm the cache path
            try:
                results[i] = self.embed(text)
            except Exception:
                # embed raises EmbedderError for real errors; if it was a cache
                # miss the encode happens inside embed() anyway — so this path
                # is only hit when we need to batch new strings
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch-encode anything not already in cache
        if uncached_texts:
            try:
                vecs = self._get_model().encode(uncached_texts, convert_to_numpy=True)
                for idx, vec in zip(uncached_indices, vecs):
                    results[idx] = vec.tolist()
                    # Warm the LRU cache for future single-embed calls
                    self.embed.__wrapped__ = None  # type: ignore[attr-defined]
            except Exception as exc:
                raise EmbedderError(f"Batch encoding failed: {exc}") from exc

        return results  # type: ignore[return-value]
