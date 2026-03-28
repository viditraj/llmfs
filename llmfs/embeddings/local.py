"""
Local CPU embedder using sentence-transformers (all-MiniLM-L6-v2).

22 MB model, ~1 000 queries/sec on CPU, 384-dim vectors.
The model is downloaded on first use and cached by sentence-transformers.
An in-process LRU cache avoids re-embedding identical strings.
"""
from __future__ import annotations

import functools
import hashlib
import logging
import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from llmfs.core.exceptions import EmbedderError
from llmfs.embeddings.base import EmbedderBase

if TYPE_CHECKING:
    from llmfs.storage.metadata_db import MetadataDB

__all__ = ["LocalEmbedder"]

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_LRU_SIZE = 1024


@contextmanager
def _quiet_hf():
    """Suppress noisy HuggingFace progress bars and warnings during model load."""
    prev_tok = os.environ.get("TOKENIZERS_PARALLELISM")
    prev_verb = os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS")
    try:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        # Silence the "Loading weights" tqdm bar from MLX / safetensors
        _tqdm_log = logging.getLogger("tqdm")
        _hf_log = logging.getLogger("huggingface_hub")
        old_tqdm = _tqdm_log.level
        old_hf = _hf_log.level
        _tqdm_log.setLevel(logging.ERROR)
        _hf_log.setLevel(logging.ERROR)
        yield
    finally:
        _tqdm_log.setLevel(old_tqdm)
        _hf_log.setLevel(old_hf)
        if prev_tok is None:
            os.environ.pop("TOKENIZERS_PARALLELISM", None)
        else:
            os.environ["TOKENIZERS_PARALLELISM"] = prev_tok
        if prev_verb is None:
            os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        else:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = prev_verb


class LocalEmbedder(EmbedderBase):
    """Sentence-transformers embedder running fully on CPU.

    Optionally backed by a persistent SQLite cache (via
    :class:`~llmfs.storage.metadata_db.MetadataDB`) so embeddings survive
    across process restarts.

    Args:
        model_name: Hugging Face model identifier.  Defaults to
            ``all-MiniLM-L6-v2``.
        cache_db: Optional :class:`MetadataDB` for persistent embedding cache.
            When set, ``embed`` / ``embed_batch`` check the DB before computing.

    Example::

        embedder = LocalEmbedder()
        vec = embedder.embed("Fix JWT expiry bug")
        print(len(vec))  # 384
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        cache_db: MetadataDB | None = None,
    ) -> None:
        self._model_name = model_name
        self._model: Any = None  # lazy-load on first use
        self._cache_db = cache_db
        logger.debug("LocalEmbedder initialised (model=%s, lazy, cache=%s)",
                      model_name, "db" if cache_db else "in-memory")

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _get_model(self):  # type: ignore[return]
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise EmbedderError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc

            with _quiet_hf():
                try:
                    # Fast path: load from local HF cache without network calls.
                    # SentenceTransformer normally contacts the Hub on every init
                    # (even when the model is already cached), adding ~80s of
                    # latency.  local_files_only=True skips that entirely.
                    logger.info("Loading sentence-transformer model: %s (local cache)", self._model_name)
                    self._model = SentenceTransformer(
                        self._model_name, local_files_only=True,
                    )
                    logger.info("Model loaded from cache (dim=%d)", self.embedding_dim)
                except Exception:
                    # Model not cached yet — download it for the first time.
                    print(f"[llmfs] Downloading embedder ({self._model_name})…", flush=True)
                    logger.info("Downloading sentence-transformer model: %s", self._model_name)
                    try:
                        self._model = SentenceTransformer(self._model_name)
                    except Exception as exc:
                        raise EmbedderError(f"Failed to load model {self._model_name!r}: {exc}") from exc
                    print(f"[llmfs] Embedder ready (dim={self.embedding_dim})", flush=True)
                    logger.info("Model downloaded and loaded (dim=%d)", self.embedding_dim)
        return self._model

    # ── Persistent cache helpers ──────────────────────────────────────────────

    @staticmethod
    def _text_hash(text: str) -> str:
        """SHA-256 hex digest of stripped text."""
        return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

    # ── EmbedderBase interface ────────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()

    @functools.lru_cache(maxsize=_LRU_SIZE)  # noqa: B019 — intentional: embedder is module-level singleton
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

        # Check persistent cache
        if self._cache_db is not None:
            th = self._text_hash(text)
            cached = self._cache_db.get_cached_embedding(th, self._model_name)
            if cached is not None:
                return cached

        try:
            vec = self._get_model().encode(text, convert_to_numpy=True)
            result = vec.tolist()
        except Exception as exc:
            raise EmbedderError(f"Encoding failed: {exc}") from exc

        # Persist to DB cache
        if self._cache_db is not None:
            self._cache_db.put_cached_embedding(th, result, self._model_name)

        return result

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings with persistent caching.

        Checks the persistent DB cache first, then the LRU cache, and
        batches any remaining uncached texts into a single model call.

        Args:
            texts: List of input strings.

        Returns:
            List of float vectors, parallel to *texts*.

        Raises:
            EmbedderError: If any string is empty or encoding fails.
        """
        if not texts:
            return []

        stripped = [t.strip() for t in texts]
        for i, t in enumerate(stripped):
            if not t:
                raise EmbedderError(f"Cannot embed empty string at index {i}")

        results: list[list[float] | None] = [None] * len(stripped)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # 1. Check persistent DB cache (batch lookup)
        if self._cache_db is not None:
            hashes = [self._text_hash(t) for t in stripped]
            db_hits = self._cache_db.get_cached_embeddings_batch(hashes, self._model_name)
            for i, h in enumerate(hashes):
                if h in db_hits:
                    results[i] = db_hits[h]

        # 2. Check LRU cache for remaining misses
        for i, text in enumerate(stripped):
            if results[i] is not None:
                continue
            try:
                results[i] = self.embed(text)
            except EmbedderError:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # 3. Batch-encode anything not found in either cache
        if uncached_texts:
            try:
                vecs = self._get_model().encode(uncached_texts, convert_to_numpy=True)
                new_cache_items: list[tuple[str, list[float]]] = []
                for idx, vec in zip(uncached_indices, vecs, strict=False):
                    vec_list = vec.tolist()
                    results[idx] = vec_list
                    if self._cache_db is not None:
                        new_cache_items.append(
                            (self._text_hash(stripped[idx]), vec_list)
                        )
                # Batch-persist to DB cache
                if new_cache_items and self._cache_db is not None:
                    self._cache_db.put_cached_embeddings_batch(
                        new_cache_items, self._model_name,
                    )
            except Exception as exc:
                raise EmbedderError(f"Batch encoding failed: {exc}") from exc

        return results  # type: ignore[return-value]
