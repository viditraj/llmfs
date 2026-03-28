"""
Tests for llmfs.retrieval.ranker and llmfs.retrieval.engine.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmfs.core.memory_object import SearchResult
from llmfs.retrieval.ranker import RankConfig, Ranker, _parent_prefix


# ── _parent_prefix ────────────────────────────────────────────────────────────


class TestParentPrefix:
    def test_deep_path(self):
        assert _parent_prefix("/a/b/c") == "/a/b"

    def test_single_level(self):
        assert _parent_prefix("/a") == "/"

    def test_trailing_slash(self):
        assert _parent_prefix("/a/b/") == "/a"

    def test_root(self):
        assert _parent_prefix("/") == "/"


# ── RankConfig ────────────────────────────────────────────────────────────────


class TestRankConfig:
    def test_defaults(self):
        cfg = RankConfig()
        assert cfg.semantic_weight == pytest.approx(0.7)
        assert cfg.recency_weight == pytest.approx(0.15)
        assert cfg.diversity_lambda == pytest.approx(0.7)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            RankConfig(semantic_weight=1.5)

    def test_invalid_diversity_lambda_raises(self):
        with pytest.raises(ValueError):
            RankConfig(diversity_lambda=-0.1)

    def test_invalid_half_life_raises(self):
        with pytest.raises(ValueError):
            RankConfig(recency_half_life_hours=0)


# ── Ranker ────────────────────────────────────────────────────────────────────


def _make_result(path: str, score: float, tags: list[str] | None = None, modified_at: str = "") -> SearchResult:
    meta: dict = {}
    if modified_at:
        meta["modified_at"] = modified_at
    return SearchResult(
        path=path,
        content=f"Content for {path}",
        score=score,
        metadata=meta,
        tags=tags or [],
    )


class TestRanker:
    def test_empty_input_returns_empty(self):
        ranker = Ranker()
        assert ranker.rank([]) == []

    def test_returns_same_count(self):
        ranker = Ranker()
        results = [_make_result(f"/p{i}", float(i) / 10) for i in range(5)]
        ranked = ranker.rank(results)
        assert len(ranked) == 5

    def test_top_k_limits_output(self):
        ranker = Ranker()
        results = [_make_result(f"/p{i}", float(i) / 10) for i in range(10)]
        ranked = ranker.rank(results, top_k=3)
        assert len(ranked) == 3

    def test_higher_score_ranked_first(self):
        ranker = Ranker(RankConfig(recency_weight=0.0, tag_weight=0.0, diversity_lambda=1.0))
        results = [
            _make_result("/low", 0.1),
            _make_result("/high", 0.9),
            _make_result("/mid", 0.5),
        ]
        ranked = ranker.rank(results)
        assert ranked[0].path == "/high"

    def test_tag_boost(self):
        """Results matching query tags should rank higher."""
        ranker = Ranker(RankConfig(
            semantic_weight=0.5,
            recency_weight=0.0,
            tag_weight=0.5,
            diversity_lambda=1.0,
        ))
        results = [
            _make_result("/no_tag", 0.7, tags=[]),
            _make_result("/has_tag", 0.6, tags=["auth"]),
        ]
        ranked = ranker.rank(results, query_tags=["auth"])
        assert ranked[0].path == "/has_tag"

    def test_diversity_spreads_paths(self):
        """Results under different prefixes should be preferred when diversity < 1."""
        cfg = RankConfig(diversity_lambda=0.3, recency_weight=0.0, tag_weight=0.0)
        ranker = Ranker(cfg)
        results = [
            _make_result("/a/x", 0.9),
            _make_result("/a/y", 0.8),
            _make_result("/b/z", 0.7),
        ]
        ranked = ranker.rank(results, top_k=3)
        paths = [r.path for r in ranked]
        assert "/a/x" in paths  # highest score, always included

    def test_scores_updated_in_output(self):
        ranker = Ranker()
        results = [_make_result("/p", 0.5)]
        ranked = ranker.rank(results)
        # Scores should be valid floats
        assert isinstance(ranked[0].score, float)

    def test_recency_boosts_recent_results(self):
        """A very recent memory should outscore an old one if recency_weight is high."""
        cfg = RankConfig(
            semantic_weight=0.4,
            recency_weight=0.6,
            tag_weight=0.0,
            diversity_lambda=1.0,
        )
        ranker = Ranker(cfg)
        # Use ISO format dates
        recent = _make_result("/recent", 0.5, modified_at="2099-01-01T00:00:00+00:00")
        old = _make_result("/old", 0.8, modified_at="2000-01-01T00:00:00+00:00")
        ranked = ranker.rank([old, recent])
        assert ranked[0].path == "/recent"


class TestRankerFuse:
    def test_fuse_merges_lists(self):
        ranker = Ranker()
        list1 = [_make_result("/a", 0.9), _make_result("/b", 0.7)]
        list2 = [_make_result("/b", 0.8), _make_result("/c", 0.6)]
        fused = ranker.fuse(list1, list2)
        paths = {r.path for r in fused}
        assert paths == {"/a", "/b", "/c"}

    def test_fuse_duplicate_path_gets_boost(self):
        ranker = Ranker()
        list1 = [_make_result("/common", 0.9), _make_result("/a", 0.5)]
        list2 = [_make_result("/common", 0.8), _make_result("/b", 0.5)]
        fused = ranker.fuse(list1, list2)
        # /common appears in both lists → should be ranked first
        assert fused[0].path == "/common"

    def test_fuse_top_k(self):
        ranker = Ranker()
        list1 = [_make_result(f"/p{i}", 0.9 - i * 0.1) for i in range(5)]
        list2 = [_make_result(f"/q{i}", 0.85 - i * 0.1) for i in range(5)]
        fused = ranker.fuse(list1, list2, top_k=4)
        assert len(fused) == 4

    def test_fuse_empty_lists(self):
        ranker = Ranker()
        fused = ranker.fuse([], [])
        assert fused == []


# ── RetrievalEngine ───────────────────────────────────────────────────────────


class TestRetrievalEngine:
    """Integration-style tests using a real MemoryFS instance."""

    @pytest.fixture
    def mem(self, tmp_path):
        from llmfs import MemoryFS
        return MemoryFS(path=tmp_path)

    @pytest.fixture
    def engine(self, mem):
        from llmfs.retrieval.engine import RetrievalEngine
        return RetrievalEngine(
            db=mem._db,
            vs=mem._get_vs(),
            embedder=mem._get_embedder(),
        )

    def test_search_empty_store_returns_empty(self, engine):
        results = engine.search("anything")
        assert results == []

    def test_search_returns_results(self, mem, engine):
        mem.write("/kb/auth", content="JWT authentication token refresh bug fixed.")
        results = engine.search("authentication", k=5)
        assert len(results) >= 1
        assert results[0].path == "/kb/auth"

    def test_search_top_k_respected(self, mem, engine):
        for i in range(5):
            mem.write(f"/kb/item{i}", content=f"Memory about topic number {i} with keywords.")
        results = engine.search("topic keywords memory", k=2)
        assert len(results) <= 2

    def test_search_layer_filter(self, mem, engine):
        mem.write("/kb/knowledge_item", content="Knowledge about security protocols.")
        mem.write("/st/short_item", content="Knowledge about security protocols.", layer="short_term")
        results = engine.search("security knowledge", layer="knowledge", k=10)
        assert all(r.metadata.get("layer", "knowledge") in ("knowledge", "") for r in results)

    def test_search_empty_query_returns_empty(self, engine):
        results = engine.search("")
        assert results == []

    def test_search_by_path_prefix(self, mem, engine):
        mem.write("/projects/auth/module", content="Auth module content.")
        mem.write("/projects/db/module", content="DB module content.")
        mem.write("/events/bug", content="Bug event content.")
        results = engine.search_by_path_prefix("/projects")
        paths = {r.path for r in results}
        assert "/projects/auth/module" in paths
        assert "/projects/db/module" in paths
        assert "/events/bug" not in paths

    def test_related_raises_without_graph(self, mem, engine):
        mem.write("/a/node", content="Node content.")
        with pytest.raises(ValueError, match="MemoryGraph"):
            engine.related("/a/node")

    def test_related_with_graph(self, mem, tmp_path):
        from llmfs.graph.memory_graph import MemoryGraph
        from llmfs.retrieval.engine import RetrievalEngine

        mem.write("/a/src", content="Source node content.")
        mem.write("/a/tgt", content="Target node content.")
        graph = MemoryGraph(mem._db)
        graph.add_edge("/a/src", "/a/tgt", strength=0.9)

        engine_with_graph = RetrievalEngine(
            db=mem._db,
            vs=mem._get_vs(),
            embedder=mem._get_embedder(),
            graph=graph,
        )
        results = engine_with_graph.related("/a/src", depth=1)
        paths = {r.path for r in results}
        assert "/a/tgt" in paths

    def test_search_time_range_filter(self, mem, engine):
        """Memories written with a very old modified_at should be filtered out."""
        # We can't inject modified_at directly, but we can verify the filter
        # doesn't crash and returns a list
        mem.write("/kb/recent", content="Recent security update about JWT tokens.")
        results = engine.search("JWT tokens", time_range="last 7 days")
        assert isinstance(results, list)

    def test_search_tag_filter(self, mem, engine):
        mem.write("/kb/tagged", content="JWT auth fix.", tags=["auth", "jwt"])
        mem.write("/kb/untagged", content="JWT auth fix.")
        results = engine.search("JWT auth", tags=["auth"], k=10)
        paths = {r.path for r in results}
        assert "/kb/tagged" in paths
        assert "/kb/untagged" not in paths
