"""
Tests for llmfs.context.manager — ContextManager.
"""
import pytest
from llmfs import MemoryFS
from llmfs.context.manager import ContextManager, TurnRecord


@pytest.fixture
def mem(tmp_path):
    return MemoryFS(path=tmp_path / "llmfs")


class TestTurnRecord:
    def test_defaults(self):
        t = TurnRecord(id="1", role="user", content="Hello", tokens=5)
        assert not t.evicted
        assert t.artifact_paths == []
        assert t.importance == 0.5


class TestContextManagerBasic:
    def test_init(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=1000)
        assert ctx.session_id
        assert ctx.get_stats()["total_tokens"] == 0
        assert ctx.get_stats()["active_turns"] == 0

    def test_custom_session_id(self, mem):
        ctx = ContextManager(mem=mem, session_id="mysession")
        assert ctx.session_id == "mysession"

    def test_on_new_turn_adds_to_active(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=10_000)
        ctx.on_new_turn("user", "Hello", tokens=5)
        assert ctx.get_stats()["active_turns"] == 1

    def test_token_count_accumulates(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=10_000)
        ctx.on_new_turn("user", "Hello", tokens=5)
        ctx.on_new_turn("assistant", "World", tokens=7)
        assert ctx.get_stats()["total_tokens"] == 12

    def test_get_active_turns_returns_dicts(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=10_000)
        ctx.on_new_turn("user", "Test content here", tokens=10)
        turns = ctx.get_active_turns()
        assert len(turns) == 1
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Test content here"

    def test_get_system_prompt_addon_empty_initially(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=10_000)
        # No eviction yet → no index
        assert ctx.get_system_prompt_addon() == ""

    def test_repr(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=1000)
        r = repr(ctx)
        assert "ContextManager" in r
        assert "1000" in r


class TestContextManagerEviction:
    def test_eviction_triggered_when_threshold_exceeded(self, mem):
        # max_tokens=100, evict_at=0.7 → threshold=70 tokens
        ctx = ContextManager(mem=mem, max_tokens=100, evict_at=0.70,
                             target_after_evict=0.50)
        # Add turns until we exceed threshold
        for i in range(8):
            ctx.on_new_turn("user", f"Turn {i} some content here to fill tokens", tokens=10)

        stats = ctx.get_stats()
        assert stats["evicted_turns"] > 0

    def test_evicted_turns_stored_in_llmfs(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=100, evict_at=0.70,
                             target_after_evict=0.50,
                             session_id="evict_test")
        for i in range(8):
            ctx.on_new_turn("user", f"Turn {i} padding content here", tokens=10)

        # LLMFS should have memories under /session/evict_test/
        objects = mem.list(f"/session/evict_test", recursive=True)
        assert len(objects) > 0

    def test_memory_index_populated_after_eviction(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=100, evict_at=0.70,
                             target_after_evict=0.50)
        for i in range(8):
            ctx.on_new_turn("user", f"Turn {i} content", tokens=10)

        index = ctx.get_system_prompt_addon()
        assert "LLMFS Memory Index" in index

    def test_tokens_reduced_after_eviction(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=100, evict_at=0.70,
                             target_after_evict=0.50)
        for i in range(8):
            ctx.on_new_turn("user", f"Turn {i} content", tokens=10)

        # After eviction tokens should be at or near target (50)
        assert ctx.get_stats()["total_tokens"] <= 60  # allow some tolerance

    def test_low_importance_evicted_first(self, mem):
        """Filler turns (low importance) should be evicted before important ones."""
        ctx = ContextManager(mem=mem, max_tokens=100, evict_at=0.70,
                             target_after_evict=0.50,
                             session_id="prio_test")
        # Add a critical turn (code block → high importance)
        ctx.on_new_turn("assistant",
                        "```python\ndef fix(): return True\n```",
                        tokens=10)
        # Add filler turns to trigger eviction
        for i in range(7):
            ctx.on_new_turn("user", "ok", tokens=10)

        # The code turn should still be active (not evicted)
        active_contents = [t["content"] for t in ctx.get_active_turns()]
        code_still_active = any("```python" in c for c in active_contents)
        # At least some low-importance turns evicted
        assert ctx.get_stats()["evicted_turns"] > 0
        # Code turn should be among the active (it has highest importance)
        assert code_still_active


class TestContextManagerReset:
    def test_reset_session_clears_turns(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=10_000)
        ctx.on_new_turn("user", "Hello", tokens=5)
        ctx.reset_session()
        assert ctx.get_stats()["active_turns"] == 0
        assert ctx.get_stats()["total_tokens"] == 0

    def test_reset_session_returns_dict(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=10_000)
        result = ctx.reset_session()
        assert "status" in result
        assert result["status"] == "ok"
        assert "session_id" in result

    def test_build_memory_index_force_rebuild(self, mem):
        ctx = ContextManager(mem=mem, max_tokens=100, evict_at=0.70,
                             target_after_evict=0.50,
                             session_id="rebuild_test")
        for i in range(8):
            ctx.on_new_turn("user", f"Turn {i}", tokens=10)

        index1 = ctx.get_system_prompt_addon()
        index2 = ctx.build_memory_index()
        assert index1 == index2  # should be same content
        assert "LLMFS Memory Index" in index2
