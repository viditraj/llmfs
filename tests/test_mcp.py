"""
Tests for llmfs.mcp — tool definitions, handlers, server, and prompts.
"""
import json
import tempfile
from pathlib import Path

import pytest
from llmfs import MemoryFS
from llmfs.mcp.tools import TOOL_DEFINITIONS, handle_tool_call
from llmfs.mcp.prompts import LLMFS_SYSTEM_PROMPT, get_prompt
from llmfs.mcp.server import LLMFSMCPServer, generate_mcp_config, install_mcp_config


@pytest.fixture
def mem(tmp_path):
    return MemoryFS(path=tmp_path / "llmfs")


# ── Tool definition schema tests ──────────────────────────────────────────────


class TestToolDefinitions:
    EXPECTED_NAMES = {
        "memory_write", "memory_search", "memory_read",
        "memory_update", "memory_forget", "memory_relate",
        "memory_graph", "memory_list",
    }

    def test_eight_tools_defined(self):
        assert len(TOOL_DEFINITIONS) == 8

    def test_all_names_present(self):
        names = {t["name"] for t in TOOL_DEFINITIONS}
        assert names == self.EXPECTED_NAMES

    def test_each_has_description(self):
        for tool in TOOL_DEFINITIONS:
            assert "description" in tool
            assert len(tool["description"]) > 10

    def test_each_has_input_schema(self):
        for tool in TOOL_DEFINITIONS:
            assert "inputSchema" in tool
            schema = tool["inputSchema"]
            assert schema["type"] == "object"
            assert "properties" in schema

    def test_memory_write_requires_path_and_content(self):
        write_tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "memory_write")
        assert "path" in write_tool["inputSchema"]["required"]
        assert "content" in write_tool["inputSchema"]["required"]

    def test_memory_search_requires_query(self):
        search_tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "memory_search")
        assert "query" in search_tool["inputSchema"]["required"]

    def test_memory_relate_requires_source_target_relationship(self):
        relate_tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "memory_relate")
        required = relate_tool["inputSchema"]["required"]
        assert "source" in required
        assert "target" in required
        assert "relationship" in required


# ── Tool handler tests ────────────────────────────────────────────────────────


class TestHandleToolCall:
    def test_memory_write_ok(self, mem):
        result = handle_tool_call(
            "memory_write",
            {"path": "/k/test", "content": "Hello world"},
            mem,
        )
        assert result["status"] == "ok"
        assert result["path"] == "/k/test"

    def test_memory_write_with_tags(self, mem):
        result = handle_tool_call(
            "memory_write",
            {"path": "/k/tagged", "content": "tagged content",
             "layer": "knowledge", "tags": ["foo", "bar"]},
            mem,
        )
        assert result["status"] == "ok"
        assert "foo" in result["tags"]

    def test_memory_search_ok(self, mem):
        handle_tool_call("memory_write", {"path": "/k/auth", "content": "auth bug at line 45"}, mem)
        result = handle_tool_call("memory_search", {"query": "authentication", "k": 3}, mem)
        assert result["status"] == "ok"
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_memory_read_ok(self, mem):
        handle_tool_call("memory_write", {"path": "/k/read_me", "content": "readable content"}, mem)
        result = handle_tool_call("memory_read", {"path": "/k/read_me"}, mem)
        assert result["status"] == "ok"
        assert "readable" in result["content"]

    def test_memory_read_not_found(self, mem):
        result = handle_tool_call("memory_read", {"path": "/nonexistent"}, mem)
        assert result["status"] == "error"

    def test_memory_update_append(self, mem):
        handle_tool_call("memory_write", {"path": "/k/update_me", "content": "original"}, mem)
        result = handle_tool_call(
            "memory_update",
            {"path": "/k/update_me", "append": " appended"},
            mem,
        )
        assert result["status"] == "ok"

    def test_memory_forget_by_path(self, mem):
        handle_tool_call("memory_write", {"path": "/k/to_delete", "content": "bye"}, mem)
        result = handle_tool_call("memory_forget", {"path": "/k/to_delete"}, mem)
        assert result.get("deleted", 0) >= 1 or result.get("status") == "ok"

    def test_memory_forget_no_criteria_error(self, mem):
        result = handle_tool_call("memory_forget", {}, mem)
        assert result["status"] == "error"

    def test_memory_relate_ok(self, mem):
        handle_tool_call("memory_write", {"path": "/k/src", "content": "source"}, mem)
        handle_tool_call("memory_write", {"path": "/k/tgt", "content": "target"}, mem)
        result = handle_tool_call(
            "memory_relate",
            {"source": "/k/src", "target": "/k/tgt", "relationship": "related_to"},
            mem,
        )
        assert result.get("status") == "ok"

    def test_unknown_tool_returns_error(self, mem):
        result = handle_tool_call("nonexistent_tool", {}, mem)
        assert result["status"] == "error"
        assert "Unknown tool" in result["error"]

    def test_tool_exception_returns_error(self, mem):
        # memory_relate with missing target
        result = handle_tool_call(
            "memory_relate",
            {"source": "/k/missing_src", "target": "/k/missing_tgt",
             "relationship": "related_to"},
            mem,
        )
        assert result["status"] == "error"


# ── Prompts tests ──────────────────────────────────────────────────────────────


class TestPrompts:
    def test_system_prompt_non_empty(self):
        assert len(LLMFS_SYSTEM_PROMPT) > 100

    def test_system_prompt_mentions_tools(self):
        assert "memory_write" in LLMFS_SYSTEM_PROMPT
        assert "memory_search" in LLMFS_SYSTEM_PROMPT
        assert "memory_read" in LLMFS_SYSTEM_PROMPT

    def test_system_prompt_mentions_layers(self):
        assert "session" in LLMFS_SYSTEM_PROMPT
        assert "knowledge" in LLMFS_SYSTEM_PROMPT

    def test_get_prompt_basic(self):
        text = get_prompt()
        assert "LLMFS" in text

    def test_get_prompt_with_index(self, mem):
        session_id = "prompt_test"
        mem.write(f"/session/{session_id}/turns/1", "Test turn", layer="session",
                  tags=["turn", "user"])
        text = get_prompt(include_index=True, mem=mem, session_id=session_id)
        assert "LLMFS Memory Index" in text

    def test_get_prompt_no_index_if_no_mem(self):
        text = get_prompt(include_index=True, mem=None, session_id="x")
        assert "LLMFS" in text
        # Should not error, just return base prompt
        assert "memory_write" in text


# ── MCP Server construction test ──────────────────────────────────────────────


class TestLLMFSMCPServer:
    def test_server_creation(self, mem):
        server = LLMFSMCPServer(mem=mem)
        assert server is not None

    def test_server_has_mcp_attribute(self, mem):
        server = LLMFSMCPServer(mem=mem)
        assert hasattr(server, "_mcp")

    def test_tools_registered(self, mem):
        """FastMCP should have eight tools registered."""
        import asyncio
        server = LLMFSMCPServer(mem=mem)
        tools = asyncio.run(server._mcp.list_tools())
        tool_names = {t.name for t in tools}
        assert {"memory_write", "memory_search", "memory_read",
                "memory_update", "memory_forget", "memory_relate",
                "memory_graph", "memory_list"} == tool_names


# ── Config generation tests ───────────────────────────────────────────────────


class TestGenerateMcpConfig:
    def test_returns_dict(self):
        config = generate_mcp_config()
        assert isinstance(config, dict)
        assert "mcpServers" in config
        assert "llmfs" in config["mcpServers"]

    def test_command_is_llmfs(self):
        config = generate_mcp_config()
        assert config["mcpServers"]["llmfs"]["command"] == "llmfs"

    def test_args_include_stdio(self):
        config = generate_mcp_config()
        assert "--stdio" in config["mcpServers"]["llmfs"]["args"]

    def test_custom_path_in_args(self):
        config = generate_mcp_config(llmfs_path="/custom/path")
        args = config["mcpServers"]["llmfs"]["args"]
        assert "--llmfs-path" in args
        assert "/custom/path" in args


class TestInstallMcpConfig:
    def test_dry_run_returns_config(self):
        result = install_mcp_config("cursor", dry_run=True)
        assert result["status"] == "dry_run"
        assert "config" in result
        assert "path" in result

    def test_unknown_client_raises(self):
        with pytest.raises(ValueError, match="Unknown client"):
            install_mcp_config("unknown_client")

    def test_write_creates_file(self, tmp_path, monkeypatch):
        """install_mcp_config should write to the config path."""
        from llmfs.mcp import server as srv_mod

        config_file = tmp_path / "mcp.json"
        monkeypatch.setitem(srv_mod._CLIENT_PATHS, "cursor", config_file)

        result = install_mcp_config("cursor")
        assert result["status"] == "ok"
        assert config_file.exists()
        data = json.loads(config_file.read_text())
        assert "mcpServers" in data
        assert "llmfs" in data["mcpServers"]

    def test_write_merges_with_existing(self, tmp_path, monkeypatch):
        """install_mcp_config should merge rather than overwrite existing config."""
        from llmfs.mcp import server as srv_mod

        config_file = tmp_path / "mcp.json"
        # Pre-existing config
        config_file.write_text(json.dumps({
            "mcpServers": {"other_tool": {"command": "other"}}
        }))
        monkeypatch.setitem(srv_mod._CLIENT_PATHS, "cursor", config_file)

        install_mcp_config("cursor")
        data = json.loads(config_file.read_text())
        assert "other_tool" in data["mcpServers"]
        assert "llmfs" in data["mcpServers"]


# ── CLI serve / install-mcp smoke tests ──────────────────────────────────────


class TestCLICommands:
    def test_install_mcp_print(self, tmp_path):
        from click.testing import CliRunner
        from llmfs.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, [
            "install-mcp", "--print",
            "--llmfs-path", str(tmp_path / "llmfs"),
        ])
        assert result.exit_code == 0
        assert "llmfs" in result.output

    def test_serve_help(self):
        from click.testing import CliRunner
        from llmfs.cli.main import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["serve", "--help"])
        assert result.exit_code == 0
        assert "stdio" in result.output.lower() or "port" in result.output.lower()
