"""llmfs.mcp — MCP server and tool definitions."""
from llmfs.mcp.prompts import LLMFS_SYSTEM_PROMPT, get_prompt
from llmfs.mcp.server import LLMFSMCPServer, generate_mcp_config, install_mcp_config
from llmfs.mcp.tools import TOOL_DEFINITIONS, handle_tool_call

__all__ = [
    "TOOL_DEFINITIONS",
    "handle_tool_call",
    "LLMFSMCPServer",
    "generate_mcp_config",
    "install_mcp_config",
    "LLMFS_SYSTEM_PROMPT",
    "get_prompt",
]
