# Installation

## Core (CLI + Python API)

```bash
pip install llmfs
```

This installs the core package with CLI, Python API, and all dependencies:

- **ChromaDB** — embedded vector store
- **sentence-transformers** — local embedding model (`all-MiniLM-L6-v2`, ~22 MB)
- **Click** — CLI framework
- **Rich** — terminal formatting
- **scikit-learn** — TF-IDF summarization
- **NumPy** — numerical computing
- **MCP** — Model Context Protocol SDK

!!! note "First-run download"
    The first `search` or `write` call downloads the `all-MiniLM-L6-v2` embedding model (~22 MB) to your HuggingFace cache. No GPU required.

## Optional Extras

Install additional integrations as needed:

=== "MCP Server"

    ```bash
    pip install "llmfs[mcp]"
    ```
    For Claude Desktop, Cursor, Windsurf, and other MCP-compatible clients.

=== "OpenAI"

    ```bash
    pip install "llmfs[openai]"
    ```
    OpenAI function-calling integration.

=== "LangChain"

    ```bash
    pip install "llmfs[langchain]"
    ```
    Drop-in LangChain memory adapters.

=== "FUSE Mount"

    ```bash
    pip install "llmfs[fuse]"
    ```
    Mount LLMFS as a real filesystem (Linux/macOS only).

=== "Everything"

    ```bash
    pip install "llmfs[mcp,openai,langchain,fuse]"
    ```

## From Source

```bash
git clone https://github.com/viditraj/llmfs.git
cd llmfs
pip install -e ".[dev]"
pytest
```

## Requirements

- Python 3.10 or later
- Linux, macOS, or Windows
- No GPU required
