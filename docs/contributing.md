# Contributing

Contributions are welcome! LLMFS is an early-stage project and there are many ways to help.

## Getting Started

```bash
# Fork and clone
git clone https://github.com/viditraj/llmfs.git
cd llmfs

# Create a branch
git checkout -b feature/my-improvement

# Install in editable mode with dev dependencies
pip install -e ".[dev,mcp,openai,langchain]"

# Run the test suite
pytest --cov=llmfs --cov-report=term-missing

# Run linting
ruff check llmfs/ tests/
```

## Running Tests

```bash
# All tests
pytest

# Specific module
pytest tests/test_filesystem.py -v

# With coverage report
pytest --cov=llmfs --cov-report=html
open htmlcov/index.html

# Fast: skip slow embedding tests
pytest -m "not slow"
```

## Areas for Contribution

- **New embedders** -- Add adapters for Cohere, Mistral, or local Ollama models
- **Retrieval improvements** -- Better score fusion, cross-encoder reranking
- **MQL extensions** -- Additional condition types, subqueries, aggregations
- **Graph algorithms** -- PageRank-based memory importance, community detection
- **Streaming support** -- Streaming writes for real-time transcript ingestion
- **Windows FUSE** -- WinFsp-based FUSE mount for Windows
- **UI** -- A web dashboard for browsing and editing memories

## Submitting a Pull Request

1. **Fork** the repository and create a feature branch from `main`
2. **Write tests** for any new behavior -- we target 90%+ coverage
3. **Run the full suite** -- `pytest && ruff check .`
4. **Update the README** if you're adding a user-facing feature
5. **Open a PR** with a clear description of what the change does and why

## Reporting Issues

Use the [GitHub issue tracker](https://github.com/viditraj/llmfs/issues). For bugs, please include:

- LLMFS version (`pip show llmfs`)
- Python version
- OS
- Minimal reproduction script
