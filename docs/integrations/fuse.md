# FUSE Filesystem Mount

Mount LLMFS as a real FUSE filesystem and access memories with ordinary shell tools. Requires Linux or macOS.

## Installation

```bash
pip install "llmfs[fuse]"
```

## Usage

```bash
mkdir /tmp/memory
llmfs mount /tmp/memory

# Now you can use standard tools
ls /tmp/memory/knowledge/
cat /tmp/memory/knowledge/auth/jwt-expiry
echo "New finding: also affects refresh endpoint" >> /tmp/memory/knowledge/auth/jwt-expiry

llmfs unmount /tmp/memory
```

## Mount Options

```bash
llmfs mount /tmp/memory --layer session      # default write layer
llmfs mount /tmp/memory --background         # detach from terminal
```

## How It Works

The FUSE mount maps LLMFS paths to filesystem paths:

- **Directories** correspond to path prefixes (e.g., `/tmp/memory/knowledge/` lists all memories under `/knowledge/`)
- **Files** correspond to individual memories — reading a file returns the memory content
- **Writing** to a file creates or updates the memory at that path
- **Deleting** a file calls `forget` on that memory

This makes it possible to use LLMFS with any tool that reads/writes files: `grep`, `cat`, `vim`, IDE file explorers, backup scripts, etc.
