"""
CLI command implementations for LLMFS.

Each command is a Click command function. They all share a common pattern:
  1. Resolve the LLMFS storage path (``--llmfs-path`` or ``.llmfs/`` or ``~/.llmfs``).
  2. Instantiate ``MemoryFS``.
  3. Call the relevant API method.
  4. Pretty-print results using ``rich``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

__all__ = [
    "cmd_init",
    "cmd_write",
    "cmd_read",
    "cmd_search",
    "cmd_query",
    "cmd_update",
    "cmd_forget",
    "cmd_relate",
    "cmd_ls",
    "cmd_status",
    "cmd_gc",
    "cmd_serve",
    "cmd_install_mcp",
    "cmd_mount",
    "cmd_unmount",
]

# ── Shared options ─────────────────────────────────────────────────────────────

_PATH_OPTION = click.option(
    "--llmfs-path",
    envvar="LLMFS_PATH",
    default=None,
    help="Path to the LLMFS storage directory. "
         "Defaults to ~/.llmfs.",
)


def _resolve_path(llmfs_path: str | None) -> Path:
    """Return the storage path: explicit > LLMFS_PATH env > ~/.llmfs."""
    if llmfs_path:
        return Path(llmfs_path)
    return Path.home() / ".llmfs"


def _get_mem(llmfs_path: str | None):
    """Instantiate MemoryFS, showing a friendly error if deps are missing."""
    from llmfs import MemoryFS
    try:
        return MemoryFS(path=_resolve_path(llmfs_path))
    except Exception as exc:
        click.echo(f"[red]Error initialising LLMFS: {exc}[/red]", err=True)
        sys.exit(1)


# ── init ──────────────────────────────────────────────────────────────────────

@click.command()
@_PATH_OPTION
def cmd_init(llmfs_path: str | None) -> None:
    """Initialise a new LLMFS store (default: ~/.llmfs)."""
    from rich.console import Console
    console = Console()

    base = Path(llmfs_path) if llmfs_path else Path.home() / ".llmfs"
    if base.exists():
        console.print(f"[yellow]LLMFS already initialised at {base}[/yellow]")
        return

    from llmfs import MemoryFS
    MemoryFS(path=base)
    console.print(f"[green]Initialised LLMFS at {base}[/green]")
    console.print("\nNext steps:")
    console.print("  llmfs write /knowledge/hello 'Hello world'")
    console.print("  llmfs search 'hello'")
    console.print("  llmfs status")


# ── write ─────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("path")
@click.argument("content", required=False)
@click.option("--file", "-f", "input_file", type=click.Path(exists=True),
              help="Read content from file instead of argument.")
@click.option("--layer", "-l", default="knowledge",
              type=click.Choice(["short_term", "session", "knowledge", "events"]),
              help="Memory layer.")
@click.option("--tags", "-t", default="",
              help="Comma-separated tags.")
@click.option("--ttl", type=int, default=None,
              help="TTL in minutes (auto-expires).")
@_PATH_OPTION
def cmd_write(
    path: str,
    content: str | None,
    input_file: str | None,
    layer: str,
    tags: str,
    ttl: int | None,
    llmfs_path: str | None,
) -> None:
    """Write content to PATH in memory.\n\nExamples:\n\n  llmfs write /k/note 'some text'\n\n  llmfs write /k/doc --file README.md"""
    from rich.console import Console
    console = Console()
    err_console = Console(stderr=True)

    if input_file:
        text = Path(input_file).read_text(encoding="utf-8")
    elif content:
        text = content
    else:
        # Read from stdin
        text = click.get_text_stream("stdin").read()

    if not text.strip():
        err_console.print("[red]No content provided.[/red]")
        sys.exit(1)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    mem = _get_mem(llmfs_path)
    obj = mem.write(path, text, layer=layer, tags=tag_list, ttl_minutes=ttl, source="cli")
    console.print(f"[green]Stored[/green] {obj.path} ({len(obj.chunks)} chunks, layer={obj.layer})")


# ── read ──────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("path")
@click.option("--query", "-q", default=None,
              help="Focus the read on a specific sub-query.")
@_PATH_OPTION
def cmd_read(path: str, query: str | None, llmfs_path: str | None) -> None:
    """Read a memory by PATH."""
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    err_console = Console(stderr=True)

    mem = _get_mem(llmfs_path)
    try:
        obj = mem.read(path, query=query)
    except Exception as exc:
        err_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    console.print(Panel(
        obj.content,
        title=f"[bold]{obj.path}[/bold]  layer={obj.layer}  "
              f"tags={obj.tags}",
        expand=False,
    ))
    console.print(f"[dim]created: {obj.metadata.created_at}  "
                  f"modified: {obj.metadata.modified_at}[/dim]")


# ── search ────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("query")
@click.option("--layer", "-l", default=None,
              type=click.Choice(["short_term", "session", "knowledge", "events"]),
              help="Restrict search to a layer.")
@click.option("--tags", "-t", default="", help="Comma-separated required tags.")
@click.option("--k", default=5, show_default=True, help="Number of results.")
@click.option("--time", "time_range", default=None,
              help='Time range, e.g. "last 7 days".')
@_PATH_OPTION
def cmd_search(
    query: str,
    layer: str | None,
    tags: str,
    k: int,
    time_range: str | None,
    llmfs_path: str | None,
) -> None:
    """Semantic search across stored memories."""
    from rich.console import Console
    from rich.table import Table
    console = Console()

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    mem = _get_mem(llmfs_path)
    results = mem.search(query, layer=layer, tags=tag_list, k=k, time_range=time_range)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    table = Table(title=f'Search: "{query}"', show_lines=True)
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Path", style="bold")
    table.add_column("Layer", width=10)
    table.add_column("Tags")
    table.add_column("Snippet")

    for r in results:
        snippet = r.chunk_text[:80].replace("\n", " ") + ("…" if len(r.chunk_text) > 80 else "")
        table.add_row(
            f"{r.score:.2f}",
            r.path,
            r.metadata.get("layer", ""),
            ", ".join(r.tags),
            snippet,
        )
    console.print(table)


# ── update ────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("path")
@click.option("--content", "-c", default=None, help="Replace full content.")
@click.option("--append", "-a", default=None, help="Append text to existing content.")
@click.option("--tags-add", default="", help="Comma-separated tags to add.")
@click.option("--tags-remove", default="", help="Comma-separated tags to remove.")
@_PATH_OPTION
def cmd_update(
    path: str,
    content: str | None,
    append: str | None,
    tags_add: str,
    tags_remove: str,
    llmfs_path: str | None,
) -> None:
    """Update an existing memory at PATH."""
    from rich.console import Console
    console = Console()
    err_console = Console(stderr=True)

    ta = [t.strip() for t in tags_add.split(",") if t.strip()] or None
    tr = [t.strip() for t in tags_remove.split(",") if t.strip()] or None

    mem = _get_mem(llmfs_path)
    try:
        obj = mem.update(path, content=content, append=append, tags_add=ta, tags_remove=tr)
    except Exception as exc:
        err_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    console.print(f"[green]Updated[/green] {obj.path}")


# ── forget ────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("path", required=False)
@click.option("--layer", "-l", default=None,
              type=click.Choice(["short_term", "session", "knowledge", "events"]),
              help="Delete all memories in a layer.")
@click.option("--older-than", default=None,
              help='Delete memories older than duration, e.g. "7 days".')
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
@_PATH_OPTION
def cmd_forget(
    path: str | None,
    layer: str | None,
    older_than: str | None,
    yes: bool,
    llmfs_path: str | None,
) -> None:
    """Delete memories by PATH, layer, or age."""
    from rich.console import Console
    console = Console()
    err_console = Console(stderr=True)

    if not path and not layer and not older_than:
        err_console.print("[red]Provide a path, --layer, or --older-than.[/red]")
        sys.exit(1)

    desc = path or (f"layer={layer}" if layer else f"older than {older_than}")
    if not yes:
        click.confirm(f"Delete memories matching: {desc}?", abort=True)

    mem = _get_mem(llmfs_path)
    try:
        result = mem.forget(path, layer=layer, older_than=older_than)
    except Exception as exc:
        err_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    console.print(f"[green]Deleted {result['deleted']} memor{'y' if result['deleted']==1 else 'ies'}.[/green]")


# ── relate ────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("source")
@click.argument("target")
@click.argument("relationship")
@click.option("--strength", default=0.8, show_default=True,
              help="Edge weight 0.0–1.0.")
@_PATH_OPTION
def cmd_relate(
    source: str,
    target: str,
    relationship: str,
    strength: float,
    llmfs_path: str | None,
) -> None:
    """Create a relationship between two memories."""
    from rich.console import Console
    console = Console()
    err_console = Console(stderr=True)

    mem = _get_mem(llmfs_path)
    try:
        result = mem.relate(source, target, relationship, strength=strength)
    except Exception as exc:
        err_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    console.print(
        f"[green]Related[/green] {source} --[{relationship}]--> {target} "
        f"(strength={strength}, id={result['relationship_id'][:8]}…)"
    )


# ── ls ────────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("path_prefix", default="/")
@click.option("--layer", "-l", default=None,
              type=click.Choice(["short_term", "session", "knowledge", "events"]))
@_PATH_OPTION
def cmd_ls(path_prefix: str, layer: str | None, llmfs_path: str | None) -> None:
    """List memories under PATH_PREFIX."""
    from rich.console import Console
    from rich.table import Table
    console = Console()

    mem = _get_mem(llmfs_path)
    objects = mem.list(path_prefix, layer=layer)

    if not objects:
        console.print("[yellow]No memories found.[/yellow]")
        return

    table = Table(title=f"Memories under {path_prefix!r}", show_lines=False)
    table.add_column("Path", style="bold")
    table.add_column("Layer", width=10)
    table.add_column("Tags")
    table.add_column("Modified")
    table.add_column("Chunks", justify="right")

    for obj in objects:
        table.add_row(
            obj.path,
            obj.layer,
            ", ".join(obj.tags),
            obj.metadata.modified_at[:19] if obj.metadata.modified_at else "",
            str(len(obj.chunks)),
        )
    console.print(table)


# ── status ────────────────────────────────────────────────────────────────────

@click.command()
@_PATH_OPTION
def cmd_status(llmfs_path: str | None) -> None:
    """Show LLMFS storage statistics."""
    from rich.console import Console
    from rich.table import Table
    console = Console()

    mem = _get_mem(llmfs_path)
    info = mem.status()

    console.print(f"\n[bold]LLMFS Status[/bold]  ({info['base_path']})\n")
    console.print(f"  Total memories : {info['total']}")
    console.print(f"  Total chunks   : {info['chunks']}")
    console.print(f"  Disk usage     : {info['disk_mb']} MB\n")

    table = Table(title="By Layer", show_header=True)
    table.add_column("Layer")
    table.add_column("Count", justify="right")
    for layer, count in sorted(info["layers"].items()):
        table.add_row(layer, str(count))
    console.print(table)


# ── gc ────────────────────────────────────────────────────────────────────────

@click.command()
@_PATH_OPTION
def cmd_gc(llmfs_path: str | None) -> None:
    """Garbage-collect expired (TTL) memories."""
    from rich.console import Console
    console = Console()

    mem = _get_mem(llmfs_path)
    result = mem.gc()
    console.print(f"[green]GC complete.[/green] Deleted {result['deleted']} expired memories.")


# ── serve ──────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--stdio", "transport", flag_value="stdio", default=True,
              help="Use stdio transport (default, for MCP clients).")
@click.option("--port", "port", type=int, default=None,
              help="Use SSE transport on this port (e.g. --port 8765).")
@_PATH_OPTION
def cmd_serve(transport: str, port: int | None, llmfs_path: str | None) -> None:
    """Start the LLMFS MCP server.

    \b
    Examples:
      llmfs serve --stdio         # stdio transport (for Claude, Cursor, etc.)
      llmfs serve --port 8765     # SSE transport on port 8765
    """
    from llmfs.mcp.server import LLMFSMCPServer
    mem = _get_mem(llmfs_path)
    server = LLMFSMCPServer(mem=mem)
    if port is not None:
        server.run_sse(port=port)
    else:
        server.run_stdio()


# ── install-mcp ────────────────────────────────────────────────────────────────

@click.command("install-mcp")
@click.option(
    "--client", "-c",
    type=click.Choice(["claude", "cursor", "continue", "windsurf"]),
    default=None,
    help="MCP client to install config for.",
)
@click.option("--print", "print_only", is_flag=True,
              help="Print the config JSON to stdout without writing.")
@_PATH_OPTION
def cmd_install_mcp(
    client: str | None,
    print_only: bool,
    llmfs_path: str | None,
) -> None:
    """Install LLMFS as an MCP server in a supported client.

    \b
    Examples:
      llmfs install-mcp --client claude
      llmfs install-mcp --client cursor
      llmfs install-mcp --print
    """
    import json

    from rich.console import Console

    from llmfs.mcp.server import generate_mcp_config, install_mcp_config

    console = Console()
    err_console = Console(stderr=True)
    resolved_path = str(_resolve_path(llmfs_path)) if llmfs_path else None

    if print_only or client is None:
        config = generate_mcp_config(llmfs_path=resolved_path)
        console.print(json.dumps(config, indent=2))
        return

    try:
        result = install_mcp_config(client, llmfs_path=resolved_path)
    except Exception as exc:
        err_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    console.print(
        f"[green]Installed[/green] LLMFS MCP config for [bold]{client}[/bold] "
        f"at {result['path']}"
    )


# ── query ──────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("mql")
@click.option("--json", "output_json", is_flag=True,
              help="Output results as JSON array.")
@_PATH_OPTION
def cmd_query(mql: str, output_json: bool, llmfs_path: str | None) -> None:
    """Execute an MQL query against the memory store.

    \b
    Examples:
      llmfs query 'SELECT memory FROM /knowledge WHERE SIMILAR TO "auth bug"'
      llmfs query 'SELECT memory FROM / WHERE TAG = "error" LIMIT 5'
      llmfs query 'SELECT memory FROM /events WHERE date > 2026-01-01' --json
    """
    import json as _json

    from rich.console import Console
    from rich.table import Table
    console = Console()
    err_console = Console(stderr=True)

    mem = _get_mem(llmfs_path)
    try:
        results = mem.query(mql)
    except Exception as exc:
        err_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    if not results:
        console.print("[yellow]No results.[/yellow]")
        return

    if output_json:
        rows = [
            {
                "path": r.path,
                "score": r.score,
                "layer": r.metadata.get("layer", ""),
                "tags": r.tags,
                "snippet": r.chunk_text[:200],
            }
            for r in results
        ]
        click.echo(_json.dumps(rows, indent=2))
        return

    table = Table(title=f"MQL: {mql[:60]}{'…' if len(mql) > 60 else ''}", show_lines=True)
    table.add_column("Score", style="cyan", width=6)
    table.add_column("Path", style="bold")
    table.add_column("Layer", width=10)
    table.add_column("Tags")
    table.add_column("Snippet")

    for r in results:
        snippet = r.chunk_text[:80].replace("\n", " ") + ("…" if len(r.chunk_text) > 80 else "")
        table.add_row(
            f"{r.score:.2f}",
            r.path,
            r.metadata.get("layer", ""),
            ", ".join(r.tags),
            snippet,
        )
    console.print(table)


# ── mount ──────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("mountpoint")
@click.option("--layer", "-l", default="knowledge",
              type=click.Choice(["short_term", "session", "knowledge", "events"]),
              help="Default write layer for new files.")
@click.option("--foreground/--background", default=True, show_default=True,
              help="Run in foreground (blocks) or background.")
@_PATH_OPTION
def cmd_mount(
    mountpoint: str,
    layer: str,
    foreground: bool,
    llmfs_path: str | None,
) -> None:
    """Mount LLMFS as a FUSE filesystem at MOUNTPOINT.

    Requires: pip install llmfs[fuse]

    \b
    Examples:
      llmfs mount /mnt/llmfs
      llmfs mount /mnt/llmfs --layer session --background
    """
    from rich.console import Console
    console = Console()
    err_console = Console(stderr=True)

    try:
        from llmfs.integrations.fuse_mount import mount
    except ImportError as exc:
        err_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    mem = _get_mem(llmfs_path)
    console.print(f"[green]Mounting LLMFS at {mountpoint}[/green]  (Ctrl-C to unmount)")
    try:
        mount(mountpoint, layer=layer, foreground=foreground, mem=mem)
    except Exception as exc:
        err_console.print(f"[red]Mount failed: {exc}[/red]")
        sys.exit(1)


# ── unmount ────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("mountpoint")
def cmd_unmount(mountpoint: str) -> None:
    """Unmount a FUSE filesystem at MOUNTPOINT.

    \b
    Examples:
      llmfs unmount /mnt/llmfs
    """
    from rich.console import Console
    console = Console()
    err_console = Console(stderr=True)

    try:
        from llmfs.integrations.fuse_mount import unmount
    except ImportError as exc:
        err_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    try:
        unmount(mountpoint)
    except Exception as exc:
        err_console.print(f"[red]{exc}[/red]")
        sys.exit(1)

    console.print(f"[green]Unmounted {mountpoint}[/green]")

