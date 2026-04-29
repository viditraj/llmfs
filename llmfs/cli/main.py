"""
CLI entry point for LLMFS.

All commands are registered here and delegate to llmfs.cli.commands.
Invoke with: llmfs <command> [options]
"""
import click

from llmfs.cli.commands import (
    cmd_forget,
    cmd_gc,
    cmd_graph,
    cmd_init,
    cmd_install_mcp,
    cmd_ls,
    cmd_mount,
    cmd_query,
    cmd_read,
    cmd_relate,
    cmd_search,
    cmd_serve,
    cmd_status,
    cmd_unmount,
    cmd_update,
    cmd_write,
)


@click.group()
@click.version_option(package_name="llmfs")
def cli() -> None:
    """LLMFS — filesystem-metaphor memory for LLMs and AI agents."""


cli.add_command(cmd_init, name="init")
cli.add_command(cmd_write, name="write")
cli.add_command(cmd_read, name="read")
cli.add_command(cmd_search, name="search")
cli.add_command(cmd_query, name="query")
cli.add_command(cmd_update, name="update")
cli.add_command(cmd_forget, name="forget")
cli.add_command(cmd_relate, name="relate")
cli.add_command(cmd_graph, name="graph")
cli.add_command(cmd_ls, name="ls")
cli.add_command(cmd_status, name="status")
cli.add_command(cmd_gc, name="gc")
cli.add_command(cmd_serve, name="serve")
cli.add_command(cmd_install_mcp, name="install-mcp")
cli.add_command(cmd_mount, name="mount")
cli.add_command(cmd_unmount, name="unmount")

if __name__ == "__main__":
    cli()
