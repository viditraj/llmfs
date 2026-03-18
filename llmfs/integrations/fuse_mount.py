"""
Optional FUSE filesystem mount for LLMFS.

Exposes LLMFS memories as a read/write filesystem directory.  Each memory
is presented as a plain text file at its path (relative to the mount point).

Requirements
------------
Install with ``pip install llmfs[fuse]`` (adds ``fusepy``).

Usage
-----
Via CLI::

    llmfs mount /mnt/llmfs --path ~/.llmfs
    # ... do work, memories appear as files ...
    llmfs unmount /mnt/llmfs

Via Python::

    from llmfs.integrations.fuse_mount import mount, unmount, LLMFSFuse
    import threading

    t = threading.Thread(target=mount, args=("/mnt/llmfs",), daemon=True)
    t.start()
    # ...
    unmount("/mnt/llmfs")

Notes
-----
* Linux and macOS only (requires FUSE kernel module).
* Paths inside the mount point map 1-to-1 to LLMFS memory paths.
* Directories are virtual — listing ``/knowledge/`` shows all memories whose
  path starts with ``/knowledge/``.
* Files are written to the ``knowledge`` layer by default; the layer can be
  overridden via the ``--layer`` CLI option.
"""
from __future__ import annotations

import errno
import logging
import os
import stat
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llmfs.core.filesystem import MemoryFS

__all__ = ["LLMFSFuse", "mount", "unmount"]

logger = logging.getLogger(__name__)

_FUSE_AVAILABLE = False
try:
    from fuse import FUSE, FuseOSError, LoggingMixIn, Operations  # type: ignore[import]
    _FUSE_AVAILABLE = True
except ImportError:
    # Create dummy base classes so the class body is still parseable
    class LoggingMixIn:  # type: ignore[no-redef]
        pass

    class Operations:  # type: ignore[no-redef]
        pass

    class FuseOSError(OSError):  # type: ignore[no-redef]
        pass

    FUSE = None  # type: ignore[assignment,misc]


def _require_fuse() -> None:
    if not _FUSE_AVAILABLE:
        raise ImportError(
            "fusepy is required for the FUSE mount feature. "
            "Install it with: pip install llmfs[fuse]"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _stat_file(size: int, mtime: float) -> dict[str, Any]:
    now = int(time.time())
    return {
        "st_mode": stat.S_IFREG | 0o644,
        "st_nlink": 1,
        "st_size": size,
        "st_ctime": now,
        "st_mtime": int(mtime),
        "st_atime": now,
        "st_uid": os.getuid(),
        "st_gid": os.getgid(),
    }


def _stat_dir() -> dict[str, Any]:
    now = int(time.time())
    return {
        "st_mode": stat.S_IFDIR | 0o755,
        "st_nlink": 2,
        "st_size": 0,
        "st_ctime": now,
        "st_mtime": now,
        "st_atime": now,
        "st_uid": os.getuid(),
        "st_gid": os.getgid(),
    }


def _normalize_path(path: str) -> str:
    """Ensure path starts with /."""
    return path if path.startswith("/") else f"/{path}"


# ── LLMFSFuse ─────────────────────────────────────────────────────────────────


class LLMFSFuse(LoggingMixIn, Operations):
    """FUSE filesystem backed by LLMFS.

    Args:
        mem: :class:`~llmfs.core.filesystem.MemoryFS` instance.
        default_layer: Layer to use when writing new files.  Defaults to
            ``"knowledge"``.
    """

    def __init__(self, mem: MemoryFS, default_layer: str = "knowledge") -> None:
        self._mem = mem
        self._default_layer = default_layer
        # In-memory open file handles: {fh: {"path": ..., "data": bytes}}
        self._handles: dict[int, dict[str, Any]] = {}
        self._next_fh: int = 1

    # ── Stat ──────────────────────────────────────────────────────────────────

    def getattr(self, path: str, fh: Any = None) -> dict[str, Any]:
        if path == "/":
            return _stat_dir()

        mem_path = _normalize_path(path)

        # Try exact match first
        obj = self._mem.read(mem_path)
        if obj is not None:
            raw = obj.content.encode("utf-8")
            ts = (
                float(obj.metadata.modified_at or obj.metadata.created_at or 0)
                if isinstance(obj.metadata.modified_at, (int, float))
                else time.time()
            )
            return _stat_file(len(raw), ts)

        # Check if it's a virtual directory (any memory starts with this prefix)
        objects = self._mem.list(mem_path, recursive=True)
        if objects:
            return _stat_dir()

        raise FuseOSError(errno.ENOENT)

    # ── Directory listing ─────────────────────────────────────────────────────

    def readdir(self, path: str, fh: Any) -> list[str]:
        mem_path = _normalize_path(path)
        base = mem_path.rstrip("/")

        objects = self._mem.list(mem_path, recursive=True)

        # Collect immediate children (both files and directories)
        children: set[str] = set()
        for obj in objects:
            rest = obj.path[len(base):].lstrip("/")
            if not rest:
                continue
            part = rest.split("/")[0]
            if part:
                children.add(part)

        return [".", ".."] + sorted(children)

    # ── File open/release ─────────────────────────────────────────────────────

    def open(self, path: str, flags: int) -> int:
        mem_path = _normalize_path(path)
        obj = self._mem.read(mem_path)
        data = obj.content.encode("utf-8") if obj else b""

        fh = self._next_fh
        self._next_fh += 1
        self._handles[fh] = {"path": mem_path, "data": data}
        return fh

    def release(self, path: str, fh: int) -> None:
        self._handles.pop(fh, None)

    # ── Read ──────────────────────────────────────────────────────────────────

    def read(self, path: str, size: int, offset: int, fh: int) -> bytes:
        mem_path = _normalize_path(path)
        handle = self._handles.get(fh)
        if handle is not None:
            data = handle["data"]
        else:
            obj = self._mem.read(mem_path)
            if obj is None:
                raise FuseOSError(errno.ENOENT)
            data = obj.content.encode("utf-8")

        return data[offset:offset + size]

    # ── Create / Write ────────────────────────────────────────────────────────

    def create(self, path: str, mode: int) -> int:
        mem_path = _normalize_path(path)
        # Create empty memory
        self._mem.write(mem_path, "", layer=self._default_layer, source="fuse")
        fh = self._next_fh
        self._next_fh += 1
        self._handles[fh] = {"path": mem_path, "data": b""}
        return fh

    def write(self, path: str, data: bytes, offset: int, fh: int) -> int:
        mem_path = _normalize_path(path)

        handle = self._handles.get(fh)
        if handle is not None:
            buf = bytearray(handle["data"])
            end = offset + len(data)
            if len(buf) < end:
                buf.extend(b"\x00" * (end - len(buf)))
            buf[offset:end] = data
            handle["data"] = bytes(buf)
            # Persist to LLMFS
            self._mem.write(
                mem_path,
                handle["data"].decode("utf-8", errors="replace"),
                layer=self._default_layer,
                source="fuse",
            )
        else:
            obj = self._mem.read(mem_path)
            existing = obj.content.encode("utf-8") if obj else b""
            buf = bytearray(existing)
            end = offset + len(data)
            if len(buf) < end:
                buf.extend(b"\x00" * (end - len(buf)))
            buf[offset:end] = data
            self._mem.write(
                mem_path,
                bytes(buf).decode("utf-8", errors="replace"),
                layer=self._default_layer,
                source="fuse",
            )

        return len(data)

    def truncate(self, path: str, length: int, fh: Any = None) -> None:
        mem_path = _normalize_path(path)
        obj = self._mem.read(mem_path)
        content = (obj.content if obj else "")
        raw = content.encode("utf-8")[:length]
        self._mem.write(
            mem_path,
            raw.decode("utf-8", errors="replace"),
            layer=self._default_layer,
            source="fuse",
        )
        if fh is not None and fh in self._handles:
            self._handles[fh]["data"] = raw

    # ── Delete ────────────────────────────────────────────────────────────────

    def unlink(self, path: str) -> None:
        mem_path = _normalize_path(path)
        self._mem.forget(path=mem_path)

    def rmdir(self, path: str) -> None:
        """Remove a virtual directory (deletes all memories under path)."""
        mem_path = _normalize_path(path)
        objects = self._mem.list(mem_path, recursive=True)
        for obj in objects:
            self._mem.forget(path=obj.path)

    # ── Rename ────────────────────────────────────────────────────────────────

    def rename(self, old: str, new: str) -> None:
        old_path = _normalize_path(old)
        new_path = _normalize_path(new)
        obj = self._mem.read(old_path)
        if obj is None:
            raise FuseOSError(errno.ENOENT)
        self._mem.write(
            new_path,
            obj.content,
            layer=obj.layer,
            tags=list(obj.tags),
            source="fuse",
        )
        self._mem.forget(path=old_path)

    # ── mkdir / chmod — no-ops for compatibility ──────────────────────────────

    def mkdir(self, path: str, mode: int) -> None:
        pass  # Directories are virtual

    def chmod(self, path: str, mode: int) -> None:
        pass

    def chown(self, path: str, uid: int, gid: int) -> None:
        pass

    def utimens(self, path: str, times: Any = None) -> None:
        pass


# ── Mount helpers ─────────────────────────────────────────────────────────────


def mount(
    mountpoint: str,
    memory_path: str = "~/.llmfs",
    layer: str = "knowledge",
    foreground: bool = True,
    mem: MemoryFS | None = None,
) -> None:
    """Mount LLMFS at *mountpoint*.

    Args:
        mountpoint: Directory to mount the filesystem at.  Must exist.
        memory_path: Path to the LLMFS storage directory.
        layer: Default write layer for new files.
        foreground: Run in foreground (blocks until Ctrl-C).
        mem: Optional pre-existing :class:`~llmfs.core.filesystem.MemoryFS`.
    """
    _require_fuse()
    from llmfs import MemoryFS

    fs_mem = mem or MemoryFS(path=memory_path)
    fuse_ops = LLMFSFuse(fs_mem, default_layer=layer)
    logger.info("Mounting LLMFS at %s", mountpoint)
    FUSE(fuse_ops, mountpoint, nothreads=True, foreground=foreground, allow_other=False)


def unmount(mountpoint: str) -> None:
    """Unmount a FUSE filesystem at *mountpoint*.

    Args:
        mountpoint: Directory to unmount.
    """
    import platform
    import subprocess

    system = platform.system()
    if system == "Darwin":
        cmd = ["diskutil", "unmount", mountpoint]
    else:
        cmd = ["fusermount", "-u", mountpoint]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to unmount {mountpoint}: {result.stderr.strip()}"
        )
    logger.info("Unmounted %s", mountpoint)
