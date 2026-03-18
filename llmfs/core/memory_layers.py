"""
Memory layer definitions, TTL defaults, and expiry helpers.

LLMFS organises memories into four layers with different persistence semantics:

- **short_term** — ephemeral scratch space; auto-expires after 60 minutes
- **session**    — current conversation; cleared at session end
- **knowledge**  — permanent facts, learnings, code patterns
- **events**     — timestamped occurrences (bugs fixed, meetings, deployments)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llmfs.core.memory_object import MemoryObject

__all__ = ["MemoryLayer", "VALID_LAYERS", "LAYER_DEFAULT_TTL", "is_expired", "ttl_expires_at"]


class MemoryLayer(str, Enum):
    """Enumeration of the four LLMFS memory layers.

    Using ``str`` as a mixin means layer values can be compared directly to
    plain strings, which simplifies SQLite storage and JSON serialisation.
    """

    SHORT_TERM = "short_term"
    SESSION = "session"
    KNOWLEDGE = "knowledge"
    EVENTS = "events"


# ── Constants ────────────────────────────────────────────────────────────────

#: All valid layer string values. Used in :meth:`MemoryObject.validate`.
VALID_LAYERS: frozenset[str] = frozenset(layer.value for layer in MemoryLayer)

#: Default TTL in **minutes** for each layer.  ``None`` means no expiry.
LAYER_DEFAULT_TTL: dict[str, int | None] = {
    MemoryLayer.SHORT_TERM: 60,
    MemoryLayer.SESSION: None,
    MemoryLayer.KNOWLEDGE: None,
    MemoryLayer.EVENTS: None,
}

#: Eviction priority (lower index = evicted first when context fills).
LAYER_EVICTION_PRIORITY: list[str] = [
    MemoryLayer.SHORT_TERM,
    MemoryLayer.SESSION,
    MemoryLayer.EVENTS,
    MemoryLayer.KNOWLEDGE,
]


# ── Helpers ──────────────────────────────────────────────────────────────────


def ttl_expires_at(layer: str, ttl_minutes: int | None = None) -> str | None:
    """Compute the ISO-8601 UTC expiry timestamp for a memory.

    Args:
        layer: One of the four layer strings.
        ttl_minutes: Explicit TTL override. If ``None``, uses the layer
            default. Pass ``0`` to force no expiry regardless of layer.

    Returns:
        ISO-8601 UTC timestamp string, or ``None`` if the memory never expires.
    """
    if ttl_minutes == 0:
        return None

    effective_ttl = ttl_minutes if ttl_minutes is not None else LAYER_DEFAULT_TTL.get(layer)
    if effective_ttl is None:
        return None

    expires = datetime.now(timezone.utc) + timedelta(minutes=effective_ttl)
    return expires.isoformat()


def is_expired(memory: MemoryObject) -> bool:
    """Return ``True`` if *memory* has passed its TTL.

    Args:
        memory: The :class:`~llmfs.core.memory_object.MemoryObject` to check.

    Returns:
        ``True`` when the memory has a TTL and the current UTC time is past it.
        ``False`` when the memory has no TTL or has not yet expired.
    """
    ttl_str: str | None = memory.metadata.ttl  # type: ignore[assignment]
    # metadata.ttl stores the *expiry timestamp* string (set via ttl_expires_at)
    # when originally written; it may also be stored as an int (minutes) for
    # legacy reasons — we handle both.
    if ttl_str is None:
        return False
    if isinstance(ttl_str, int):
        # Shouldn't happen in new code, but be defensive.
        return False
    try:
        expiry = datetime.fromisoformat(ttl_str)
    except ValueError:
        return False
    return datetime.now(timezone.utc) >= expiry
