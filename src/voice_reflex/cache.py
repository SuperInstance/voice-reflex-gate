"""
Reflex cache — stores compiled responses with TTL and decay.

Each reflex entry carries:
- A cached response (arbitrary dict)
- A temporal validity window (how long the response is fresh)
- A confidence score that decays over time
- Access metadata (created, last accessed, access count)

Three kinds of forgetting:
1. Fading — confidence decays; reflex still matchable but with lower weight
2. Supersession — old reflex replaced by a new one for the same key
3. Eviction — reflex removed entirely (dangerous or contradicted)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator


class CacheMiss(Exception):
    """Raised when a reflex key is not in the cache or is expired."""

    def __init__(self, key: str, reason: str = "not found") -> None:
        self.key = key
        self.reason = reason
        super().__init__(f"Cache miss for key {key!r}: {reason}")


# Default temporal validity windows (in seconds) — domain-specific.
DEFAULT_TTL: dict[str, float] = {
    "weather": 30 * 60,       # 30 minutes
    "tide": 6 * 3600,         # 6 hours
    "navigation": 12 * 3600,  # 12 hours
    "generic": 60 * 60,       # 1 hour fallback
    "permanent": float("inf"),  # never expires (e.g., vessel specs)
}

# Decay rates applied daily to confidence scores.
BASE_DECAY_RATE = 0.02  # 2% per day by default


@dataclass
class ReflexEntry:
    """
    A single cached reflex.

    The response is an arbitrary dict — the caller decides the structure.
    The confidence score starts at 1.0 and decays over time.
    """

    key: str
    text: str  # canonical STT text that triggered this reflex
    response: dict[str, Any]
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    confidence: float = 1.0
    ttl_seconds: float = field(default=DEFAULT_TTL["generic"])
    category: str = "generic"  # used for TTL lookup and seasonal grouping
    context_snapshot: dict[str, str] = field(default_factory=dict)  # context at creation time

    @property
    def is_expired(self) -> bool:
        """Whether the entry has exceeded its TTL."""
        if self.ttl_seconds == float("inf"):
            return False
        age = (datetime.now(timezone.utc) - self.created).total_seconds()
        return age > self.ttl_seconds

    @property
    def is_stale(self) -> bool:
        """
        Whether the entry is past 75% of its TTL — it's still usable
        but should be treated with increasing skepticism.
        """
        if self.ttl_seconds == float("inf"):
            return False
        age = (datetime.now(timezone.utc) - self.created).total_seconds()
        return age > (self.ttl_seconds * 0.75)

    def touch(self) -> None:
        """Record an access — updates last_accessed and increments count."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def apply_decay(self, now: datetime | None = None) -> float:
        """
        Apply daily decay to the confidence score.

        Decay is stronger for entries that haven't been accessed recently.
        Returns the new confidence value.
        """
        now = now or datetime.now(timezone.utc)
        days_since_creation = (now - self.created).total_seconds() / 86400
        days_since_access = (now - self.last_accessed).total_seconds() / 86400

        # Recency factor: recently-accessed reflexes decay slowly
        if days_since_access <= 7:
            recency_factor = 0.5  # halve the decay rate
        elif days_since_access <= 30:
            recency_factor = 1.0  # normal decay
        elif days_since_access <= 90:
            recency_factor = 7.5  # accelerate
        else:
            recency_factor = 20.0  # severe decay

        decay_amount = BASE_DECAY_RATE * recency_factor
        self.confidence = max(0.0, self.confidence - decay_amount)
        return self.confidence

    def to_dict(self) -> dict[str, Any]:
        """Serialize for persistence."""
        return {
            "key": self.key,
            "text": self.text,
            "response": self.response,
            "created": self.created.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "confidence": self.confidence,
            "ttl_seconds": self.ttl_seconds,
            "category": self.category,
            "context_snapshot": self.context_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReflexEntry":
        """Deserialize from dict."""
        return cls(
            key=data["key"],
            text=data["text"],
            response=data["response"],
            created=datetime.fromisoformat(data["created"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            confidence=data.get("confidence", 1.0),
            ttl_seconds=data.get("ttl_seconds", DEFAULT_TTL["generic"]),
            category=data.get("category", "generic"),
            context_snapshot=data.get("context_snapshot", {}),
        )


class ReflexCache:
    """
    The reflex cache — stores and retrieves compiled responses.

    Keyed by reflex key (hash of STT text + context vector).
    Supports TTL, decay, supersession, and eviction.
    """

    def __init__(
        self,
        *,
        min_confidence: float = 0.3,
        ttl_overrides: dict[str, float] | None = None,
    ) -> None:
        """
        Args:
            min_confidence: Below this confidence, an entry is treated as a miss
                           (it's still in the cache, but the gate won't use it).
            ttl_overrides: Override default TTL per category.
        """
        self._entries: dict[str, ReflexEntry] = {}
        self._min_confidence = min_confidence
        self._ttls = {**DEFAULT_TTL, **(ttl_overrides or {})}

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: str) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[ReflexEntry]:
        return iter(self._entries.values())

    @property
    def entries(self) -> dict[str, ReflexEntry]:
        """Return a shallow copy of all entries."""
        return dict(self._entries)

    def get_ttl_for_category(self, category: str) -> float:
        """Get the TTL for a category, falling back to generic."""
        return self._ttls.get(category, self._ttls["generic"])

    def put(
        self,
        key: str,
        text: str,
        response: dict[str, Any],
        *,
        category: str = "generic",
        context_snapshot: dict[str, str] | None = None,
        ttl_seconds: float | None = None,
        confidence: float = 1.0,
    ) -> ReflexEntry:
        """
        Store or replace a reflex entry.

        If an entry already exists for this key, it is SUPERSEDED —
        the old entry is archived and the new one takes its place.
        """
        if ttl_seconds is None:
            ttl_seconds = self.get_ttl_for_category(category)

        entry = ReflexEntry(
            key=key,
            text=text,
            response=response,
            category=category,
            context_snapshot=context_snapshot or {},
            ttl_seconds=ttl_seconds,
            confidence=confidence,
        )
        self._entries[key] = entry
        return entry

    def get(self, key: str) -> ReflexEntry:
        """
        Retrieve a reflex entry by key.

        Raises CacheMiss if:
        - Key not in cache
        - Entry is expired (past TTL)
        - Confidence below minimum threshold
        """
        entry = self._entries.get(key)
        if entry is None:
            raise CacheMiss(key, "not in cache")

        if entry.is_expired:
            # Evict expired entries on access
            del self._entries[key]
            raise CacheMiss(key, "expired")

        if entry.confidence < self._min_confidence:
            raise CacheMiss(key, f"confidence {entry.confidence:.3f} below minimum {self._min_confidence}")

        entry.touch()
        return entry

    def try_get(self, key: str) -> ReflexEntry | None:
        """Like get() but returns None instead of raising."""
        try:
            return self.get(key)
        except CacheMiss:
            return None

    def evict(self, key: str) -> ReflexEntry | None:
        """
        Actively remove a reflex entry — the hard forget.

        Use when a reflex is dangerous or contradicted by external data.
        Returns the evicted entry, or None if it wasn't in the cache.
        """
        return self._entries.pop(key, None)

    def supersed(self, key: str, text: str, response: dict[str, Any], **kwargs: Any) -> ReflexEntry:
        """
        Replace an existing reflex with a new response for the same key.

        This is the clean replace — the system knows WHY the old reflex
        was replaced, and the replacement is immediate.
        """
        # The old entry (if any) is simply replaced
        return self.put(key, text, response, **kwargs)

    def decay_all(self, now: datetime | None = None) -> dict[str, float]:
        """
        Apply daily decay to all entries.

        Returns a dict mapping keys to their new confidence values.
        Entries that decay to 0 are evicted.
        """
        results: dict[str, float] = {}
        to_evict: list[str] = []

        for key, entry in self._entries.items():
            new_conf = entry.apply_decay(now)
            results[key] = new_conf
            if new_conf <= 0.0:
                to_evict.append(key)

        for key in to_evict:
            del self._entries[key]
            results.pop(key, None)

        return results

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns the count removed."""
        expired_keys = [k for k, e in self._entries.items() if e.is_expired]
        for k in expired_keys:
            del self._entries[k]
        return len(expired_keys)

    def stats(self) -> dict[str, Any]:
        """Return summary statistics about the cache."""
        entries = list(self._entries.values())
        if not entries:
            return {
                "count": 0,
                "avg_confidence": 0.0,
                "total_accesses": 0,
                "categories": {},
            }
        return {
            "count": len(entries),
            "avg_confidence": sum(e.confidence for e in entries) / len(entries),
            "total_accesses": sum(e.access_count for e in entries),
            "categories": {cat: sum(1 for e in entries if e.category == cat) for cat in {e.category for e in entries}},
        }
