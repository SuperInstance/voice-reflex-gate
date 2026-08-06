"""Extended tests for the reflex cache — edge cases, decay tiers, serialization, stress."""

from datetime import datetime, timedelta, timezone

import pytest

from voice_reflex.cache import (
    BASE_DECAY_RATE,
    CacheMiss,
    DEFAULT_TTL,
    ReflexCache,
    ReflexEntry,
)


class TestReflexEntryExpired:
    def test_expired_at_exact_boundary(self):
        """Entry at exactly its TTL boundary is expired (age > ttl)."""
        created = datetime.now(timezone.utc) - timedelta(seconds=10)
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={},
            created=created,
            ttl_seconds=10,
        )
        # Age is slightly > 10s due to execution time
        assert entry.is_expired

    def test_not_expired_just_under_ttl(self):
        created = datetime.now(timezone.utc) - timedelta(seconds=5)
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={},
            created=created,
            ttl_seconds=10,
        )
        assert not entry.is_expired

    def test_permanent_ttl_zero_age_not_expired(self):
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={},
            ttl_seconds=float("inf"),
        )
        assert not entry.is_expired

    def test_zero_ttl_is_immediately_expired(self):
        """TTL of 0 means immediately expired."""
        created = datetime.now(timezone.utc) - timedelta(microseconds=1)
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={},
            created=created,
            ttl_seconds=0,
        )
        assert entry.is_expired


class TestReflexEntryStale:
    def test_permanent_entry_never_stale(self):
        """Entry with inf TTL should never be stale."""
        old = datetime.now(timezone.utc) - timedelta(days=3650)
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={},
            created=old,
            ttl_seconds=float("inf"),
        )
        assert not entry.is_stale

    def test_stale_at_exactly_75_percent(self):
        """Entry is stale at exactly 75% of TTL."""
        ttl = 40  # seconds
        created = datetime.now(timezone.utc) - timedelta(seconds=30)
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={},
            created=created,
            ttl_seconds=ttl,
        )
        # 30/40 = 75%
        assert entry.is_stale

    def test_not_stale_at_74_percent(self):
        ttl = 100
        created = datetime.now(timezone.utc) - timedelta(seconds=73)
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={},
            created=created,
            ttl_seconds=ttl,
        )
        assert not entry.is_stale


class TestReflexEntryTouch:
    def test_touch_increments_count_multiple_times(self):
        entry = ReflexEntry(key="k1", text="test", response={})
        for _ in range(5):
            entry.touch()
        assert entry.access_count == 5

    def test_touch_updates_last_accessed_monotonic(self):
        entry = ReflexEntry(key="k1", text="test", response={})
        import time

        times = []
        for _ in range(3):
            entry.touch()
            times.append(entry.last_accessed)
            time.sleep(0.001)

        # Each access should be >= previous
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]


class TestReflexEntryDecay:
    def test_decay_with_explicit_now(self):
        entry = ReflexEntry(key="k1", text="test", response={})
        now = datetime.now(timezone.utc) + timedelta(days=10)
        result = entry.apply_decay(now)
        assert result < 1.0

    def test_decay_recency_7_days_exact(self):
        """At exactly 7 days, recency factor is 0.5."""
        entry = ReflexEntry(key="k1", text="test", response={})
        now = datetime.now(timezone.utc)
        entry.created = now - timedelta(days=7)
        entry.last_accessed = now - timedelta(days=7)

        original = entry.confidence
        entry.apply_decay(now)
        # days_since_access = 7 → recency_factor = 0.5
        # decay = 0.02 * 0.5 = 0.01
        assert entry.confidence == pytest.approx(original - 0.01, abs=0.001)

    def test_decay_recency_8_days_normal(self):
        """At 8 days, recency factor is 1.0 (normal decay)."""
        entry = ReflexEntry(key="k1", text="test", response={})
        now = datetime.now(timezone.utc)
        entry.created = now - timedelta(days=8)
        entry.last_accessed = now - timedelta(days=8)

        original = entry.confidence
        entry.apply_decay(now)
        # days_since_access = 8 → recency_factor = 1.0
        # decay = 0.02 * 1.0 = 0.02
        assert entry.confidence == pytest.approx(original - 0.02, abs=0.001)

    def test_decay_recency_30_days_exact(self):
        """At exactly 30 days, recency factor is 1.0."""
        entry = ReflexEntry(key="k1", text="test", response={})
        now = datetime.now(timezone.utc)
        entry.created = now - timedelta(days=30)
        entry.last_accessed = now - timedelta(days=30)

        original = entry.confidence
        entry.apply_decay(now)
        # days_since_access = 30 → recency_factor = 1.0
        assert entry.confidence == pytest.approx(original - 0.02, abs=0.001)

    def test_decay_recency_31_days_accelerated(self):
        """At 31 days, recency factor jumps to 7.5."""
        entry = ReflexEntry(key="k1", text="test", response={})
        now = datetime.now(timezone.utc)
        entry.created = now - timedelta(days=31)
        entry.last_accessed = now - timedelta(days=31)

        original = entry.confidence
        entry.apply_decay(now)
        # days_since_access = 31 → recency_factor = 7.5
        # decay = 0.02 * 7.5 = 0.15
        assert entry.confidence == pytest.approx(original - 0.15, abs=0.01)

    def test_decay_recency_90_days_exact(self):
        """At exactly 90 days, still in accelerated tier (7.5)."""
        entry = ReflexEntry(key="k1", text="test", response={})
        now = datetime.now(timezone.utc)
        entry.created = now - timedelta(days=90)
        entry.last_accessed = now - timedelta(days=90)

        original = entry.confidence
        entry.apply_decay(now)
        assert entry.confidence == pytest.approx(original - 0.15, abs=0.01)

    def test_decay_recency_91_days_severe(self):
        """At 91 days, recency factor is 20.0 (severe)."""
        entry = ReflexEntry(key="k1", text="test", response={})
        now = datetime.now(timezone.utc)
        entry.created = now - timedelta(days=91)
        entry.last_accessed = now - timedelta(days=91)

        original = entry.confidence
        entry.apply_decay(now)
        # decay = 0.02 * 20.0 = 0.40
        assert entry.confidence == pytest.approx(original - 0.40, abs=0.01)

    def test_decay_never_goes_negative(self):
        entry = ReflexEntry(key="k1", text="test", response={}, confidence=0.01)
        now = datetime.now(timezone.utc)
        entry.created = now - timedelta(days=500)
        entry.last_accessed = now - timedelta(days=500)

        result = entry.apply_decay(now)
        assert result >= 0.0
        assert entry.confidence == 0.0

    def test_decay_recently_created_minimal_decay(self):
        """Fresh entry (0 days since creation) should decay minimally."""
        entry = ReflexEntry(key="k1", text="test", response={})
        now = datetime.now(timezone.utc)
        # Just created — 0 days
        result = entry.apply_decay(now)
        # days_since_creation = ~0, days_since_access = ~0
        # recency_factor = 0.5
        # decay = 0.02 * 0.5 = 0.01
        assert result == pytest.approx(0.99, abs=0.001)


class TestReflexEntrySerialization:
    def test_roundtrip_with_context_snapshot(self):
        entry = ReflexEntry(
            key="k1",
            text="check depth",
            response={"depth": 15.5, "unit": "fathoms"},
            confidence=0.85,
            ttl_seconds=3600,
            category="navigation",
            context_snapshot={"time_of_day": "night", "operational_mode": "cruising"},
        )
        d = entry.to_dict()
        restored = ReflexEntry.from_dict(d)

        assert restored.context_snapshot == entry.context_snapshot
        assert restored.category == entry.category

    def test_roundtrip_preserves_access_count(self):
        entry = ReflexEntry(key="k1", text="test", response={})
        for _ in range(10):
            entry.touch()
        d = entry.to_dict()
        restored = ReflexEntry.from_dict(d)
        assert restored.access_count == 10

    def test_from_dict_defaults_missing_fields(self):
        """from_dict should handle missing optional fields."""
        minimal = {
            "key": "k1",
            "text": "test",
            "response": {"d": 1},
            "created": datetime.now(timezone.utc).isoformat(),
            "last_accessed": datetime.now(timezone.utc).isoformat(),
        }
        restored = ReflexEntry.from_dict(minimal)
        assert restored.access_count == 0
        assert restored.confidence == 1.0
        assert restored.category == "generic"
        assert restored.context_snapshot == {}
        assert restored.ttl_seconds == DEFAULT_TTL["generic"]

    def test_to_dict_has_all_fields(self):
        entry = ReflexEntry(key="k1", text="test", response={"d": 1})
        d = entry.to_dict()
        expected_keys = {
            "key", "text", "response", "created", "last_accessed",
            "access_count", "confidence", "ttl_seconds", "category",
            "context_snapshot",
        }
        assert set(d.keys()) == expected_keys


class TestReflexCacheTTLCategories:
    def test_all_default_ttls_present(self):
        """All default TTL categories should be defined."""
        assert "weather" in DEFAULT_TTL
        assert "tide" in DEFAULT_TTL
        assert "navigation" in DEFAULT_TTL
        assert "generic" in DEFAULT_TTL
        assert "permanent" in DEFAULT_TTL

    def test_default_ttls_reasonable(self):
        assert DEFAULT_TTL["weather"] > 0
        assert DEFAULT_TTL["tide"] > DEFAULT_TTL["weather"]
        assert DEFAULT_TTL["navigation"] > DEFAULT_TTL["tide"]
        assert DEFAULT_TTL["permanent"] == float("inf")

    def test_put_with_explicit_ttl_overrides_category(self):
        cache = ReflexCache()
        cache.put("k1", "test", {}, category="weather", ttl_seconds=60)
        assert cache.get_ttl_for_category("weather") != 60  # category lookup is separate
        # But the entry itself should have the explicit TTL
        entry = cache._entries["k1"]
        assert entry.ttl_seconds == 60

    def test_put_with_permanent_category(self):
        cache = ReflexCache()
        cache.put("k1", "specs", {"vessel": "Test"}, category="permanent")
        entry = cache.get("k1")
        assert entry.ttl_seconds == float("inf")
        assert not entry.is_expired


class TestReflexCacheGetEdgeCases:
    def test_get_updates_access_count(self):
        cache = ReflexCache()
        cache.put("k1", "test", {"d": 1})
        entry = cache.get("k1")
        assert entry.access_count == 1

        entry2 = cache.get("k1")
        assert entry2.access_count == 2

    def test_get_updates_last_accessed(self):
        import time

        cache = ReflexCache()
        cache.put("k1", "test", {"d": 1})
        first = cache.get("k1")
        time.sleep(0.02)
        second = cache.get("k1")
        assert second.last_accessed >= first.last_accessed

    def test_expired_entry_removed_from_iteration(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {}, ttl_seconds=0.01)
        cache.put("k2", "t2", {})

        import time
        time.sleep(0.02)

        # Trigger get on expired entry to evict it
        with pytest.raises(CacheMiss):
            cache.get("k1")

        entries = list(cache)
        assert len(entries) == 1
        assert entries[0].key == "k2"

    def test_cache_miss_reason_not_found(self):
        cache = ReflexCache()
        try:
            cache.get("missing")
        except CacheMiss as e:
            assert e.reason == "not in cache"
            assert e.key == "missing"

    def test_cache_miss_reason_expired(self):
        cache = ReflexCache()
        cache.put("k1", "test", {}, ttl_seconds=0.01)
        import time
        time.sleep(0.02)
        try:
            cache.get("k1")
        except CacheMiss as e:
            assert e.reason == "expired"

    def test_cache_miss_reason_confidence(self):
        cache = ReflexCache(min_confidence=0.5)
        cache.put("k1", "test", {}, confidence=0.1)
        try:
            cache.get("k1")
        except CacheMiss as e:
            assert "below minimum" in e.reason


class TestReflexCacheSupersedWithKwargs:
    def test_supersed_preserves_category(self):
        cache = ReflexCache()
        cache.put("k1", "old", {"v": 1}, category="weather")
        cache.supersed("k1", "new", {"v": 2}, category="tide")
        entry = cache.get("k1")
        assert entry.category == "tide"
        assert entry.text == "new"

    def test_supersed_with_confidence(self):
        cache = ReflexCache()
        cache.put("k1", "old", {"v": 1})
        cache.supersed("k1", "new", {"v": 2}, confidence=0.5)
        entry = cache.get("k1")
        assert entry.confidence == 0.5

    def test_supersed_on_missing_key_just_puts(self):
        """Superseding a key that doesn't exist should just insert."""
        cache = ReflexCache()
        entry = cache.supersed("k1", "test", {"v": 1})
        assert entry.key == "k1"
        assert "k1" in cache


class TestReflexCacheDecayAll:
    def test_decay_all_multiple_entries(self):
        cache = ReflexCache()
        e1 = cache.put("k1", "t1", {})
        e2 = cache.put("k2", "t2", {})

        now = datetime.now(timezone.utc)
        old = now - timedelta(days=50)
        e1.created = old
        e1.last_accessed = old
        e2.created = old
        e2.last_accessed = old

        results = cache.decay_all()
        assert len(results) == 2
        assert all(v < 1.0 for v in results.values())

    def test_decay_all_empty_cache(self):
        cache = ReflexCache()
        results = cache.decay_all()
        assert results == {}

    def test_decay_all_with_explicit_now(self):
        cache = ReflexCache()
        e1 = cache.put("k1", "t1", {})
        now = datetime.now(timezone.utc) + timedelta(days=100)
        results = cache.decay_all(now=now)
        assert "k1" in results
        assert results["k1"] < 1.0

    def test_decay_all_evicts_zero_confidence(self):
        cache = ReflexCache()
        e1 = cache.put("k1", "t1", {}, confidence=0.005)
        old = datetime.now(timezone.utc) - timedelta(days=500)
        e1.created = old
        e1.last_accessed = old

        cache.decay_all()
        assert "k1" not in cache


class TestReflexCacheCleanup:
    def test_cleanup_all_expired(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {}, ttl_seconds=0.01)
        cache.put("k2", "t2", {}, ttl_seconds=0.01)

        import time
        time.sleep(0.02)

        removed = cache.cleanup_expired()
        assert removed == 2
        assert len(cache) == 0

    def test_cleanup_none_expired(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {})
        cache.put("k2", "t2", {})
        removed = cache.cleanup_expired()
        assert removed == 0
        assert len(cache) == 2

    def test_cleanup_permanent_not_removed(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {}, category="permanent")
        removed = cache.cleanup_expired()
        assert removed == 0


class TestReflexCacheStatsExtended:
    def test_stats_many_entries(self):
        cache = ReflexCache()
        for i in range(100):
            cache.put(f"k{i}", f"t{i}", {"i": i}, category="weather" if i % 2 == 0 else "tide")

        stats = cache.stats()
        assert stats["count"] == 100
        assert stats["avg_confidence"] == 1.0  # all fresh
        assert stats["categories"]["weather"] == 50
        assert stats["categories"]["tide"] == 50

    def test_stats_single_entry(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {"d": 1})
        stats = cache.stats()
        assert stats["count"] == 1
        assert stats["avg_confidence"] == 1.0
        assert stats["total_accesses"] == 0

    def test_stats_tracks_accesses(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {"d": 1})
        cache.get("k1")
        cache.get("k1")
        cache.get("k1")
        stats = cache.stats()
        assert stats["total_accesses"] == 3


class TestReflexCacheStress:
    def test_large_number_of_entries(self):
        cache = ReflexCache()
        for i in range(1000):
            cache.put(f"key_{i}", f"text_{i}", {"index": i})

        assert len(cache) == 1000

        # Random retrieval
        entry = cache.get("key_500")
        assert entry.response == {"index": 500}

    def test_rapid_put_get_cycle(self):
        cache = ReflexCache()
        for i in range(500):
            cache.put(f"k{i}", f"t{i}", {"i": i})
            entry = cache.get(f"k{i}")
            assert entry.response["i"] == i

    def test_entries_property_is_copy(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {})
        entries = cache.entries
        entries["k2"] = ReflexEntry(key="k2", text="t2", response={})
        assert "k2" not in cache

    def test_contains_after_eviction(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {})
        assert "k1" in cache
        cache.evict("k1")
        assert "k1" not in cache
