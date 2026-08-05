"""Tests for the reflex cache with TTL and decay."""

from datetime import datetime, timedelta, timezone

import pytest

from voice_reflex.cache import CacheMiss, ReflexCache, ReflexEntry, DEFAULT_TTL


class TestReflexEntry:
    def test_fresh_entry_not_expired(self):
        entry = ReflexEntry(key="k1", text="test", response={"data": 1})
        assert not entry.is_expired

    def test_entry_expired_past_ttl(self):
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={"data": 1},
            created=old_time,
            ttl_seconds=3600,  # 1 hour
        )
        assert entry.is_expired

    def test_permanent_entry_never_expires(self):
        old_time = datetime.now(timezone.utc) - timedelta(days=365)
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={"data": 1},
            created=old_time,
            ttl_seconds=float("inf"),
        )
        assert not entry.is_expired

    def test_entry_stale_at_75_percent_ttl(self):
        """Entry is stale when past 75% of its TTL."""
        created = datetime.now(timezone.utc) - timedelta(minutes=26)
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={"data": 1},
            created=created,
            ttl_seconds=30 * 60,  # 30 minutes
        )
        assert entry.is_stale

    def test_entry_not_stale_before_75_percent(self):
        created = datetime.now(timezone.utc) - timedelta(minutes=10)
        entry = ReflexEntry(
            key="k1",
            text="test",
            response={"data": 1},
            created=created,
            ttl_seconds=30 * 60,
        )
        assert not entry.is_stale

    def test_touch_updates_access(self):
        entry = ReflexEntry(key="k1", text="test", response={})
        old_accessed = entry.last_accessed
        old_count = entry.access_count

        # Small sleep to ensure timestamp difference
        import time

        time.sleep(0.01)
        entry.touch()

        assert entry.last_accessed > old_accessed
        assert entry.access_count == old_count + 1

    def test_to_dict_from_dict_roundtrip(self):
        entry = ReflexEntry(
            key="k1",
            text="check weather",
            response={"temp": 72, "conditions": "clear"},
            confidence=0.85,
            ttl_seconds=1800,
            category="weather",
        )
        d = entry.to_dict()
        restored = ReflexEntry.from_dict(d)

        assert restored.key == entry.key
        assert restored.text == entry.text
        assert restored.response == entry.response
        assert restored.confidence == entry.confidence
        assert restored.ttl_seconds == entry.ttl_seconds
        assert restored.category == entry.category


class TestReflexCacheBasics:
    def test_empty_cache(self):
        cache = ReflexCache()
        assert len(cache) == 0
        assert "anything" not in cache

    def test_put_and_get(self):
        cache = ReflexCache()
        cache.put("key1", "test text", {"response": "data"})
        assert len(cache) == 1
        assert "key1" in cache

        entry = cache.get("key1")
        assert entry.text == "test text"
        assert entry.response == {"response": "data"}

    def test_get_missing_raises(self):
        cache = ReflexCache()
        with pytest.raises(CacheMiss):
            cache.get("nonexistent")

    def test_try_get_missing_returns_none(self):
        cache = ReflexCache()
        assert cache.try_get("nonexistent") is None

    def test_put_overwrites(self):
        cache = ReflexCache()
        cache.put("key1", "text1", {"v": 1})
        cache.put("key1", "text2", {"v": 2})

        entry = cache.get("key1")
        assert entry.text == "text2"
        assert entry.response == {"v": 2}


class TestReflexCacheTTL:
    def test_expired_entry_evicted_on_access(self):
        cache = ReflexCache()
        cache.put(
            "key1",
            "test",
            {"data": 1},
            ttl_seconds=0.1,  # 100ms TTL
        )

        import time

        time.sleep(0.15)

        with pytest.raises(CacheMiss):
            cache.get("key1")

        # Entry should be removed from cache
        assert "key1" not in cache

    def test_category_ttl_lookup(self):
        cache = ReflexCache()
        assert cache.get_ttl_for_category("weather") == DEFAULT_TTL["weather"]
        assert cache.get_ttl_for_category("tide") == DEFAULT_TTL["tide"]
        assert cache.get_ttl_for_category("navigation") == DEFAULT_TTL["navigation"]
        assert cache.get_ttl_for_category("unknown") == DEFAULT_TTL["generic"]

    def test_custom_ttl_overrides(self):
        cache = ReflexCache(ttl_overrides={"weather": 60})
        assert cache.get_ttl_for_category("weather") == 60

    def test_cleanup_expired(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {}, ttl_seconds=0.01)
        cache.put("k2", "t2", {}, ttl_seconds=100)

        import time

        time.sleep(0.02)

        removed = cache.cleanup_expired()
        assert removed == 1
        assert "k1" not in cache
        assert "k2" in cache


class TestReflexCacheConfidence:
    def test_low_confidence_treated_as_miss(self):
        cache = ReflexCache(min_confidence=0.5)
        cache.put("k1", "test", {"data": 1}, confidence=0.3)

        with pytest.raises(CacheMiss) as exc_info:
            cache.get("k1")
        assert "below minimum" in str(exc_info.value.reason)

    def test_high_confidence_returns(self):
        cache = ReflexCache(min_confidence=0.5)
        cache.put("k1", "test", {"data": 1}, confidence=0.8)
        entry = cache.get("k1")
        assert entry.response == {"data": 1}


class TestReflexCacheDecay:
    def test_decay_reduces_confidence(self):
        cache = ReflexCache()
        entry = cache.put("k1", "test", {"data": 1})

        # Simulate 30 days of no access
        old_time = datetime.now(timezone.utc) - timedelta(days=30)
        entry.created = old_time
        entry.last_accessed = old_time

        results = cache.decay_all()
        assert "k1" in results
        assert results["k1"] < 1.0

    def test_recent_access_decays_slowly(self):
        cache = ReflexCache()
        entry = cache.put("k1", "test", {"data": 1})

        # Access it recently
        entry.last_accessed = datetime.now(timezone.utc) - timedelta(days=2)

        results = cache.decay_all()
        # Recency factor 0.5 → decay_rate = 0.02 * 0.5 = 0.01
        assert results["k1"] >= 0.99

    def test_old_unaccessed_decays_aggressively(self):
        cache = ReflexCache()
        entry = cache.put("k1", "test", {"data": 1})

        old_time = datetime.now(timezone.utc) - timedelta(days=100)
        entry.created = old_time
        entry.last_accessed = old_time

        results = cache.decay_all()
        # 100 days unaccessed → recency_factor = 20 → decay = 0.02 * 20 = 0.40
        assert results["k1"] <= 0.65

    def test_completely_decayed_entry_evicted(self):
        cache = ReflexCache()
        entry = cache.put("k1", "test", {"data": 1}, confidence=0.01)

        old_time = datetime.now(timezone.utc) - timedelta(days=200)
        entry.created = old_time
        entry.last_accessed = old_time

        cache.decay_all()
        assert "k1" not in cache


class TestReflexCacheEvictionSupersed:
    def test_evict_removes_entry(self):
        cache = ReflexCache()
        cache.put("k1", "test", {"data": 1})
        assert "k1" in cache

        evicted = cache.evict("k1")
        assert evicted is not None
        assert evicted.key == "k1"
        assert "k1" not in cache

    def test_evict_missing_returns_none(self):
        cache = ReflexCache()
        assert cache.evict("nonexistent") is None

    def test_supersed_replaces_entry(self):
        cache = ReflexCache()
        cache.put("k1", "old text", {"v": "old"})
        cache.supersed("k1", "new text", {"v": "new"})

        entry = cache.get("k1")
        assert entry.text == "new text"
        assert entry.response == {"v": "new"}


class TestReflexCacheStats:
    def test_stats_empty_cache(self):
        cache = ReflexCache()
        stats = cache.stats()
        assert stats["count"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_stats_with_entries(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {"d": 1}, category="weather", confidence=0.9)
        cache.put("k2", "t2", {"d": 2}, category="tide", confidence=0.7)

        stats = cache.stats()
        assert stats["count"] == 2
        assert 0.79 <= stats["avg_confidence"] <= 0.81
        assert stats["categories"]["weather"] == 1
        assert stats["categories"]["tide"] == 1

    def test_iteration(self):
        cache = ReflexCache()
        cache.put("k1", "t1", {"d": 1})
        cache.put("k2", "t2", {"d": 2})

        entries = list(cache)
        assert len(entries) == 2
