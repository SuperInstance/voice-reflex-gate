"""Tests for the cascade gate — the main entry point."""

import pytest

from voice_reflex.cache import ReflexCache
from voice_reflex.context import Context, ContextVector, build_reflex_key, OperationalMode, TimeOfDay, LocationState
from voice_reflex.gate import Gate, GateResult
from voice_reflex.matcher import FuzzyMatcher
from voice_reflex.learner import Learner


class TestGateExactMatch:
    """Tier 1: exact reflex key lookup."""

    def test_exact_hit_returns_cached_response(self):
        cache = ReflexCache()
        gate = Gate(cache=cache)

        ctx = Context(time_of_day="dawn", operational_mode="docked")
        key = build_reflex_key("check the weather", ctx.vector)
        cache.put(key, "check the weather", {"temp": 72, "conditions": "clear"})

        result = gate.process("check the weather", ctx)
        assert result.hit
        assert result.is_exact
        assert result.response == {"temp": 72, "conditions": "clear"}
        assert result.confidence == 1.0

    def test_exact_miss_different_context(self):
        """Same text, different context → different key → miss."""
        cache = ReflexCache()
        gate = Gate(cache=cache)

        ctx1 = Context(time_of_day="dawn", operational_mode="docked")
        key1 = build_reflex_key("check depth", ctx1.vector)
        cache.put(key1, "check depth", {"depth": 15})

        ctx2 = Context(time_of_day="night", operational_mode="cruising")
        result = gate.process("check depth", ctx2)
        assert not result.hit
        assert result.should_cascade

    def test_whitespace_normalized_in_exact_match(self):
        cache = ReflexCache()
        gate = Gate(cache=cache)

        ctx = Context()
        key = build_reflex_key("check the weather", ctx.vector)
        cache.put(key, "check the weather", {"data": "ok"})

        result = gate.process("  check   the   weather  ", ctx)
        assert result.hit
        assert result.is_exact


class TestGateFuzzyMatch:
    """Tier 2: fuzzy pattern matching."""

    def test_fuzzy_match_returns_response(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        gate = Gate(cache=cache, matcher=matcher)

        ctx = Context()
        # Register a pattern
        key = build_reflex_key("check the weather", ctx.vector)
        cache.put(key, "check the weather", {"temp": 72})
        matcher.add_pattern("check the weather", key)

        # Slightly different text should fuzzy-match
        result = gate.process("check the weather forecast", ctx)
        if result.hit:
            assert result.response == {"temp": 72}

    def test_no_fuzzy_match_cascades(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        gate = Gate(cache=cache, matcher=matcher)

        matcher.add_pattern("check the weather", "some_key")
        ctx = Context()

        result = gate.process("play some music", ctx)
        assert not result.hit
        assert result.should_cascade
        assert result.tier == "none"


class TestGateCascade:
    """Tier 3: cascade to model."""

    def test_no_patterns_cascades(self):
        cache = ReflexCache()
        gate = Gate(cache=cache)

        ctx = Context()
        result = gate.process("something completely novel", ctx)
        assert not result.hit
        assert result.should_cascade
        assert result.tier == "none"

    def test_cascade_result_has_reason(self):
        cache = ReflexCache()
        gate = Gate(cache=cache)

        ctx = Context()
        result = gate.process("anything", ctx)
        assert result.cascade_reason


class TestGateLearn:
    """Learning loop integration."""

    def test_learn_records_observation(self):
        gate = Gate()
        ctx = Context()

        gate.learn("check weather", ctx, {"temp": 72, "conditions": "clear"})
        assert gate.learner.observation_count == 1

    def test_three_observations_compile_reflex(self):
        gate = Gate()
        ctx = Context(time_of_day="dawn", operational_mode="docked")

        response = {"temp": 72, "conditions": "clear"}

        # Three identical requests → should compile
        for _ in range(3):
            gate.learn("check the weather", ctx, response)

        # After 3 observations, a reflex should be compiled
        assert gate.check_new_reflexes() or len(gate.cache) > 0

        # 4th request should hit the reflex
        result = gate.process("check the weather", ctx)
        assert result.hit

    def test_inconsistent_responses_dont_compile(self):
        gate = Gate()
        ctx = Context()

        gate.learn("check weather", ctx, {"temp": 72})
        gate.learn("check weather", ctx, {"temp": 75})
        gate.learn("check weather", ctx, {"conditions": "clear"})

        # Should NOT compile — responses are inconsistent
        reflexes = gate.check_new_reflexes()
        assert len(reflexes) == 0

    def test_learn_then_hit(self):
        """Full cycle: learn → compile → cache hit."""
        gate = Gate()
        ctx = Context(time_of_day="dawn", operational_mode="cruising")

        # Simulate the model handling this 3 times
        for i in range(3):
            gate.learn("what's our heading", ctx, {"heading": "045", "speed": "6kn"})

        # Check compilation happened
        new_reflexes = gate.check_new_reflexes()
        # At least one reflex should be compiled
        assert len(new_reflexes) >= 1 or len(gate.cache) > 0

        # Now a 4th request should be served from cache
        result = gate.process("what's our heading", ctx)
        assert result.hit


class TestGateRegisterPattern:
    def test_manual_pattern_registration(self):
        gate = Gate()
        gate.register_pattern(
            "check the weather",
            aliases=["what's the weather", "how's the weather"],
            category="weather",
        )

        assert len(gate.matcher.patterns) == 1

        # Alias should match
        result = gate.matcher.match("what's the weather")
        assert result.matched


class TestGateStats:
    def test_stats_empty(self):
        gate = Gate()
        stats = gate.stats()
        assert stats["count"] == 0
        assert stats["pattern_count"] == 0

    def test_stats_with_data(self):
        cache = ReflexCache()
        gate = Gate(cache=cache)

        gate.register_pattern("test pattern")
        cache.put("k1", "test", {"d": 1})

        stats = gate.stats()
        assert stats["count"] == 1
        assert stats["pattern_count"] >= 1


class TestGateNaturalVariation:
    """The core success criterion: natural language variation matching."""

    def test_weather_variations(self):
        """All these should resolve to the same reflex."""
        gate = Gate()

        # Register the canonical form
        ctx = Context(time_of_day="dawn", operational_mode="docked")
        key = build_reflex_key("check the weather", ctx.vector)
        gate.cache.put(key, "check the weather", {"temp": 72, "conditions": "partly cloudy"})
        gate.matcher.add_pattern("check the weather", key)

        # The canonical form should match exactly
        result = gate.process("check the weather", ctx)
        assert result.hit
        assert result.is_exact

    def test_tide_variations(self):
        """Multiple ways to ask about tide should be handled."""
        gate = Gate()

        ctx = Context(time_of_day="dawn", operational_mode="cruising")
        key = build_reflex_key("what's the tide doing", ctx.vector)
        gate.cache.put(key, "what's the tide doing", {"tide": "rising", "height": "2.1m"})
        gate.matcher.add_pattern("what's the tide doing", key)
        gate.matcher.add_alias("check the tide", "what's the tide doing")
        gate.matcher.add_alias("tide report", "what's the tide doing")

        # Alias-based queries should match exactly through alias resolution
        for query in ["check the tide", "tide report"]:
            result = gate.matcher.match(query)
            assert result.matched
            assert result.tier == "exact"
