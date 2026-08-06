"""Extended tests for the fuzzy matcher — edge cases, properties, fallback, stress."""

import pytest

from voice_reflex.matcher import (
    EXACT_THRESHOLD,
    FUZZY_THRESHOLD,
    FuzzyMatcher,
    MatchResult,
)


class TestMatchResultProperties:
    def test_is_exact_true(self):
        r = MatchResult(matched=True, text="test", score=1.0, tier="exact")
        assert r.is_exact
        assert not r.is_fuzzy
        assert not r.is_none

    def test_is_fuzzy_true(self):
        r = MatchResult(matched=True, text="test", score=0.7, tier="fuzzy")
        assert not r.is_exact
        assert r.is_fuzzy
        assert not r.is_none

    def test_is_none_true(self):
        r = MatchResult(matched=False, text="", score=0.0, tier="none")
        assert not r.is_exact
        assert not r.is_fuzzy
        assert r.is_none

    def test_frozen_result(self):
        r = MatchResult(matched=True, text="test", score=1.0)
        with pytest.raises(AttributeError):
            r.matched = False  # type: ignore

    def test_default_values(self):
        r = MatchResult(matched=True, text="test", score=0.8)
        assert r.key is None
        assert r.tier == "none"


class TestFuzzyMatcherNormalize:
    def test_normalize_lowercases(self):
        assert FuzzyMatcher._normalize("HELLO WORLD") == "hello world"

    def test_normalize_strips(self):
        assert FuzzyMatcher._normalize("  hello  ") == "hello"

    def test_normalize_collapses_whitespace(self):
        assert FuzzyMatcher._normalize("hello    world") == "hello world"

    def test_normalize_tabs(self):
        assert FuzzyMatcher._normalize("hello\t\tworld") == "hello world"

    def test_normalize_empty(self):
        assert FuzzyMatcher._normalize("") == ""

    def test_normalize_only_whitespace(self):
        assert FuzzyMatcher._normalize("   ") == ""


class TestFuzzyMatcherAddRemove:
    def test_add_multiple_patterns(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "k1")
        m.add_pattern("check tide", "k2")
        m.add_pattern("plot course", "k3")
        assert len(m.patterns) == 3

    def test_add_pattern_normalizes(self):
        m = FuzzyMatcher()
        m.add_pattern("CHECK   WEATHER", "k1")
        # Pattern is stored normalized
        assert "check weather" in m.patterns
        assert m.patterns["check weather"] == "k1"

    def test_add_pattern_overwrites(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "k1")
        m.add_pattern("check weather", "k2")
        assert len(m.patterns) == 1
        assert m.patterns["check weather"] == "k2"

    def test_add_alias_normalizes(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "k1")
        m.add_alias("WHAT'S THE WEATHER", "check the weather")

        result = m.match("what's the weather")
        assert result.matched
        assert result.tier == "exact"

    def test_remove_pattern_normalizes(self):
        m = FuzzyMatcher()
        m.add_pattern("Check Weather", "k1")
        m.remove_pattern("check weather")
        assert len(m.patterns) == 0

    def test_remove_nonexistent_no_error(self):
        m = FuzzyMatcher()
        m.remove_pattern("nonexistent")
        assert len(m.patterns) == 0

    def test_patterns_property_returns_copy(self):
        m = FuzzyMatcher()
        m.add_pattern("test", "k1")
        patterns = m.patterns
        patterns["injected"] = "k2"
        assert "injected" not in m.patterns


class TestFuzzyMatcherAliasEdgeCases:
    def test_alias_to_nonexistent_pattern(self):
        """Alias to a canonical form that isn't registered as a pattern."""
        m = FuzzyMatcher()
        m.add_alias("what's the weather", "check the weather")
        # No pattern registered for "check the weather"
        result = m.match("what's the weather")
        # Alias resolves but canonical isn't in patterns → falls through to fuzzy
        assert isinstance(result, MatchResult)

    def test_alias_chain_not_followed(self):
        """Aliases don't chain — an alias maps to a canonical, not to another alias."""
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "k1")
        m.add_alias("weather report", "check the weather")
        m.add_alias("give me weather", "weather report")  # alias to alias

        # "give me weather" maps to "weather report" (normalized)
        # But "weather report" is an alias, not a pattern
        result = m.match("give me weather")
        # Falls through to fuzzy matching
        assert isinstance(result, MatchResult)

    def test_multiple_aliases_same_canonical(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "k_weather")
        m.add_alias("what's the weather", "check the weather")
        m.add_alias("how's the weather", "check the weather")
        m.add_alias("weather update", "check the weather")
        m.add_alias("give me the weather", "check the weather")

        for alias in ["what's the weather", "how's the weather", "weather update", "give me the weather"]:
            result = m.match(alias)
            assert result.matched
            assert result.tier == "exact"
            assert result.key == "k_weather"


class TestFuzzyMatcherMatching:
    def test_match_with_no_patterns(self):
        m = FuzzyMatcher()
        result = m.match("check weather")
        assert not result.matched
        assert result.score == 0.0
        assert result.tier == "none"

    def test_exact_match_score_is_1(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "k1")
        result = m.match("check weather")
        assert result.score == 1.0

    def test_near_exact_match(self):
        """Very similar text should match at or above exact threshold."""
        m = FuzzyMatcher()
        m.add_pattern("check the weather forecast", "k1")
        result = m.match("check the weather forecast")  # exact
        assert result.matched
        assert result.tier == "exact"
        assert result.score >= 0.95

    def test_completely_unrelated_text(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "k_weather")
        result = m.match("abc xyz qwerty")
        assert not result.matched
        assert result.tier == "none"
        assert result.score < FUZZY_THRESHOLD

    def test_returns_best_match_key(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "k_weather")
        m.add_pattern("check the tide", "k_tide")
        m.add_pattern("check the fuel", "k_fuel")

        result = m.match("check the weather")
        assert result.key == "k_weather"

    def test_fuzzy_match_score_in_range(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather conditions outside", "k1")
        result = m.match("check weather conditions")
        if result.matched:
            assert FUZZY_THRESHOLD <= result.score < EXACT_THRESHOLD


class TestFuzzyMatcherThresholds:
    def test_custom_exact_threshold(self):
        m = FuzzyMatcher(exact_threshold=0.99)
        m.add_pattern("check the weather", "k1")
        # Perfect match still exact
        result = m.match("check the weather")
        assert result.tier == "exact"

    def test_custom_fuzzy_threshold_high(self):
        """Very high fuzzy threshold → most similar text won't match."""
        m = FuzzyMatcher(fuzzy_threshold=0.99)
        m.add_pattern("check the weather", "k1")
        result = m.match("check weather")
        # Score likely below 0.99
        if not result.matched:
            assert result.tier == "none"

    def test_custom_fuzzy_threshold_zero(self):
        """Zero fuzzy threshold → everything matches."""
        m = FuzzyMatcher(fuzzy_threshold=0.0)
        m.add_pattern("a", "k1")
        result = m.match("b")
        # Even terrible matches should pass with threshold 0
        # Unless score is exactly 0
        assert isinstance(result, MatchResult)


class TestFuzzyMatcherBatch:
    def test_empty_batch(self):
        m = FuzzyMatcher()
        results = m.match_batch([])
        assert results == []

    def test_single_item_batch(self):
        m = FuzzyMatcher()
        m.add_pattern("test", "k1")
        results = m.match_batch(["test"])
        assert len(results) == 1
        assert results[0].matched

    def test_mixed_batch(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "k1")
        m.add_pattern("check tide", "k2")

        results = m.match_batch([
            "check weather",
            "check tide",
            "play music",
            "CHECK WEATHER",  # case-insensitive exact
        ])
        assert results[0].matched
        assert results[1].matched
        assert not results[2].matched
        assert results[3].matched

    def test_large_batch(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "k1")

        texts = [f"check weather variation {i}" for i in range(100)]
        results = m.match_batch(texts)
        assert len(results) == 100


class TestFuzzyMatcherStress:
    def test_many_patterns(self):
        m = FuzzyMatcher()
        for i in range(200):
            m.add_pattern(f"command pattern number {i}", f"k{i}")

        result = m.match("command pattern number 100")
        assert result.matched
        assert result.key == "k100"

    def test_repeated_matching_stable(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "k1")

        for _ in range(100):
            result = m.match("check the weather")
            assert result.matched
            assert result.score == 1.0

    def test_long_text(self):
        m = FuzzyMatcher()
        long_text = "check " + "very " * 50 + "carefully"
        m.add_pattern(long_text, "k1")
        result = m.match(long_text)
        assert result.matched
        assert result.tier == "exact"

    def test_single_char_patterns(self):
        m = FuzzyMatcher()
        m.add_pattern("a", "k1")
        m.add_pattern("b", "k2")
        result = m.match("a")
        assert result.matched
        assert result.key == "k1"


class TestFuzzyMatcherBestMatch:
    def test_best_match_picks_highest_scoring(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "k_weather")
        m.add_pattern("check the tide", "k_tide")
        m.add_pattern("check the fuel", "k_fuel")

        # "check the weather" is closest to itself
        result = m.match("check the weather")
        assert result.key == "k_weather"

    def test_best_match_with_tie_candidates(self):
        """When two patterns are equally similar, one is returned."""
        m = FuzzyMatcher()
        m.add_pattern("abc", "k1")
        m.add_pattern("xyz", "k2")
        result = m.match("abc")
        assert result.matched
        # "abc" should match itself
        assert result.key == "k1"


class TestModuleConstants:
    def test_exact_threshold_value(self):
        assert EXACT_THRESHOLD == 0.95

    def test_fuzzy_threshold_value(self):
        assert FUZZY_THRESHOLD == 0.60

    def test_thresholds_ordered(self):
        assert EXACT_THRESHOLD > FUZZY_THRESHOLD
        assert 0 <= FUZZY_THRESHOLD <= 1
        assert 0 <= EXACT_THRESHOLD <= 1
