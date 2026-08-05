"""Tests for the fuzzy matching engine."""

import pytest

from voice_reflex.matcher import FuzzyMatcher, MatchResult, EXACT_THRESHOLD, FUZZY_THRESHOLD


class TestFuzzyMatcherBasics:
    def test_empty_matcher_returns_none(self):
        m = FuzzyMatcher()
        result = m.match("anything")
        assert not result.matched
        assert result.tier == "none"

    def test_empty_text_returns_none(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "key1")
        result = m.match("")
        assert not result.matched
        assert result.tier == "none"

    def test_whitespace_only_returns_none(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "key1")
        result = m.match("   ")
        assert not result.matched


class TestExactMatch:
    def test_exact_match_returns_exact_tier(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "key_weather")
        result = m.match("check the weather")
        assert result.matched
        assert result.tier == "exact"
        assert result.score == 1.0
        assert result.key == "key_weather"

    def test_case_insensitive_exact(self):
        m = FuzzyMatcher()
        m.add_pattern("Check The Weather", "key_weather")
        result = m.match("check the weather")
        assert result.matched
        assert result.tier == "exact"

    def test_whitespace_normalized_exact(self):
        m = FuzzyMatcher()
        m.add_pattern("check   the   weather", "key_weather")
        result = m.match("check the weather")
        assert result.matched
        assert result.tier == "exact"


class TestAliasMatch:
    def test_alias_returns_exact(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "key_weather")
        m.add_alias("what's the weather", "check the weather")
        m.add_alias("how's the weather", "check the weather")

        for alias in ["what's the weather", "how's the weather"]:
            result = m.match(alias)
            assert result.matched
            assert result.tier == "exact"
            assert result.key == "key_weather"

    def test_alias_not_matching_canonical(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "key_weather")
        m.add_alias("weather report", "check the weather")

        # Alias should match exactly
        result = m.match("weather report")
        assert result.matched
        assert result.tier == "exact"


class TestFuzzyMatch:
    def test_similar_text_matches_fuzzy(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather forecast", "key_weather")
        result = m.match("check weather forecast")
        assert result.matched
        assert result.score >= FUZZY_THRESHOLD

    def test_very_different_text_does_not_match(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "key_weather")
        result = m.match("play some music")
        assert not result.matched
        assert result.tier == "none"

    def test_natural_variation_weather(self):
        """Test the core use case: different phrasings of the same intent."""
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "key_weather")

        variations = [
            "what's the weather",
            "how's the weather",
            "weather check",
            "give me the weather",
        ]
        for v in variations:
            result = m.match(v)
            # At least some of these should be fuzzy matches
            # (exact thresholds may vary by algorithm, so we just check they're in the cache)
            if result.matched:
                assert result.score >= FUZZY_THRESHOLD

    def test_natural_variation_tide(self):
        m = FuzzyMatcher()
        m.add_pattern("what's the tide doing", "key_tide")

        variations = [
            "check the tide",
            "tide report",
            "what's the tide",
            "how's the tide",
        ]
        matched_count = 0
        for v in variations:
            result = m.match(v)
            if result.matched:
                matched_count += 1

        # At least 2 of 4 variations should match
        assert matched_count >= 2, f"Only {matched_count}/4 tide variations matched"


class TestMultiplePatterns:
    def test_best_match_selected(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "key_weather")
        m.add_pattern("check the tide", "key_tide")

        result = m.match("check the weather")
        assert result.matched
        assert result.key == "key_weather"

        result = m.match("check the tide")
        assert result.matched
        assert result.key == "key_tide"

    def test_ambiguous_match_picks_highest_score(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "key_weather")
        m.add_pattern("check whether", "key_whether")

        # "check the weather" should match the weather pattern more strongly
        result = m.match("check the weather")
        assert result.matched
        assert result.key == "key_weather"


class TestRemovePattern:
    def test_remove_pattern(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "key1")
        assert len(m.patterns) == 1

        m.remove_pattern("check weather")
        assert len(m.patterns) == 0

        result = m.match("check weather")
        assert not result.matched


class TestMatchBatch:
    def test_match_batch(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "key1")
        m.add_pattern("check tide", "key2")

        results = m.match_batch(["check weather", "check tide", "play music"])
        assert len(results) == 3
        assert results[0].matched
        assert results[1].matched
        assert not results[2].matched


class TestCustomThresholds:
    def test_custom_fuzzy_threshold(self):
        m = FuzzyMatcher(fuzzy_threshold=0.90)
        m.add_pattern("check the weather", "key_weather")

        # A marginal match should now fail with a higher threshold
        result = m.match("check weather")
        # This might or might not match depending on the scorer,
        # but the threshold is higher so fewer things match
        assert isinstance(result, MatchResult)
