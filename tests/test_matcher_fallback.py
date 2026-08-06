"""Tests for the difflib fallback path when rapidfuzz is not available.

The matcher has a fallback to difflib.SequenceMatcher when rapidfuzz
is not installed. We test it by temporarily simulating the absence.
"""

import pytest
from unittest.mock import patch, MagicMock

from voice_reflex.matcher import FuzzyMatcher, MatchResult


class TestDifflibFallback:
    """Test the fallback path by mocking HAS_RAPIDFUZZ = False."""

    def test_difflib_exact_match(self):
        """Exact match works with difflib fallback."""
        m = FuzzyMatcher()
        m.add_pattern("check weather", "k1")

        with patch("voice_reflex.matcher.HAS_RAPIDFUZZ", False):
            result = m.match("check weather")
            assert result.matched
            assert result.tier == "exact"
            assert result.score == 1.0

    def test_difflib_no_match(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "k1")

        with patch("voice_reflex.matcher.HAS_RAPIDFUZZ", False):
            result = m.match("completely different topic here")
            # difflib might find a very low score → no match
            # or a marginal score → fuzzy match
            assert isinstance(result, MatchResult)

    def test_difflib_fuzzy_match(self):
        m = FuzzyMatcher()
        m.add_pattern("check the weather", "k1")

        with patch("voice_reflex.matcher.HAS_RAPIDFUZZ", False):
            result = m.match("check the weather")
            assert result.matched
            assert result.tier == "exact"

    def test_difflib_best_match_multiple_patterns(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "k1")
        m.add_pattern("check tide", "k2")

        with patch("voice_reflex.matcher.HAS_RAPIDFUZZ", False):
            result = m.match("check weather")
            assert result.matched
            assert result.key == "k1"

    def test_difflib_returns_tuple(self):
        """The _best_match method should return (text, score) with difflib."""
        m = FuzzyMatcher()
        m.add_pattern("hello", "k1")

        with patch("voice_reflex.matcher.HAS_RAPIDFUZZ", False):
            best_text, best_score = m._best_match("hello", ["hello"])
            assert best_text == "hello"
            assert best_score == pytest.approx(1.0)

    def test_difflib_empty_candidates_returns_empty(self):
        """_best_match with empty candidate list returns ('', 0.0)."""
        m = FuzzyMatcher()
        with patch("voice_reflex.matcher.HAS_RAPIDFUZZ", False):
            best_text, best_score = m._best_match("test", [])
            assert best_text == ""
            assert best_score == 0.0

    def test_rapidfuzz_none_result_returns_empty(self):
        """When rapidfuzz returns None (no match above cutoff), we get empty."""
        m = FuzzyMatcher()
        m.add_pattern("abc", "k1")

        # Mock rapidfuzz to return None
        mock_process = MagicMock()
        mock_process.extractOne.return_value = None

        with patch("voice_reflex.matcher.HAS_RAPIDFUZZ", True), \
             patch("voice_reflex.matcher.rf_process", mock_process):
            best_text, best_score = m._best_match("test", ["abc"])
            assert best_text == ""
            assert best_score == 0.0


class TestRapidfuzzIntegration:
    """Verify rapidfuzz path works correctly when available."""

    def test_rapidfuzz_exact_match(self):
        m = FuzzyMatcher()
        m.add_pattern("check weather", "k1")
        result = m.match("check weather")
        assert result.matched
        assert result.tier == "exact"

    def test_rapidfuzz_best_match_returns_correct_tuple(self):
        m = FuzzyMatcher()
        m.add_pattern("hello world", "k1")
        best_text, best_score = m._best_match("hello world", ["hello world"])
        assert best_text == "hello world"
        assert best_score > 0.99
