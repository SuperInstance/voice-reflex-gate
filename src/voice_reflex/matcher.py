"""
Fuzzy matching engine — matches STT text against known request patterns.

Uses rapidfuzz for fast fuzzy string matching, with a fallback to difflib
if rapidfuzz is not available.

The matcher handles natural variation:
    "check weather" vs "what's the weather" vs "how's the weather"
    → all match the same pattern family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import difflib

try:
    from rapidfuzz import fuzz as rf_fuzz
    from rapidfuzz import process as rf_process

    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

    rf_fuzz = None  # type: ignore
    rf_process = None  # type: ignore


@dataclass(frozen=True)
class MatchResult:
    """
    Result of a fuzzy match attempt.

    Attributes:
        matched: Whether a match was found above threshold.
        text: The matched pattern text (or best candidate if no match).
        score: Confidence score in [0.0, 1.0].
        key: The reflex key associated with the matched pattern (if any).
        tier: Match tier: "exact" | "fuzzy" | "none".
    """

    matched: bool
    text: str
    score: float
    key: str | None = None
    tier: str = "none"

    @property
    def is_exact(self) -> bool:
        return self.tier == "exact"

    @property
    def is_fuzzy(self) -> bool:
        return self.tier == "fuzzy"

    @property
    def is_none(self) -> bool:
        return self.tier == "none"


# Score thresholds — calibrated for natural speech variation.
EXACT_THRESHOLD = 0.95  # Near-identical (normalized text matches exactly)
FUZZY_THRESHOLD = 0.60  # Different phrasing, same intent
# Below FUZZY_THRESHOLD = no match → cascade to model


class FuzzyMatcher:
    """
    Fuzzy matching engine for STT text against known request patterns.

    Patterns are registered with associated reflex keys. The matcher
    returns the best match above threshold, or a miss.
    """

    def __init__(
        self,
        *,
        exact_threshold: float = EXACT_THRESHOLD,
        fuzzy_threshold: float = FUZZY_THRESHOLD,
    ) -> None:
        self.exact_threshold = exact_threshold
        self.fuzzy_threshold = fuzzy_threshold
        # patterns[text] = reflex_key — the canonical forms
        self._patterns: dict[str, str] = {}
        # aliases[alias] = canonical_text — for grouping variations
        self._aliases: dict[str, str] = {}

    def add_pattern(self, text: str, key: str) -> None:
        """Register a canonical request pattern with its reflex key."""
        normalized = self._normalize(text)
        self._patterns[normalized] = key

    def add_alias(self, alias: str, canonical: str) -> None:
        """Register an alias that maps to a canonical pattern."""
        self._aliases[self._normalize(alias)] = self._normalize(canonical)

    def remove_pattern(self, text: str) -> None:
        """Remove a pattern from the matcher."""
        normalized = self._normalize(text)
        self._patterns.pop(normalized, None)

    @property
    def patterns(self) -> dict[str, str]:
        """Return a copy of the current patterns."""
        return dict(self._patterns)

    def match(self, text: str) -> MatchResult:
        """
        Match STT text against known patterns.

        Three tiers:
        1. Exact: normalized text matches a pattern exactly (score ≥ 0.95)
        2. Fuzzy: text is similar to a pattern (0.60 ≤ score < 0.95)
        3. None: no match above threshold → cascade to model
        """
        normalized = self._normalize(text)

        if not normalized:
            return MatchResult(matched=False, text=text, score=0.0, tier="none")

        # Check for alias match first (exact, but through alias mapping)
        if normalized in self._aliases:
            canonical = self._aliases[normalized]
            if canonical in self._patterns:
                return MatchResult(
                    matched=True,
                    text=canonical,
                    score=1.0,
                    key=self._patterns[canonical],
                    tier="exact",
                )

        # Exact match check (normalized text == pattern)
        if normalized in self._patterns:
            return MatchResult(
                matched=True,
                text=normalized,
                score=1.0,
                key=self._patterns[normalized],
                tier="exact",
            )

        if not self._patterns:
            return MatchResult(matched=False, text=text, score=0.0, tier="none")

        # Fuzzy match against all patterns
        pattern_texts = list(self._patterns.keys())
        best_text, best_score = self._best_match(normalized, pattern_texts)

        if best_score >= self.exact_threshold:
            # Effectively exact
            return MatchResult(
                matched=True,
                text=best_text,
                score=best_score,
                key=self._patterns[best_text],
                tier="exact",
            )
        elif best_score >= self.fuzzy_threshold:
            return MatchResult(
                matched=True,
                text=best_text,
                score=best_score,
                key=self._patterns[best_text],
                tier="fuzzy",
            )
        else:
            return MatchResult(
                matched=False,
                text=best_text,
                score=best_score,
                tier="none",
            )

    def match_batch(self, texts: Sequence[str]) -> list[MatchResult]:
        """Match multiple texts in one call."""
        return [self.match(t) for t in texts]

    def _best_match(self, text: str, candidates: list[str]) -> tuple[str, float]:
        """Find the best matching candidate for text. Returns (candidate, score)."""
        if HAS_RAPIDFUZZ:
            # rapidfuzz returns scores in 0-100, we normalize to 0-1
            result = rf_process.extractOne(
                text,
                candidates,
                scorer=rf_fuzz.WRatio,
                score_cutoff=0,  # we handle threshold ourselves
            )
            if result is None:
                return ("", 0.0)
            match_text, score, _ = result  # type: ignore[misc]
            return (match_text, score / 100.0)
        else:
            # Fallback to difflib
            best_text = ""
            best_score = 0.0
            for candidate in candidates:
                score = difflib.SequenceMatcher(None, text, candidate).ratio()
                if score > best_score:
                    best_score = score
                    best_text = candidate
            return (best_text, best_score)

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize text for matching: lowercase, strip, collapse whitespace.

        This is intentionally simple — the fuzzy matcher handles the rest.
        We do NOT remove punctuation because it can carry meaning
        ("plot course to: fishing grounds" vs "plot course to fishing grounds"
        are effectively the same, and the fuzzy matcher handles this).
        """
        return " ".join(text.strip().lower().split())
