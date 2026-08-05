"""
The cascade gate — the entry point for voice reflex routing.

STT text + context → Gate → either return cached response or cascade to model.

Three tiers:
1. Exact match (instant, score ≥ 0.95) — return cached response directly
2. Fuzzy match (confidence-weighted, 0.60 ≤ score < 0.95) — return cached response with confidence
3. No match (score < 0.60) — cascade to model

The gate also integrates with the learner — when a model response is recorded,
the learner checks if it should compile a new reflex.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from voice_reflex.cache import CacheMiss, ReflexCache, ReflexEntry
from voice_reflex.context import Context, build_reflex_key
from voice_reflex.learner import Learner
from voice_reflex.matcher import FuzzyMatcher, MatchResult


@dataclass
class GateResult:
    """
    Result of processing a voice command through the gate.

    If `hit` is True, the response was served from the reflex cache.
    If `hit` is False, the request should cascade to the model.
    """

    hit: bool
    response: dict[str, Any] | None = None
    confidence: float = 0.0
    tier: str = "none"  # "exact" | "fuzzy" | "none"
    matched_text: str = ""
    reflex_key: str | None = None
    entry: ReflexEntry | None = None
    cascade_reason: str = ""  # why the gate cascaded (if it did)

    @property
    def is_exact(self) -> bool:
        return self.tier == "exact"

    @property
    def is_fuzzy(self) -> bool:
        return self.tier == "fuzzy"

    @property
    def should_cascade(self) -> bool:
        return not self.hit


class Gate:
    """
    The voice reflex gate — sits between STT and the model cascade.

    Usage:
        gate = Gate(cache=cache)
        result = gate.process("check the weather", context)

    The gate uses a FuzzyMatcher for pattern matching and a ReflexCache
    for response storage. The Learner observes model-handled requests
    and compiles new reflexes when patterns emerge.
    """

    def __init__(
        self,
        cache: ReflexCache | None = None,
        matcher: FuzzyMatcher | None = None,
        learner: Learner | None = None,
        *,
        min_fuzzy_confidence: float = 0.60,
    ) -> None:
        self.cache = cache if cache is not None else ReflexCache()
        self.matcher = matcher if matcher is not None else FuzzyMatcher()
        self.learner = learner if learner is not None else Learner(cache=self.cache, matcher=self.matcher)
        self.min_fuzzy_confidence = min_fuzzy_confidence

    def process(self, stt_text: str, ctx: Context) -> GateResult:
        """
        Process a voice command.

        This is the main entry point. Returns a GateResult indicating
        whether the response was served from cache or should cascade.
        """
        ctx_vector = ctx.vector

        # --- TIER 1: Exact reflex key lookup ---
        # Build the exact reflex key from STT text + context
        reflex_key = build_reflex_key(stt_text, ctx_vector)

        try:
            entry = self.cache.get(reflex_key)
            # Fresh exact match — instant response
            return GateResult(
                hit=True,
                response=entry.response,
                confidence=entry.confidence,
                tier="exact",
                matched_text=entry.text,
                reflex_key=reflex_key,
                entry=entry,
            )
        except CacheMiss:
            pass  # Fall through to fuzzy matching

        # --- TIER 2: Fuzzy match against known patterns ---
        match = self.matcher.match(stt_text)

        if match.matched and match.score >= self.min_fuzzy_confidence:
            # We have a fuzzy match — try to get the cached response
            # for the matched pattern's key in THIS context
            matched_key = match.key
            if matched_key:
                # The matched key might be context-specific.
                # Try building a context-specific key for the matched text.
                context_reflex_key = build_reflex_key(match.text, ctx_vector)
                entry = self.cache.try_get(context_reflex_key)

                if entry is not None:
                    # Found the cached response for this matched pattern in this context
                    return GateResult(
                        hit=True,
                        response=entry.response,
                        confidence=min(entry.confidence, match.score),
                        tier="fuzzy" if match.tier == "fuzzy" else "exact",
                        matched_text=match.text,
                        reflex_key=context_reflex_key,
                        entry=entry,
                    )
                else:
                    # The pattern is known, but no cached response exists for
                    # this exact context. Try the matched key directly.
                    entry = self.cache.try_get(matched_key)
                    if entry is not None:
                        return GateResult(
                            hit=True,
                            response=entry.response,
                            confidence=min(entry.confidence, match.score),
                            tier="fuzzy",
                            matched_text=match.text,
                            reflex_key=matched_key,
                            entry=entry,
                        )

        # --- TIER 3: No match — cascade to model ---
        return GateResult(
            hit=False,
            confidence=match.score if match else 0.0,
            tier="none",
            matched_text=match.text if match else "",
            cascade_reason="no reflex match above threshold",
        )

    def learn(
        self,
        stt_text: str,
        ctx: Context,
        response: dict[str, Any],
        *,
        category: str = "generic",
    ) -> None:
        """
        Record a model response for learning.

        The learner observes this response and checks if the STT+context
        has been seen before. After 3 similar requests with consistent
        responses, it auto-compiles a reflex.
        """
        self.learner.observe(stt_text, ctx, response, category=category)

    def check_new_reflexes(self) -> list[ReflexEntry]:
        """
        Check if the learner has compiled any new reflexes.

        Returns the list of newly compiled reflex entries.
        """
        return self.learner.check_compilations()

    def register_pattern(
        self,
        text: str,
        aliases: list[str] | None = None,
        category: str = "generic",
    ) -> None:
        """
        Manually register a request pattern with the matcher.

        This bypasses the learning loop — useful for seeding initial patterns.
        """
        # Create a key for this pattern
        from voice_reflex.context import ContextVector

        # Use a null context for the base key — the gate will combine with actual context at lookup
        base_key = build_reflex_key(text, ContextVector())
        self.matcher.add_pattern(text, base_key)

        if aliases:
            for alias in aliases:
                self.matcher.add_alias(alias, text)

    def stats(self) -> dict[str, Any]:
        """Return combined stats from cache and matcher."""
        cache_stats = self.cache.stats()
        return {
            **cache_stats,
            "pattern_count": len(self.matcher.patterns),
            "pending_observations": len(self.learner._observations),
        }
