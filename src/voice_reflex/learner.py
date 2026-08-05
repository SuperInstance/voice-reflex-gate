"""
The learner — observes model responses and compiles new reflexes.

When the model handles a request (cache miss), the learner records:
- The STT text
- The context vector
- The model's response

After 3 similar requests (same STT pattern + similar context) with
consistent responses, it auto-compiles a reflex. The 4th request
is handled by the cache — no model needed.

This is the self-training pipeline: the system learns which commands
are predictable enough to cache, from natural usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from voice_reflex.cache import ReflexCache, ReflexEntry
from voice_reflex.context import Context, ContextVector, build_reflex_key
from voice_reflex.matcher import FuzzyMatcher

# How many similar observations before we compile a reflex?
COMPILATION_THRESHOLD = 3

# How similar do responses need to be to count as "consistent"?
# 1.0 = exact match required, 0.8 = very similar, etc.
RESPONSE_SIMILARITY_THRESHOLD = 0.80

# How similar do STT texts need to be to count as "same pattern"?
STT_SIMILARITY_THRESHOLD = 0.70


@dataclass
class Observation:
    """
    A single observation of a model-handled request.

    Recorded every time the gate cascades to the model.
    """

    stt_text: str
    context_vector: ContextVector
    response: dict[str, Any]
    category: str = "generic"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reflex_key: str = ""  # the reflex key that WOULD have been used

    def __post_init__(self) -> None:
        if not self.reflex_key:
            self.reflex_key = build_reflex_key(self.stt_text, self.context_vector)


def _responses_similar(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """
    Heuristic check: are two responses similar enough to be "consistent"?

    This is intentionally simple — we compare the sorted key sets and
    the string representation of values. For production, this could be
    replaced with a semantic similarity check.
    """
    # Quick check: same keys
    if set(a.keys()) != set(b.keys()):
        return False

    # Compare values as strings (handles most cases)
    for k in a:
        sa = str(a[k]).strip().lower()
        sb = str(b[k]).strip().lower()
        if sa != sb:
            # Allow partial match on longer strings
            if len(sa) > 20 and len(sb) > 20:
                from difflib import SequenceMatcher

                ratio = SequenceMatcher(None, sa, sb).ratio()
                if ratio < RESPONSE_SIMILARITY_THRESHOLD:
                    return False
            else:
                return False

    return True


def _texts_similar(a: str, b: str) -> bool:
    """Check if two STT texts are similar enough to be the same pattern."""
    a_norm = " ".join(a.strip().lower().split())
    b_norm = " ".join(b.strip().lower().split())
    if a_norm == b_norm:
        return True

    from difflib import SequenceMatcher

    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    return ratio >= STT_SIMILARITY_THRESHOLD


class Learner:
    """
    Observes model responses and compiles new reflexes.

    The learner maintains a list of observations grouped by similarity.
    When a group reaches COMPILATION_THRESHOLD with consistent responses,
    a reflex is compiled and added to the cache.
    """

    def __init__(
        self,
        cache: ReflexCache,
        matcher: FuzzyMatcher,
        *,
        compilation_threshold: int = COMPILATION_THRESHOLD,
    ) -> None:
        self.cache = cache
        self.matcher = matcher
        self.compilation_threshold = compilation_threshold
        self._observations: list[Observation] = []
        # Track which groups have already been compiled to avoid recompiling
        self._compiled_keys: set[str] = set()

    def observe(
        self,
        stt_text: str,
        ctx: Context,
        response: dict[str, Any],
        *,
        category: str = "generic",
    ) -> Observation:
        """
        Record a model-handled request.

        The learner will check if this observation, combined with previous
        ones, should trigger a reflex compilation.
        """
        obs = Observation(
            stt_text=stt_text,
            context_vector=ctx.vector,
            response=response,
            category=category,
        )
        self._observations.append(obs)

        # Check if we should compile
        self._check_for_compilation(obs)
        return obs

    def check_compilations(self) -> list[ReflexEntry]:
        """
        Check all pending observations for compilable patterns.

        This is a more thorough sweep than the per-observation check
        done in observe(). Useful for periodic batch processing.
        """
        compiled: list[ReflexEntry] = []

        # Group observations by similarity
        groups = self._group_similar_observations()

        for group in groups:
            if len(group) >= self.compilation_threshold and self._responses_consistent(group):
                entry = self._compile_group(group)
                if entry:
                    compiled.append(entry)

        return compiled

    def _check_for_compilation(self, obs: Observation) -> ReflexEntry | None:
        """
        Check if this observation, combined with similar previous ones,
        should trigger a reflex compilation.
        """
        # Already compiled for this key?
        if obs.reflex_key in self._compiled_keys:
            return None

        # Already in the cache?
        if obs.reflex_key in self.cache:
            self._compiled_keys.add(obs.reflex_key)
            return None

        # Find similar observations
        similar = [
            o for o in self._observations
            if _texts_similar(o.stt_text, obs.stt_text)
            and o.context_vector.key_component() == obs.context_vector.key_component()
        ]

        if len(similar) < self.compilation_threshold:
            return None

        # Check response consistency
        if not self._responses_consistent(similar):
            return None

        # Compile!
        entry = self._compile_group(similar)
        return entry

    def _responses_consistent(self, observations: list[Observation]) -> bool:
        """
        Check if all observations in a group have consistent responses.

        We compare each pair — they all need to be similar.
        """
        if len(observations) < 2:
            return True

        base = observations[0].response
        for obs in observations[1:]:
            if not _responses_similar(base, obs.response):
                return False
        return True

    def _group_similar_observations(self) -> list[list[Observation]]:
        """
        Group observations by STT text similarity + context similarity.

        Uses a simple greedy clustering approach.
        """
        groups: list[list[Observation]] = []
        assigned: set[int] = set()

        for i, obs in enumerate(self._observations):
            if i in assigned:
                continue

            group = [obs]
            assigned.add(i)

            for j in range(i + 1, len(self._observations)):
                if j in assigned:
                    continue
                other = self._observations[j]
                if (
                    _texts_similar(obs.stt_text, other.stt_text)
                    and obs.context_vector.key_component() == other.context_vector.key_component()
                ):
                    group.append(other)
                    assigned.add(j)

            groups.append(group)

        return groups

    def _compile_group(self, group: list[Observation]) -> ReflexEntry | None:
        """
        Compile a group of consistent observations into a reflex entry.

        Uses the most recent observation's response (most fresh).
        Also registers the pattern with the matcher.
        """
        if not group:
            return None

        # Sort by timestamp, use most recent
        group.sort(key=lambda o: o.timestamp)
        latest = group[-1]

        # Already compiled?
        if latest.reflex_key in self._compiled_keys:
            return None
        if latest.reflex_key in self.cache:
            self._compiled_keys.add(latest.reflex_key)
            return None

        # Compile the reflex
        entry = self.cache.put(
            key=latest.reflex_key,
            text=latest.stt_text,
            response=latest.response,
            category=latest.category,
            context_snapshot=latest.context_vector.to_dict(),
            confidence=0.9,  # compiled reflexes start slightly below 1.0
        )

        # Register pattern with matcher so future fuzzy matches can find it
        self.matcher.add_pattern(latest.stt_text, latest.reflex_key)

        # Mark as compiled
        self._compiled_keys.add(latest.reflex_key)

        return entry

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def reset_observations(self) -> None:
        """Clear all observations (but keep compiled reflexes)."""
        self._observations.clear()

    def purge_old_observations(self, max_age_days: int = 30) -> int:
        """
        Remove observations older than max_age_days.

        Returns the number of observations removed.
        Old observations that never triggered compilation are likely
        noise — the pattern wasn't consistent enough.
        """
        cutoff = datetime.now(timezone.utc).timestamp() - (max_age_days * 86400)
        before = len(self._observations)
        self._observations = [
            o for o in self._observations if o.timestamp.timestamp() >= cutoff
        ]
        return before - len(self._observations)
