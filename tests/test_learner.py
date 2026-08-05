"""Tests for the learner — observation, compilation, and self-training."""

from datetime import datetime, timezone

import pytest

from voice_reflex.cache import ReflexCache
from voice_reflex.context import Context, build_reflex_key
from voice_reflex.learner import Learner, Observation, COMPILATION_THRESHOLD
from voice_reflex.matcher import FuzzyMatcher
from voice_reflex.context import ContextVector


class TestObservation:
    def test_observation_auto_computes_key(self):
        ctx = Context(time_of_day="dawn", operational_mode="docked")
        obs = Observation(
            stt_text="check weather",
            context_vector=ctx.vector,
            response={"temp": 72},
        )
        assert obs.reflex_key
        expected = build_reflex_key("check weather", ctx.vector)
        assert obs.reflex_key == expected

    def test_observation_explicit_key(self):
        obs = Observation(
            stt_text="check weather",
            context_vector=ContextVector(),
            response={"temp": 72},
            reflex_key="custom_key",
        )
        assert obs.reflex_key == "custom_key"


class TestLearnerBasics:
    def test_observe_records(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context()
        learner.observe("check weather", ctx, {"temp": 72})

        assert learner.observation_count == 1

    def test_observe_does_not_immediately_compile(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context()
        learner.observe("check weather", ctx, {"temp": 72})

        assert len(cache) == 0


class TestCompilation:
    def test_three_consistent_observations_compile(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context(time_of_day="dawn", operational_mode="docked")
        response = {"temp": 72, "conditions": "clear"}

        for _ in range(COMPILATION_THRESHOLD):
            learner.observe("check the weather", ctx, response)

        # Should have compiled a reflex
        assert len(cache) >= 1

    def test_two_observations_do_not_compile(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher, compilation_threshold=3)

        ctx = Context()
        for _ in range(2):
            learner.observe("check weather", ctx, {"temp": 72})

        assert len(cache) == 0

    def test_inconsistent_responses_prevent_compilation(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context()
        learner.observe("check weather", ctx, {"temp": 72})
        learner.observe("check weather", ctx, {"temp": 85})
        learner.observe("check weather", ctx, {"temp": 60})

        # Different responses → not consistent → no compilation
        assert len(cache) == 0

    def test_compiled_reflex_registered_in_matcher(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context()
        for _ in range(3):
            learner.observe("check the weather", ctx, {"temp": 72})

        # The pattern should be registered with the matcher
        assert len(matcher.patterns) >= 1

    def test_compiled_reflex_has_correct_category(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context()
        for _ in range(3):
            learner.observe("check weather", ctx, {"temp": 72}, category="weather")

        for entry in cache:
            assert entry.category == "weather"

    def test_compiled_reflex_starts_below_full_confidence(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context()
        for _ in range(3):
            learner.observe("check weather", ctx, {"temp": 72})

        for entry in cache:
            assert entry.confidence == 0.9  # compiled reflexes start at 0.9, not 1.0


class TestCheckCompilations:
    def test_batch_check_compiles_pending(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context()
        # Add observations without triggering the per-observe check
        for _ in range(5):
            learner._observations.append(
                Observation(
                    stt_text="check weather",
                    context_vector=ctx.vector,
                    response={"temp": 72},
                )
            )

        # Batch check should find and compile
        compiled = learner.check_compilations()
        assert len(compiled) >= 1

    def test_already_compiled_not_recompiled(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context()
        for _ in range(3):
            learner.observe("check weather", ctx, {"temp": 72})

        initial_count = len(cache)
        # Run check again — should not recompile
        learner.check_compilations()
        assert len(cache) == initial_count


class TestDifferentContexts:
    def test_same_text_different_context_separate_groups(self):
        """Same STT text but different contexts should be tracked separately."""
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx1 = Context(time_of_day="dawn", operational_mode="docked")
        ctx2 = Context(time_of_day="night", operational_mode="cruising")

        for _ in range(3):
            learner.observe("check depth", ctx1, {"depth": 5})
        for _ in range(3):
            learner.observe("check depth", ctx2, {"depth": 100})

        # Should compile TWO separate reflexes for different contexts
        assert len(cache) >= 2


class TestSimilarTextsGrouping:
    def test_similar_texts_grouped_together(self):
        """Texts that are similar enough should be grouped."""
        from voice_reflex.learner import _texts_similar

        assert _texts_similar("check the weather", "check the weather")
        assert _texts_similar("check the weather", "check weather")
        assert not _texts_similar("check the weather", "play some music")

    def test_responses_similar(self):
        from voice_reflex.learner import _responses_similar

        assert _responses_similar({"temp": 72}, {"temp": 72})
        assert not _responses_similar({"temp": 72}, {"temp": 85})
        assert not _responses_similar({"temp": 72}, {"conditions": "clear"})
        assert _responses_similar({"a": "1", "b": "2"}, {"a": "1", "b": "2"})


class TestPurgeOldObservations:
    def test_purge_removes_old(self):
        from datetime import timedelta

        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context()

        # Add an old observation
        old_obs = Observation(
            stt_text="old query",
            context_vector=ctx.vector,
            response={"data": "old"},
        )
        old_obs.timestamp = datetime.now(timezone.utc) - timedelta(days=60)
        learner._observations.append(old_obs)

        # Add a recent one
        learner.observe("recent query", ctx, {"data": "recent"})

        removed = learner.purge_old_observations(max_age_days=30)
        assert removed == 1
        assert learner.observation_count == 1


class TestResetObservations:
    def test_reset_clears_but_keeps_cache(self):
        cache = ReflexCache()
        matcher = FuzzyMatcher()
        learner = Learner(cache=cache, matcher=matcher)

        ctx = Context()
        for _ in range(3):
            learner.observe("check weather", ctx, {"temp": 72})

        assert learner.observation_count > 0
        assert len(cache) >= 1

        learner.reset_observations()

        assert learner.observation_count == 0
        # Cache should still have the compiled reflex
        assert len(cache) >= 1


class TestFullLearningCycle:
    """The complete learn → compile → serve cycle."""

    def test_fourth_request_served_from_cache(self):
        """After 3 model-handled requests, the 4th is a cache hit."""
        from voice_reflex.gate import Gate

        gate = Gate()
        ctx = Context(time_of_day="dawn", operational_mode="cruising")

        # First 3 requests: model handles them, learner observes
        for i in range(3):
            result = gate.process("what's our fuel level", ctx)
            assert not result.hit  # Cache miss — would cascade to model
            gate.learn("what's our fuel level", ctx, {"fuel_pct": 78, "range_nm": 120})

        # After 3 observations, reflex should be compiled
        assert len(gate.cache) >= 1

        # 4th request: should be served from cache!
        result = gate.process("what's our fuel level", ctx)
        assert result.hit
        assert result.response == {"fuel_pct": 78, "range_nm": 120}
