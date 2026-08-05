"""Voice Reflex Gate — STT output as hash key for deterministic response routing."""

from voice_reflex.context import Context, ContextVector
from voice_reflex.cache import ReflexCache, ReflexEntry, CacheMiss
from voice_reflex.matcher import FuzzyMatcher, MatchResult
from voice_reflex.gate import Gate, GateResult
from voice_reflex.learner import Learner, Observation

__version__ = "0.1.0"

__all__ = [
    "Context",
    "ContextVector",
    "ReflexCache",
    "ReflexEntry",
    "CacheMiss",
    "FuzzyMatcher",
    "MatchResult",
    "Gate",
    "GateResult",
    "Learner",
    "Observation",
]
