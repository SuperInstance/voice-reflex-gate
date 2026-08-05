"""
Context vector builder — assembles real-time context into a compact vector for reflex keying.

The context vector is what makes "check depth" at the dock different from "check depth" underway.
Same words, different context → different reflex lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from hashlib import sha256
from typing import Sequence


class TimeOfDay(str, Enum):
    """Coarse time-of-day bucket. Dawn, midday, dusk, night."""
    DAWN = "dawn"
    MIDDAY = "midday"
    DUSK = "dusk"
    NIGHT = "night"

    @classmethod
    def from_datetime(cls, dt: datetime) -> "TimeOfDay":
        """Classify a datetime into a time-of-day bucket based on hour."""
        hour = dt.hour
        if 5 <= hour < 9:
            return cls.DAWN
        if 9 <= hour < 17:
            return cls.MIDDAY
        if 17 <= hour < 21:
            return cls.DUSK
        return cls.NIGHT


class OperationalMode(str, Enum):
    """What the vessel/system is doing right now."""
    DOCKED = "docked"
    CRUISING = "cruising"
    FISHING = "fishing"
    ANCHORED = "anchored"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class LocationState(str, Enum):
    """Where the vessel/system is."""
    UNDERWAY = "underway"
    AT_ANCHOR = "at_anchor"
    AT_DOCK = "at_dock"
    IN_HARBOR = "in_harbor"
    OPEN_WATER = "open_water"


@dataclass(frozen=True)
class ContextVector:
    """
    Compact context representation used as part of the reflex key.

    Fields are intentionally coarse-grained — fine enough to discriminate
    meaningfully different contexts, coarse enough that similar situations
    produce the same key.
    """
    time_of_day: TimeOfDay = TimeOfDay.MIDDAY
    operational_mode: OperationalMode = OperationalMode.DOCKED
    location_state: LocationState = LocationState.AT_DOCK
    recent_command_hash: str = ""  # hash of the last N commands, empty = no history
    season: str = ""  # "spring" | "summer" | "autumn" | "winter", empty = unspecified

    def to_dict(self) -> dict[str, str]:
        """Serialize to a flat dict — used for logging, debugging, persistence."""
        return {
            "time_of_day": self.time_of_day.value,
            "operational_mode": self.operational_mode.value,
            "location_state": self.location_state.value,
            "recent_command_hash": self.recent_command_hash,
            "season": self.season,
        }

    def key_component(self) -> str:
        """
        Produce a stable string that can be combined with STT text to form the reflex key.

        The format is deliberately pipe-separated for readability during debugging.
        """
        return "|".join([
            self.time_of_day.value,
            self.operational_mode.value,
            self.location_state.value,
            self.recent_command_hash[:8] if self.recent_command_hash else "none",
            self.season or "unspecified",
        ])


@dataclass
class Context:
    """
    High-level context wrapper — builds a ContextVector from raw inputs.

    This is the user-facing API. ContextVector is the internal representation.
    """

    # Either provide a ContextVector directly, or provide the components.
    _vector: ContextVector | None = field(default=None, repr=False)

    # Component fields — used if _vector is None
    time_of_day: TimeOfDay | str = TimeOfDay.MIDDAY
    operational_mode: OperationalMode | str = OperationalMode.DOCKED
    location_state: LocationState | str = LocationState.AT_DOCK
    recent_commands: Sequence[str] = field(default_factory=list)
    season: str = ""
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if self._vector is not None:
            return

        # Coerce string values to enums
        tod = self._coerce_enum(self.time_of_day, TimeOfDay)
        om = self._coerce_enum(self.operational_mode, OperationalMode)
        ls = self._coerce_enum(self.location_state, LocationState)

        # Build recent command hash
        cmd_hash = ""
        if self.recent_commands:
            joined = "\n".join(self.recent_commands[-10:])  # last 10 commands
            cmd_hash = sha256(joined.encode("utf-8")).hexdigest()

        # Auto-detect time of day from timestamp if provided
        if self.timestamp is not None and self.time_of_day == TimeOfDay.MIDDAY:
            # Only auto-detect if user didn't explicitly set it
            tod = TimeOfDay.from_datetime(self.timestamp)

        # Auto-detect season from timestamp if not provided
        season = self.season
        if not season and self.timestamp is not None:
            season = self._season_from_datetime(self.timestamp)

        self._vector = ContextVector(
            time_of_day=tod,
            operational_mode=om,
            location_state=ls,
            recent_command_hash=cmd_hash,
            season=season,
        )

    @property
    def vector(self) -> ContextVector:
        assert self._vector is not None
        return self._vector

    @staticmethod
    def _coerce_enum(value: str | Enum, enum_cls: type[Enum]) -> Enum:
        if isinstance(value, enum_cls):
            return value
        # Try to match by value
        for member in enum_cls:
            if member.value == value:
                return member
        # Try to match by name (case-insensitive)
        for member in enum_cls:
            if member.name.lower() == str(value).lower():
                return member
        raise ValueError(f"Cannot coerce {value!r} to {enum_cls.__name__}")

    @staticmethod
    def _season_from_datetime(dt: datetime) -> str:
        """Rough Northern Hemisphere season classification by month."""
        month = dt.month
        if 3 <= month <= 5:
            return "spring"
        if 6 <= month <= 8:
            return "summer"
        if 9 <= month <= 11:
            return "autumn"
        return "winter"

    @classmethod
    def from_datetime(
        cls,
        dt: datetime,
        *,
        operational_mode: OperationalMode | str = OperationalMode.DOCKED,
        location_state: LocationState | str = LocationState.AT_DOCK,
        recent_commands: Sequence[str] | None = None,
        season: str = "",
    ) -> "Context":
        """Convenience: build a Context auto-classified from a datetime."""
        return cls(
            time_of_day=TimeOfDay.from_datetime(dt),
            operational_mode=operational_mode,
            location_state=location_state,
            recent_commands=recent_commands or [],
            season=season,
            timestamp=dt,
        )


def build_reflex_key(stt_text: str, ctx_vector: ContextVector) -> str:
    """
    Combine STT text + context vector into a stable hash key.

    The key is deterministic: same text + same context = same key.
    Used for exact-match lookups in the reflex cache.
    """
    normalized = " ".join(stt_text.strip().lower().split())
    combined = f"{normalized}::{ctx_vector.key_component()}"
    return sha256(combined.encode("utf-8")).hexdigest()
