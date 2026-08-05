"""Tests for the context vector builder and reflex key construction."""

from datetime import datetime

import pytest

from voice_reflex.context import (
    Context,
    ContextVector,
    LocationState,
    OperationalMode,
    TimeOfDay,
    build_reflex_key,
)


class TestTimeOfDay:
    def test_from_datetime_dawn(self):
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 6, 0)) == TimeOfDay.DAWN

    def test_from_datetime_midday(self):
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 12, 0)) == TimeOfDay.MIDDAY

    def test_from_datetime_dusk(self):
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 18, 0)) == TimeOfDay.DUSK

    def test_from_datetime_night(self):
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 22, 0)) == TimeOfDay.NIGHT

    def test_from_datetime_boundary_dawn(self):
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 5, 0)) == TimeOfDay.DAWN
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 4, 59)) == TimeOfDay.NIGHT

    def test_from_datetime_boundary_dusk(self):
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 21, 0)) == TimeOfDay.NIGHT
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 17, 0)) == TimeOfDay.DUSK


class TestContextVector:
    def test_default_values(self):
        v = ContextVector()
        assert v.time_of_day == TimeOfDay.MIDDAY
        assert v.operational_mode == OperationalMode.DOCKED
        assert v.location_state == LocationState.AT_DOCK
        assert v.recent_command_hash == ""
        assert v.season == ""

    def test_key_component_is_deterministic(self):
        v1 = ContextVector(time_of_day=TimeOfDay.DAWN, operational_mode=OperationalMode.CRUISING)
        v2 = ContextVector(time_of_day=TimeOfDay.DAWN, operational_mode=OperationalMode.CRUISING)
        assert v1.key_component() == v2.key_component()

    def test_key_component_differs_for_different_contexts(self):
        v1 = ContextVector(time_of_day=TimeOfDay.DAWN)
        v2 = ContextVector(time_of_day=TimeOfDay.NIGHT)
        assert v1.key_component() != v2.key_component()

    def test_to_dict_roundtrip(self):
        v = ContextVector(
            time_of_day=TimeOfDay.DUSK,
            operational_mode=OperationalMode.FISHING,
            location_state=LocationState.UNDERWAY,
            season="summer",
        )
        d = v.to_dict()
        assert d["time_of_day"] == "dusk"
        assert d["operational_mode"] == "fishing"
        assert d["location_state"] == "underway"
        assert d["season"] == "summer"


class TestContext:
    def test_build_from_components(self):
        ctx = Context(
            time_of_day="dawn",
            operational_mode="cruising",
            location_state="underway",
        )
        assert ctx.vector.time_of_day == TimeOfDay.DAWN
        assert ctx.vector.operational_mode == OperationalMode.CRUISING
        assert ctx.vector.location_state == LocationState.UNDERWAY

    def test_build_from_enums(self):
        ctx = Context(
            time_of_day=TimeOfDay.NIGHT,
            operational_mode=OperationalMode.FISHING,
            location_state=LocationState.AT_ANCHOR,
        )
        assert ctx.vector.time_of_day == TimeOfDay.NIGHT
        assert ctx.vector.operational_mode == OperationalMode.FISHING
        assert ctx.vector.location_state == LocationState.AT_ANCHOR

    def test_auto_detect_time_of_day(self):
        ctx = Context.from_datetime(datetime(2026, 8, 4, 6, 30))
        assert ctx.vector.time_of_day == TimeOfDay.DAWN

    def test_auto_detect_season(self):
        ctx = Context.from_datetime(datetime(2026, 8, 4, 12, 0))
        assert ctx.vector.season == "summer"

        ctx2 = Context.from_datetime(datetime(2026, 12, 15, 12, 0))
        assert ctx2.vector.season == "winter"

    def test_recent_commands_produce_hash(self):
        ctx1 = Context(recent_commands=["check weather", "plot course"])
        ctx2 = Context(recent_commands=["check weather", "plot course"])
        ctx3 = Context(recent_commands=["check weather", "set speed"])

        # Same commands → same hash
        assert ctx1.vector.recent_command_hash == ctx2.vector.recent_command_hash
        # Different commands → different hash
        assert ctx1.vector.recent_command_hash != ctx3.vector.recent_command_hash

    def test_no_recent_commands_empty_hash(self):
        ctx = Context()
        assert ctx.vector.recent_command_hash == ""

    def test_invalid_enum_value_raises(self):
        with pytest.raises(ValueError):
            Context(time_of_day="invalid_value")

    def test_season_from_datetime(self):
        assert Context._season_from_datetime(datetime(2026, 3, 15)) == "spring"
        assert Context._season_from_datetime(datetime(2026, 6, 15)) == "summer"
        assert Context._season_from_datetime(datetime(2026, 9, 15)) == "autumn"
        assert Context._season_from_datetime(datetime(2026, 12, 15)) == "winter"


class TestReflexKey:
    def test_same_text_same_context_same_key(self):
        ctx = Context(time_of_day="dawn", operational_mode="cruising")
        k1 = build_reflex_key("check the weather", ctx.vector)
        k2 = build_reflex_key("check the weather", ctx.vector)
        assert k1 == k2

    def test_different_text_different_key(self):
        ctx = Context()
        k1 = build_reflex_key("check the weather", ctx.vector)
        k2 = build_reflex_key("check the tide", ctx.vector)
        assert k1 != k2

    def test_same_text_different_context_different_key(self):
        ctx1 = Context(time_of_day="dawn", operational_mode="docked")
        ctx2 = Context(time_of_day="night", operational_mode="cruising", location_state="underway")
        k1 = build_reflex_key("check depth", ctx1.vector)
        k2 = build_reflex_key("check depth", ctx2.vector)
        assert k1 != k2

    def test_whitespace_normalization(self):
        ctx = Context()
        k1 = build_reflex_key("check   the   weather", ctx.vector)
        k2 = build_reflex_key("check the weather", ctx.vector)
        assert k1 == k2

    def test_leading_trailing_whitespace_normalized(self):
        ctx = Context()
        k1 = build_reflex_key("  check the weather  ", ctx.vector)
        k2 = build_reflex_key("check the weather", ctx.vector)
        assert k1 == k2

    def test_case_insensitive(self):
        ctx = Context()
        k1 = build_reflex_key("Check The Weather", ctx.vector)
        k2 = build_reflex_key("check the weather", ctx.vector)
        assert k1 == k2

    def test_key_is_hex_string(self):
        ctx = Context()
        key = build_reflex_key("test", ctx.vector)
        assert len(key) == 64  # SHA-256 hex digest
        int(key, 16)  # Should not raise — valid hex
