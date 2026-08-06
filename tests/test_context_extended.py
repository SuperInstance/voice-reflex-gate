"""Extended tests for context — edge cases, season boundaries, enum coercion, key construction."""

from datetime import datetime
from hashlib import sha256

import pytest

from voice_reflex.context import (
    Context,
    ContextVector,
    LocationState,
    OperationalMode,
    TimeOfDay,
    build_reflex_key,
)


class TestTimeOfDayBoundaries:
    """Exhaustive boundary testing for TimeOfDay classification."""

    @pytest.mark.parametrize("hour,expected", [
        (0, TimeOfDay.NIGHT),
        (1, TimeOfDay.NIGHT),
        (2, TimeOfDay.NIGHT),
        (3, TimeOfDay.NIGHT),
        (4, TimeOfDay.NIGHT),
        (5, TimeOfDay.DAWN),
        (6, TimeOfDay.DAWN),
        (7, TimeOfDay.DAWN),
        (8, TimeOfDay.DAWN),
        (9, TimeOfDay.MIDDAY),
        (10, TimeOfDay.MIDDAY),
        (11, TimeOfDay.MIDDAY),
        (12, TimeOfDay.MIDDAY),
        (13, TimeOfDay.MIDDAY),
        (14, TimeOfDay.MIDDAY),
        (15, TimeOfDay.MIDDAY),
        (16, TimeOfDay.MIDDAY),
        (17, TimeOfDay.DUSK),
        (18, TimeOfDay.DUSK),
        (19, TimeOfDay.DUSK),
        (20, TimeOfDay.DUSK),
        (21, TimeOfDay.NIGHT),
        (22, TimeOfDay.NIGHT),
        (23, TimeOfDay.NIGHT),
    ])
    def test_all_hours_classified_correctly(self, hour, expected):
        dt = datetime(2026, 8, 4, hour, 0)
        assert TimeOfDay.from_datetime(dt) == expected

    def test_boundary_4_59(self):
        """4:59 is night, 5:00 is dawn."""
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 4, 59)) == TimeOfDay.NIGHT
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 5, 0)) == TimeOfDay.DAWN

    def test_boundary_8_59(self):
        """8:59 is dawn, 9:00 is midday."""
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 8, 59)) == TimeOfDay.DAWN
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 9, 0)) == TimeOfDay.MIDDAY

    def test_boundary_16_59(self):
        """16:59 is midday, 17:00 is dusk."""
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 16, 59)) == TimeOfDay.MIDDAY
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 17, 0)) == TimeOfDay.DUSK

    def test_boundary_20_59(self):
        """20:59 is dusk, 21:00 is night."""
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 20, 59)) == TimeOfDay.DUSK
        assert TimeOfDay.from_datetime(datetime(2026, 8, 4, 21, 0)) == TimeOfDay.NIGHT


class TestTimeOfDayEnum:
    def test_enum_values(self):
        assert TimeOfDay.DAWN.value == "dawn"
        assert TimeOfDay.MIDDAY.value == "midday"
        assert TimeOfDay.DUSK.value == "dusk"
        assert TimeOfDay.NIGHT.value == "night"

    def test_enum_is_string(self):
        assert isinstance(TimeOfDay.DAWN, str)
        assert TimeOfDay.DAWN == "dawn"  # str enum equality


class TestOperationalModeEnum:
    @pytest.mark.parametrize("mode", [
        OperationalMode.DOCKED,
        OperationalMode.CRUISING,
        OperationalMode.FISHING,
        OperationalMode.ANCHORED,
        OperationalMode.EMERGENCY,
        OperationalMode.MAINTENANCE,
    ])
    def test_all_modes_have_values(self, mode):
        assert mode.value
        assert isinstance(mode.value, str)


class TestLocationStateEnum:
    @pytest.mark.parametrize("state", [
        LocationState.UNDERWAY,
        LocationState.AT_ANCHOR,
        LocationState.AT_DOCK,
        LocationState.IN_HARBOR,
        LocationState.OPEN_WATER,
    ])
    def test_all_states_have_values(self, state):
        assert state.value
        assert isinstance(state.value, str)


class TestContextVectorKeyComponent:
    def test_key_component_format(self):
        v = ContextVector(
            time_of_day=TimeOfDay.DAWN,
            operational_mode=OperationalMode.CRUISING,
            location_state=LocationState.UNDERWAY,
            recent_command_hash="abcdef1234567890",
            season="summer",
        )
        key = v.key_component()
        parts = key.split("|")
        assert len(parts) == 5
        assert parts[0] == "dawn"
        assert parts[1] == "cruising"
        assert parts[2] == "underway"
        assert parts[3] == "abcdef12"  # first 8 chars of hash
        assert parts[4] == "summer"

    def test_key_component_empty_hash_shows_none(self):
        v = ContextVector(recent_command_hash="")
        key = v.key_component()
        parts = key.split("|")
        assert parts[3] == "none"

    def test_key_component_empty_season_shows_unspecified(self):
        v = ContextVector(season="")
        key = v.key_component()
        parts = key.split("|")
        assert parts[4] == "unspecified"

    def test_key_component_short_hash_uses_full(self):
        """Hash shorter than 8 chars is used as-is."""
        v = ContextVector(recent_command_hash="abc")
        key = v.key_component()
        parts = key.split("|")
        assert parts[3] == "abc"

    def test_different_command_hashes_produce_different_keys(self):
        v1 = ContextVector(recent_command_hash="aaa")
        v2 = ContextVector(recent_command_hash="bbb")
        assert v1.key_component() != v2.key_component()

    def test_frozen_dataclass_is_hashable(self):
        v = ContextVector()
        # frozen=True means it's hashable
        hash(v)  # should not raise


class TestContextVectorToDict:
    def test_to_dict_has_all_fields(self):
        v = ContextVector(
            time_of_day=TimeOfDay.NIGHT,
            operational_mode=OperationalMode.FISHING,
            location_state=LocationState.AT_ANCHOR,
            recent_command_hash="abc123",
            season="autumn",
        )
        d = v.to_dict()
        assert set(d.keys()) == {
            "time_of_day", "operational_mode", "location_state",
            "recent_command_hash", "season",
        }
        assert d["time_of_day"] == "night"
        assert d["operational_mode"] == "fishing"
        assert d["location_state"] == "at_anchor"
        assert d["recent_command_hash"] == "abc123"
        assert d["season"] == "autumn"


class TestContextCoercion:
    def test_coerce_enum_by_value(self):
        ctx = Context(time_of_day="dawn", operational_mode="cruising")
        assert ctx.vector.time_of_day == TimeOfDay.DAWN
        assert ctx.vector.operational_mode == OperationalMode.CRUISING

    def test_coerce_enum_by_name_case_insensitive(self):
        ctx = Context(
            time_of_day="DAWN",
            operational_mode="CRUISING",
            location_state="UNDERWAY",
        )
        assert ctx.vector.time_of_day == TimeOfDay.DAWN
        assert ctx.vector.operational_mode == OperationalMode.CRUISING
        assert ctx.vector.location_state == LocationState.UNDERWAY

    def test_coerce_enum_already_enum(self):
        ctx = Context(
            time_of_day=TimeOfDay.NIGHT,
            operational_mode=OperationalMode.EMERGENCY,
        )
        assert ctx.vector.time_of_day == TimeOfDay.NIGHT
        assert ctx.vector.operational_mode == OperationalMode.EMERGENCY

    def test_coerce_invalid_time_of_day(self):
        with pytest.raises(ValueError, match="Cannot coerce"):
            Context(time_of_day="invalid")

    def test_coerce_invalid_operational_mode(self):
        with pytest.raises(ValueError, match="Cannot coerce"):
            Context(operational_mode="invalid")

    def test_coerce_invalid_location_state(self):
        with pytest.raises(ValueError, match="Cannot coerce"):
            Context(location_state="invalid")


class TestContextSeasonDetection:
    @pytest.mark.parametrize("month,season", [
        (1, "winter"), (2, "winter"),
        (3, "spring"), (4, "spring"), (5, "spring"),
        (6, "summer"), (7, "summer"), (8, "summer"),
        (9, "autumn"), (10, "autumn"), (11, "autumn"),
        (12, "winter"),
    ])
    def test_all_months_correct_season(self, month, season):
        dt = datetime(2026, month, 15)
        assert Context._season_from_datetime(dt) == season

    def test_season_boundary_months(self):
        """Test months at season boundaries."""
        # Feb (winter) → Mar (spring)
        assert Context._season_from_datetime(datetime(2026, 2, 28)) == "winter"
        assert Context._season_from_datetime(datetime(2026, 3, 1)) == "spring"

        # May (spring) → Jun (summer)
        assert Context._season_from_datetime(datetime(2026, 5, 31)) == "spring"
        assert Context._season_from_datetime(datetime(2026, 6, 1)) == "summer"

        # Aug (summer) → Sep (autumn)
        assert Context._season_from_datetime(datetime(2026, 8, 31)) == "summer"
        assert Context._season_from_datetime(datetime(2026, 9, 1)) == "autumn"

        # Nov (autumn) → Dec (winter)
        assert Context._season_from_datetime(datetime(2026, 11, 30)) == "autumn"
        assert Context._season_from_datetime(datetime(2026, 12, 1)) == "winter"


class TestContextAutoDetection:
    def test_timestamp_auto_detects_time_of_day(self):
        """When timestamp is given and time_of_day is default (midday), auto-detect from timestamp."""
        ctx = Context(timestamp=datetime(2026, 8, 4, 6, 0))
        # 6am → dawn (auto-detected)
        assert ctx.vector.time_of_day == TimeOfDay.DAWN

    def test_explicit_time_of_day_not_overridden_by_timestamp(self):
        """If user explicitly sets time_of_day, timestamp doesn't override it."""
        ctx = Context(
            time_of_day=TimeOfDay.NIGHT,
            timestamp=datetime(2026, 8, 4, 6, 0),
        )
        assert ctx.vector.time_of_day == TimeOfDay.NIGHT

    def test_timestamp_auto_detects_season(self):
        ctx = Context(timestamp=datetime(2026, 1, 15))
        assert ctx.vector.season == "winter"

    def test_explicit_season_not_overridden(self):
        ctx = Context(season="summer", timestamp=datetime(2026, 1, 15))
        assert ctx.vector.season == "summer"


class TestContextRecentCommands:
    def test_last_10_commands_used(self):
        """Only the last 10 commands are hashed."""
        cmds = [f"command_{i}" for i in range(20)]
        ctx1 = Context(recent_commands=cmds)
        ctx2 = Context(recent_commands=cmds[-10:])
        # Should be the same since only last 10 matter
        assert ctx1.vector.recent_command_hash == ctx2.vector.recent_command_hash

    def test_single_command(self):
        ctx = Context(recent_commands=["check weather"])
        assert ctx.vector.recent_command_hash
        # Verify the hash is correct
        expected = sha256("check weather".encode("utf-8")).hexdigest()
        assert ctx.vector.recent_command_hash == expected

    def test_command_order_matters(self):
        ctx1 = Context(recent_commands=["a", "b"])
        ctx2 = Context(recent_commands=["b", "a"])
        assert ctx1.vector.recent_command_hash != ctx2.vector.recent_command_hash

    def test_empty_commands_no_hash(self):
        ctx = Context(recent_commands=[])
        assert ctx.vector.recent_command_hash == ""

    def test_default_no_commands(self):
        ctx = Context()
        assert ctx.vector.recent_command_hash == ""


class TestContextFromDatetime:
    def test_builds_full_context(self):
        dt = datetime(2026, 8, 4, 6, 30)
        ctx = Context.from_datetime(dt, operational_mode="fishing", location_state="underway")
        assert ctx.vector.time_of_day == TimeOfDay.DAWN
        assert ctx.vector.operational_mode == OperationalMode.FISHING
        assert ctx.vector.location_state == LocationState.UNDERWAY
        assert ctx.vector.season == "summer"

    def test_with_recent_commands(self):
        dt = datetime(2026, 6, 15, 12, 0)
        ctx = Context.from_datetime(dt, recent_commands=["check depth"])
        assert ctx.vector.recent_command_hash

    def test_with_explicit_season(self):
        dt = datetime(2026, 6, 15, 12, 0)
        ctx = Context.from_datetime(dt, season="winter")
        assert ctx.vector.season == "winter"


class TestContextDirectVector:
    def test_provided_vector_used_directly(self):
        vec = ContextVector(
            time_of_day=TimeOfDay.NIGHT,
            operational_mode=OperationalMode.EMERGENCY,
        )
        ctx = Context(_vector=vec)
        assert ctx.vector is vec

    def test_provided_vector_ignores_components(self):
        vec = ContextVector(time_of_day=TimeOfDay.NIGHT)
        ctx = Context(
            _vector=vec,
            time_of_day="dawn",  # should be ignored
        )
        assert ctx.vector.time_of_day == TimeOfDay.NIGHT


class TestBuildReflexKey:
    def test_key_is_sha256_hex(self):
        ctx = Context()
        key = build_reflex_key("test", ctx.vector)
        # SHA-256 produces 64 hex chars
        assert len(key) == 64
        int(key, 16)  # valid hex

    def test_key_deterministic_across_calls(self):
        ctx = Context()
        keys = [build_reflex_key("check weather", ctx.vector) for _ in range(10)]
        assert all(k == keys[0] for k in keys)

    def test_empty_stt_text(self):
        ctx = Context()
        key = build_reflex_key("", ctx.vector)
        assert len(key) == 64

    def test_tab_whitespace_normalized(self):
        ctx = Context()
        k1 = build_reflex_key("check\tweather", ctx.vector)
        k2 = build_reflex_key("check weather", ctx.vector)
        assert k1 == k2

    def test_newline_whitespace_normalized(self):
        ctx = Context()
        k1 = build_reflex_key("check\nweather", ctx.vector)
        k2 = build_reflex_key("check weather", ctx.vector)
        assert k1 == k2

    def test_mixed_case_normalized(self):
        ctx = Context()
        k1 = build_reflex_key("CHECK WEATHER", ctx.vector)
        k2 = build_reflex_key("check weather", ctx.vector)
        assert k1 == k2

    def test_different_context_vectors_different_keys(self):
        ctx1 = Context(time_of_day="dawn")
        ctx2 = Context(time_of_day="dusk")
        k1 = build_reflex_key("check weather", ctx1.vector)
        k2 = build_reflex_key("check weather", ctx2.vector)
        assert k1 != k2

    def test_key_uses_pipe_separated_context(self):
        """Verify the key format includes the context vector key component."""
        ctx = Context(time_of_day="dawn", operational_mode="cruising")
        normalized = "check weather"
        combined = f"{normalized}::{ctx.vector.key_component()}"
        expected = sha256(combined.encode("utf-8")).hexdigest()
        assert build_reflex_key("check weather", ctx.vector) == expected


class TestContextRepr:
    def test_context_repr_does_not_explode(self):
        ctx = Context(time_of_day="dawn", operational_mode="cruising")
        repr_str = repr(ctx)
        assert "Context" in repr_str or "_vector" in repr_str
