"""Tests for ConditionSeries and extrapolation policies."""

from datetime import datetime, timedelta, timezone

import pytest

from world_model.contracts.condition import (
    ConditionSeries,
    ExtrapolationPolicy,
    InterpolationPolicy,
)
from world_model.contracts.errors import (
    ConditionCoverageInvalidError,
    InvalidInputError,
    TimeRangeInvalidError,
)


def test_condition_series_rejects_forbidden_extrapolation(base_utc_time: datetime) -> None:
    """When extrapolation_policy=forbidden, queries outside coverage must fail."""
    t0 = base_utc_time
    t1 = base_utc_time + timedelta(seconds=10)

    series = ConditionSeries(
        condition_spec_ref="spec_tide",
        timestamps=[t0, t1],
        values=[1.0, 2.0],
        extrapolation_policy=ExtrapolationPolicy.FORBIDDEN,
    )

    # Within bounds -> valid
    series.validate_target_time(t0 + timedelta(seconds=5))

    # Before start -> raises ConditionCoverageInvalidError
    with pytest.raises(ConditionCoverageInvalidError):
        series.validate_target_time(t0 - timedelta(seconds=1))

    # After end -> raises ConditionCoverageInvalidError
    with pytest.raises(ConditionCoverageInvalidError):
        series.validate_target_time(t1 + timedelta(seconds=1))


def test_condition_series_strictly_monotonic_timestamps(base_utc_time: datetime) -> None:
    """Timestamps must be strictly monotonic increasing."""
    t0 = base_utc_time
    # Non-monotonic
    with pytest.raises(TimeRangeInvalidError):
        ConditionSeries(
            condition_spec_ref="spec_test",
            timestamps=[t0, t0],  # duplicate time
            values=[1.0, 2.0],
        )

    # Decreasing
    with pytest.raises(TimeRangeInvalidError):
        ConditionSeries(
            condition_spec_ref="spec_test",
            timestamps=[t0, t0 - timedelta(seconds=1)],
            values=[1.0, 2.0],
        )


def test_condition_series_requires_exactly_one_data_source(base_utc_time: datetime) -> None:
    """ConditionSeries must specify values XOR artifact_ref, never both or neither."""
    t0 = base_utc_time
    t1 = t0 + timedelta(seconds=1)

    # Neither provided -> fail
    with pytest.raises(InvalidInputError, match="values XOR artifact_ref"):
        ConditionSeries(
            condition_spec_ref="spec_test",
            timestamps=[t0, t1],
        )

    # Both provided -> fail
    with pytest.raises(InvalidInputError, match="values XOR artifact_ref"):
        ConditionSeries(
            condition_spec_ref="spec_test",
            timestamps=[t0, t1],
            values=[1.0, 2.0],
            artifact_ref="data/condition.h5",
        )

    # Only artifact_ref -> success
    series_artifact = ConditionSeries(
        condition_spec_ref="spec_test",
        timestamps=[t0, t1],
        artifact_ref="data/condition.h5",
    )
    assert series_artifact.artifact_ref == "data/condition.h5"


def test_condition_series_inline_values_match_timestamps(base_utc_time: datetime) -> None:
    """When using inline values, len(values) must equal len(timestamps)."""
    t0 = base_utc_time
    t1 = t0 + timedelta(seconds=1)

    # Length mismatch: 2 timestamps, 3 values
    with pytest.raises(InvalidInputError, match="values length .* must match timestamps length"):
        ConditionSeries(
            condition_spec_ref="spec_test",
            timestamps=[t0, t1],
            values=[1.0, 2.0, 3.0],
        )

    # Length mismatch: 2 timestamps, 1 value
    with pytest.raises(InvalidInputError, match="values length .* must match timestamps length"):
        ConditionSeries(
            condition_spec_ref="spec_test",
            timestamps=[t0, t1],
            values=[1.0],
        )


def test_condition_series_validity_interval_covers_samples(base_utc_time: datetime) -> None:
    """If validity_interval is declared, it must cover [timestamps[0], timestamps[-1]]."""
    t0 = base_utc_time
    t1 = t0 + timedelta(seconds=10)

    # Start after first timestamp -> fail
    with pytest.raises(TimeRangeInvalidError, match="must cover all sample timestamps"):
        ConditionSeries(
            condition_spec_ref="spec_test",
            timestamps=[t0, t1],
            values=[1.0, 2.0],
            validity_interval=(t0 + timedelta(seconds=1), t1),
        )

    # End before last timestamp -> fail
    with pytest.raises(TimeRangeInvalidError, match="must cover all sample timestamps"):
        ConditionSeries(
            condition_spec_ref="spec_test",
            timestamps=[t0, t1],
            values=[1.0, 2.0],
            validity_interval=(t0, t1 - timedelta(seconds=1)),
        )

    # Covering interval -> success
    series = ConditionSeries(
        condition_spec_ref="spec_test",
        timestamps=[t0, t1],
        values=[1.0, 2.0],
        validity_interval=(t0 - timedelta(seconds=5), t1 + timedelta(seconds=5)),
    )
    assert series.validity_interval is not None
