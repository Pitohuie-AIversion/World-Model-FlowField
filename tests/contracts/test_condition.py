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


def test_condition_series_forbidden_effective_interval(base_utc_time: datetime) -> None:
    """FORBIDDEN effective interval = samples ∩ declared.

    Correction: the old requirement that validity_interval must cover all
    sample timestamps was inaccurate. Under FORBIDDEN, widening the declared
    interval beyond sample support must NOT extend coverage. Narrowing is
    permitted and shrinks the queryable range.

    Acceptance matrix (seconds relative to base_utc_time):
        samples    declared      effective    must-pass   must-reject
        [0,10]     None          [0,10]       0,10        -1,11
        [0,10]     [-5,15]       [0,10]       0,10        -1,11
        [0,10]     [2,8]         [2,8]        2,5,8       1,9
        [0,10]     [8,2]         (invalid)    —           construct fails
        [0,10]     [20,30]       (no overlap) —           construct fails
        [5]        [0,10]        [5,5]        5           4,6
    """
    t = lambda s: base_utc_time + timedelta(seconds=s)  # noqa: E731

    # --- Case 1: No declared interval → effective = sample support [0, 10] ---
    s1 = ConditionSeries(
        condition_spec_ref="spec", timestamps=[t(0), t(10)], values=[1.0, 2.0],
        extrapolation_policy=ExtrapolationPolicy.FORBIDDEN,
    )
    assert s1.get_effective_validity_interval() == (t(0), t(10))
    s1.validate_target_time(t(0))
    s1.validate_target_time(t(10))
    with pytest.raises(ConditionCoverageInvalidError):
        s1.validate_target_time(t(-1))
    with pytest.raises(ConditionCoverageInvalidError):
        s1.validate_target_time(t(11))

    # --- Case 2: Widened interval [-5, 15] → effective clamped to [0, 10] ---
    s2 = ConditionSeries(
        condition_spec_ref="spec", timestamps=[t(0), t(10)], values=[1.0, 2.0],
        extrapolation_policy=ExtrapolationPolicy.FORBIDDEN,
        validity_interval=(t(-5), t(15)),
    )
    assert s2.get_effective_validity_interval() == (t(0), t(10))
    s2.validate_target_time(t(0))
    s2.validate_target_time(t(10))
    with pytest.raises(ConditionCoverageInvalidError):
        s2.validate_target_time(t(-1))
    with pytest.raises(ConditionCoverageInvalidError):
        s2.validate_target_time(t(11))

    # --- Case 3: Narrowed interval [2, 8] → effective = [2, 8] ---
    s3 = ConditionSeries(
        condition_spec_ref="spec", timestamps=[t(0), t(10)], values=[1.0, 2.0],
        extrapolation_policy=ExtrapolationPolicy.FORBIDDEN,
        validity_interval=(t(2), t(8)),
    )
    assert s3.get_effective_validity_interval() == (t(2), t(8))
    s3.validate_target_time(t(2))
    s3.validate_target_time(t(5))
    s3.validate_target_time(t(8))
    with pytest.raises(ConditionCoverageInvalidError):
        s3.validate_target_time(t(1))
    with pytest.raises(ConditionCoverageInvalidError):
        s3.validate_target_time(t(9))

    # --- Case 4: Inverted interval [8, 2] → construction fails ---
    with pytest.raises(TimeRangeInvalidError, match="start must not exceed end"):
        ConditionSeries(
            condition_spec_ref="spec", timestamps=[t(0), t(10)], values=[1.0, 2.0],
            validity_interval=(t(8), t(2)),
        )

    # --- Case 5: No overlap [20, 30] → construction fails ---
    with pytest.raises(TimeRangeInvalidError, match="does not overlap"):
        ConditionSeries(
            condition_spec_ref="spec", timestamps=[t(0), t(10)], values=[1.0, 2.0],
            validity_interval=(t(20), t(30)),
        )

    # --- Case 6: Single-point sample [5] with declared [0, 10] → effective [5, 5] ---
    s6 = ConditionSeries(
        condition_spec_ref="spec", timestamps=[t(5)], values=[1.0],
        extrapolation_policy=ExtrapolationPolicy.FORBIDDEN,
        validity_interval=(t(0), t(10)),
    )
    assert s6.get_effective_validity_interval() == (t(5), t(5))
    s6.validate_target_time(t(5))
    with pytest.raises(ConditionCoverageInvalidError):
        s6.validate_target_time(t(4))
    with pytest.raises(ConditionCoverageInvalidError):
        s6.validate_target_time(t(6))
