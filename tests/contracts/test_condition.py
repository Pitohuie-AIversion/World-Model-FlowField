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
