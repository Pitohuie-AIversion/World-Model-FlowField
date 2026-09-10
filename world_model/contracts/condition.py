"""Condition and ConditionSeries contracts.

Temporal conditions driving physical state evolution.
Phase 1 default prohibits silent extrapolation outside coverage.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from world_model.contracts.common import TimezoneAwareDatetime
from world_model.contracts.errors import (
    ConditionCoverageInvalidError,
    InvalidInputError,
    TimeRangeInvalidError,
)


class InterpolationPolicy(str, Enum):
    """Interpolation method inside valid condition coverage."""

    EXACT = "exact"
    LINEAR = "linear"
    STEP = "step"
    MODEL_SPECIFIC = "model_specific"


class ExtrapolationPolicy(str, Enum):
    """Behavior when condition is queried outside validity interval."""

    FORBIDDEN = "forbidden"
    CONSTANT_EDGE = "constant_edge"
    PERIODIC = "periodic"


class ConditionSpec(BaseModel):
    """Metadata specification for a physical condition."""

    canonical_name: str = Field(..., description="Canonical condition name")
    units: str = Field(..., description="Physical units string")
    value_type: str = Field(default="scalar", description="scalar | vector | tensor | field")
    role: str = Field(
        default="forcing",
        description="Physical role: boundary | forcing | parameter | auxiliary",
    )

    model_config = ConfigDict(extra="forbid")


class ConditionSeries(BaseModel):
    """Time series of external physical conditions (e.g. boundary conditions, forcing).

    Default extrapolation policy is 'forbidden' to prevent silent drift beyond coverage.
    """

    condition_spec_ref: str = Field(..., description="Reference to ConditionSpec")
    timestamps: list[TimezoneAwareDatetime] = Field(
        ...,
        description="Chronologically sorted sample timestamps",
    )
    values: Optional[Any] = Field(
        default=None,
        description="Inline values corresponding to timestamps",
    )
    artifact_ref: Optional[str] = Field(
        default=None,
        description="Reference to external file/array if not stored inline",
    )
    source: Optional[str] = Field(
        default=None,
        description="Provenance of condition data (e.g. tide gauge, ECMWF, prescribed)",
    )
    interpolation_policy: InterpolationPolicy = Field(
        default=InterpolationPolicy.LINEAR,
        description="Interpolation policy within valid interval",
    )
    extrapolation_policy: ExtrapolationPolicy = Field(
        default=ExtrapolationPolicy.FORBIDDEN,
        description="Default is forbidden; silent extrapolation outside coverage is prohibited",
    )
    availability_policy: Optional[str] = Field(
        default=None,
        description="Causal availability semantics at inference time",
    )
    validity_interval: Optional[tuple[TimezoneAwareDatetime, TimezoneAwareDatetime]] = Field(
        default=None,
        description="Explicit [start, end] valid interval. If None, derived from timestamps.",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_timestamps_and_interval(self) -> ConditionSeries:
        if not self.timestamps:
            raise InvalidInputError("ConditionSeries timestamps list must not be empty.")

        # Check strictly monotonic increasing
        for i in range(len(self.timestamps) - 1):
            if self.timestamps[i] >= self.timestamps[i + 1]:
                raise TimeRangeInvalidError(
                    f"ConditionSeries timestamps must be strictly monotonic: "
                    f"{self.timestamps[i]} >= {self.timestamps[i + 1]}"
                )

        if self.validity_interval is not None:
            t_start, t_end = self.validity_interval
            if t_start > t_end:
                raise TimeRangeInvalidError(
                    f"Invalid validity_interval: start {t_start} > end {t_end}"
                )

        return self

    def get_effective_validity_interval(self) -> tuple[datetime, datetime]:
        """Return explicit validity_interval or min/max timestamps."""
        if self.validity_interval is not None:
            return self.validity_interval
        return self.timestamps[0], self.timestamps[-1]

    def validate_target_time(self, target_time: datetime) -> None:
        """Validate whether target_time is covered.

        Raises ConditionCoverageInvalidError if out of bounds and extrapolation is forbidden.
        """
        if target_time.tzinfo is None:
            raise TimeRangeInvalidError("Target timestamp must be timezone-aware.")

        t_min, t_max = self.get_effective_validity_interval()
        if target_time < t_min or target_time > t_max:
            if self.extrapolation_policy == ExtrapolationPolicy.FORBIDDEN:
                raise ConditionCoverageInvalidError(
                    f"Target time {target_time.isoformat()} is outside condition coverage "
                    f"[{t_min.isoformat()}, {t_max.isoformat()}] and extrapolation_policy is 'forbidden'.",
                    details={
                        "target_time": target_time.isoformat(),
                        "coverage_start": t_min.isoformat(),
                        "coverage_end": t_max.isoformat(),
                        "extrapolation_policy": self.extrapolation_policy,
                    },
                )
