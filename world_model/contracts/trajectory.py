"""PredictedTrajectory contract.

Represents a single continuous rollout trajectory of predicted world states.
All states contained in PredictedTrajectory must strictly be state_kind=predicted.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from world_model.contracts.common import generate_id
from world_model.contracts.errors import (
    InvalidInputError,
    SchemaMismatchError,
    TimeRangeInvalidError,
)
from world_model.contracts.world_state import StateKind, WorldState


class PredictedTrajectory(BaseModel):
    """Single predicted future state rollout trajectory."""

    trajectory_id: str = Field(
        default_factory=lambda: generate_id("traj"),
        description="Unique trajectory identifier",
    )
    states: list[WorldState] = Field(
        ...,
        description="Chronologically ordered predicted future world states [W_{t + tau_1}, ...]",
    )
    sample_weight: Optional[float] = Field(
        default=None,
        description="Optional likelihood or ensemble weighting for probabilistic backends",
    )
    rollout_metadata: Optional[dict[str, Any]] = Field(
        default=None,
        description="Autoregressive step count, intermediate residuals, or lead time log",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_predicted_states(self) -> PredictedTrajectory:
        if not self.states:
            raise InvalidInputError("PredictedTrajectory states list must not be empty.")

        for idx, state in enumerate(self.states):
            if state.state_kind != StateKind.PREDICTED:
                raise SchemaMismatchError(
                    f"PredictedTrajectory state at index {idx} has invalid state_kind='{state.state_kind}'. "
                    f"All states in PredictedTrajectory must strictly have state_kind='predicted'.",
                    details={"index": idx, "state_id": state.state_id, "state_kind": state.state_kind},
                )

        for i in range(len(self.states) - 1):
            if self.states[i].timestamp >= self.states[i + 1].timestamp:
                raise TimeRangeInvalidError(
                    f"PredictedTrajectory timestamps must be strictly monotonic increasing: "
                    f"{self.states[i].timestamp} >= {self.states[i + 1].timestamp}"
                )

        return self
