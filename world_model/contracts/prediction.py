"""PredictionRequest and WorldPrediction contracts.

Defines the system-level interface:
PredictionRequest -> WorldModelBackend -> WorldPrediction
Phase 1: Deterministic prediction enforces K=1 trajectory.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from world_model.contracts.action import ActionSequence
from world_model.contracts.common import (
    ContractBase,
    TimezoneAwareDatetime,
    current_utc_time,
    generate_id,
)
from world_model.contracts.errors import (
    InvalidInputError,
    OutputModeUnsupportedError,
    TimeRangeInvalidError,
)
from world_model.contracts.task_spec import TransitionTaskSpec
from world_model.contracts.trajectory import PredictedTrajectory
from world_model.contracts.world_context import WorldContext
from world_model.contracts.world_state import WorldState


class OutputMode(str, Enum):
    """Prediction output distribution mode."""

    DETERMINISTIC = "deterministic"
    ENSEMBLE = "ensemble"


class PredictionOptions(BaseModel):
    """Runtime options for prediction request."""

    output_mode: OutputMode = Field(
        default=OutputMode.DETERMINISTIC,
        description="deterministic (K=1) | ensemble (K>=1)",
    )
    num_trajectories: int = Field(
        default=1,
        description="Number of trajectories requested. Must equal 1 if deterministic.",
    )
    sampling_seed: Optional[int] = Field(
        default=None,
        description="RNG seed for stochastic/ensemble generation",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_deterministic_single_trajectory(self) -> PredictionOptions:
        if self.output_mode == OutputMode.DETERMINISTIC and self.num_trajectories != 1:
            raise OutputModeUnsupportedError(
                f"Deterministic output mode requires num_trajectories=1, got {self.num_trajectories}."
            )
        if self.num_trajectories < 1:
            raise InvalidInputError(f"num_trajectories must be >= 1, got {self.num_trajectories}.")
        return self


class PredictionRequest(ContractBase):
    """System-level input for world model inference.

    Enforces exactly-one-of mutual exclusion:
    - state_history_ref XOR state_history
    - context_ref XOR context
    - task_spec_ref XOR task_spec
    """

    contract_type: str = Field(default="PredictionRequest", description="Contract type identifier")
    request_id: str = Field(
        default_factory=lambda: generate_id("req"),
        description="Unique request identifier",
    )
    state_history_ref: Optional[str] = Field(
        default=None,
        description="External storage reference to state history",
    )
    state_history: Optional[list[WorldState]] = Field(
        default=None,
        description="Inline state history sequence",
    )
    context_ref: Optional[str] = Field(
        default=None,
        description="External storage reference or ID of WorldContext",
    )
    context: Optional[WorldContext] = Field(
        default=None,
        description="Inline WorldContext",
    )
    task_spec_ref: Optional[str] = Field(
        default=None,
        description="Reference or ID of TransitionTaskSpec",
    )
    task_spec: Optional[TransitionTaskSpec] = Field(
        default=None,
        description="Inline TransitionTaskSpec",
    )
    target_times: list[TimezoneAwareDatetime] = Field(
        ...,
        description="Concrete future valid times requested for prediction",
    )
    actions: Optional[ActionSequence] = Field(
        default=None,
        description="Action conditioning sequence (Phase 1: None)",
    )
    prediction_options: PredictionOptions = Field(
        default_factory=PredictionOptions,
        description="Inference runtime configuration",
    )
    strict: bool = Field(
        default=True,
        description="Phase 1 default: Fail atomically if any component/target cannot be satisfied",
    )

    @model_validator(mode="after")
    def validate_exactly_one_of_references(self) -> PredictionRequest:
        # 1. state_history_ref XOR state_history
        has_sh_ref = self.state_history_ref is not None
        has_sh_val = self.state_history is not None
        if has_sh_ref == has_sh_val:
            raise InvalidInputError(
                f"PredictionRequest must specify exactly one of (state_history_ref, state_history). "
                f"Got state_history_ref={has_sh_ref}, state_history={has_sh_val}."
            )

        # 2. context_ref XOR context
        has_ctx_ref = self.context_ref is not None
        has_ctx_val = self.context is not None
        if has_ctx_ref == has_ctx_val:
            raise InvalidInputError(
                f"PredictionRequest must specify exactly one of (context_ref, context). "
                f"Got context_ref={has_ctx_ref}, context={has_ctx_val}."
            )

        # 3. task_spec_ref XOR task_spec
        has_ts_ref = self.task_spec_ref is not None
        has_ts_val = self.task_spec is not None
        if has_ts_ref == has_ts_val:
            raise InvalidInputError(
                f"PredictionRequest must specify exactly one of (task_spec_ref, task_spec). "
                f"Got task_spec_ref={has_ts_ref}, task_spec={has_ts_val}."
            )

        if not self.target_times:
            raise InvalidInputError("PredictionRequest target_times must not be empty.")

        # Check monotonic increasing
        for i in range(len(self.target_times) - 1):
            if self.target_times[i] >= self.target_times[i + 1]:
                raise TimeRangeInvalidError(
                    f"PredictionRequest target_times must be strictly monotonic increasing: "
                    f"{self.target_times[i]} >= {self.target_times[i + 1]}"
                )

        return self


class WorldPrediction(ContractBase):
    """Unified system-level world model output.

    Contains 1..K trajectories of predicted states.
    Phase 1 deterministic baseline enforces K=1.
    """

    contract_type: str = Field(default="WorldPrediction", description="Contract type identifier")
    prediction_id: str = Field(
        default_factory=lambda: generate_id("pred"),
        description="Unique prediction result identifier",
    )
    request_id: str = Field(..., description="ID of corresponding PredictionRequest")
    generated_at: TimezoneAwareDatetime = Field(
        default_factory=current_utc_time,
        description="Timestamp when prediction was computed (not state valid time)",
    )
    base_state_ids: list[str] = Field(
        default_factory=list,
        description="IDs of base conditioning history states",
    )
    predicted_components: list[str] = Field(
        default_factory=lambda: ["environment.fluid_state"],
        description="Derived summary of predicted components from TaskSpec",
    )
    trajectories: list[PredictedTrajectory] = Field(
        ...,
        description="List of predicted rollout trajectories [1..K]",
    )
    uncertainty_summary: Optional[dict[str, Any]] = Field(
        default=None,
        description="Calibration and uncertainty quantification metadata",
    )
    validity: Optional[dict[str, Any]] = Field(
        default=None,
        description="Prediction validity and convergence diagnostic bounds",
    )
    model_manifest_ref: Optional[str] = Field(
        default=None,
        description="Reference to ModelManifest used for inference",
    )
    context_ref: Optional[str] = Field(
        default=None,
        description="Reference or ID of WorldContext used",
    )
    task_spec_ref: Optional[str] = Field(
        default=None,
        description="Reference or ID of TransitionTaskSpec satisfied",
    )
    status: str = Field(
        default="success",
        description="Execution status: success | partial | failed",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Diagnostic warnings encountered during rollout",
    )

    @model_validator(mode="after")
    def validate_trajectories_invariant(self) -> WorldPrediction:
        if self.status == "success" and not self.trajectories:
            raise InvalidInputError("Successful WorldPrediction must contain at least one trajectory.")
        return self
