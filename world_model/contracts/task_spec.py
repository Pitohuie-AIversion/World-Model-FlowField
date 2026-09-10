"""TransitionTaskSpec contract: Single Source of Truth for state transitions.

Answers: What state transition is being asked?
Explicitly DOES NOT own:
- dataset split / revision
- metric suite
- compute budget
- model architecture
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from world_model.contracts.common import ContractBase, generate_id
from world_model.contracts.errors import InvalidInputError
from world_model.contracts.world_state import StateKind

FORBIDDEN_TASK_SPEC_KEYS = {
    "dataset_split",
    "split",
    "dataset_manifest_ref",
    "metric_suite",
    "metrics",
    "compute_budget",
    "model_architecture",
    "model_id",
    "model_type",
}


class ActionPolicy(str, Enum):
    """Action conditioning requirements for transition."""

    NONE = "none"
    OPTIONAL = "optional"
    REQUIRED = "required"


class RolloutPolicy(str, Enum):
    """Rollout formulation expected for transition."""

    ONE_STEP = "one_step"
    LEAD_TIME = "lead_time"
    DIRECT_MULTI_STEP = "direct_multi_step"
    AUTOREGRESSIVE = "autoregressive"


class HistorySelection(BaseModel):
    """Specification of required historical observation steps."""

    history_offsets: Optional[list[int]] = Field(
        default=None,
        description="Explicit discrete step offsets, e.g. [-3, -2, -1, 0]",
    )
    window_size: Optional[int] = Field(
        default=None,
        description="Consecutive history length L",
    )
    step: Optional[int] = Field(
        default=1,
        description="Temporal stride between history points",
    )
    policy_name: Optional[str] = Field(
        default=None,
        description="Named selection policy if dynamic",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_history_params(self) -> HistorySelection:
        if self.window_size is not None and self.window_size <= 0:
            raise InvalidInputError(f"HistorySelection window_size must be > 0, got {self.window_size}.")
        if self.step is not None and self.step <= 0:
            raise InvalidInputError(f"HistorySelection step must be > 0, got {self.step}.")
        if self.history_offsets is not None:
            if not self.history_offsets:
                raise InvalidInputError("HistorySelection history_offsets list must not be empty if provided.")
            if any(offset > 0 for offset in self.history_offsets):
                raise InvalidInputError(
                    f"HistorySelection history_offsets must be non-positive (<= 0), got {self.history_offsets}."
                )
        return self


class TargetTimePolicy(BaseModel):
    """Specification of future target evaluation horizons."""

    target_offsets: Optional[list[int]] = Field(
        default=None,
        description="Discrete future offsets from t0, e.g. [1, 2, 4, 8]",
    )
    lead_times: Optional[list[float]] = Field(
        default=None,
        description="Continuous physical lead times tau in seconds",
    )
    horizon: Optional[int] = Field(
        default=None,
        description="Number of target rollout steps H",
    )
    step: Optional[int] = Field(
        default=1,
        description="Stride between rollout targets",
    )
    policy_name: Optional[str] = Field(
        default=None,
        description="Named target horizon policy",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_target_params(self) -> TargetTimePolicy:
        if self.horizon is not None and self.horizon <= 0:
            raise InvalidInputError(f"TargetTimePolicy horizon must be > 0, got {self.horizon}.")
        if self.step is not None and self.step <= 0:
            raise InvalidInputError(f"TargetTimePolicy step must be > 0, got {self.step}.")
        if self.target_offsets is not None:
            if any(offset <= 0 for offset in self.target_offsets):
                raise InvalidInputError(
                    f"TargetTimePolicy target_offsets must be strictly positive (> 0), got {self.target_offsets}."
                )
        if self.lead_times is not None:
            if any(lt <= 0 for lt in self.lead_times):
                raise InvalidInputError(
                    f"TargetTimePolicy lead_times must be strictly positive (> 0), got {self.lead_times}."
                )
        if self.horizon is not None and self.lead_times is not None:
            if self.horizon != len(self.lead_times):
                raise InvalidInputError(
                    f"TargetTimePolicy horizon ({self.horizon}) contradicts len(lead_times) ({len(self.lead_times)})."
                )
        return self


class StateKindPolicy(BaseModel):
    """Permitted state kinds for input and target."""

    allowed_input_kinds: list[StateKind] = Field(
        default_factory=lambda: [
            StateKind.SIMULATED_GROUND_TRUTH,
            StateKind.OBSERVED,
            StateKind.ESTIMATED,
        ],
        description="Permissible input state kinds",
    )
    allowed_target_kinds: list[StateKind] = Field(
        default_factory=lambda: [
            StateKind.SIMULATED_GROUND_TRUTH,
            StateKind.OBSERVED,
            StateKind.PREDICTED,
        ],
        description="Permissible target state kinds",
    )

    model_config = ConfigDict(extra="forbid")


class TransitionTaskSpec(ContractBase):
    """Canonical specification defining a state transition problem.

    Decoupled from model architectures and benchmark evaluation suites.
    """

    contract_type: str = Field(default="TransitionTaskSpec", description="Contract type identifier")
    task_spec_id: str = Field(
        default_factory=lambda: generate_id("task"),
        description="Unique task specification identifier",
    )
    input_components: list[str] = Field(
        default_factory=lambda: ["environment.fluid_state"],
        description="State components required as inputs",
    )
    target_components: list[str] = Field(
        default_factory=lambda: ["environment.fluid_state"],
        description="State components to be predicted",
    )
    history_selection: HistorySelection = Field(
        ...,
        description="Temporal history requirement",
    )
    target_time_policy: TargetTimePolicy = Field(
        ...,
        description="Temporal prediction horizon policy",
    )
    required_context_keys: list[str] = Field(
        default_factory=list,
        description="Keys in WorldContext required for transition (e.g. physics_parameters.Re)",
    )
    action_policy: ActionPolicy = Field(
        default=ActionPolicy.NONE,
        description="Action conditioning requirement (Phase 1: none)",
    )
    rollout_policy: RolloutPolicy = Field(
        default=RolloutPolicy.ONE_STEP,
        description="one_step | lead_time | direct_multi_step | autoregressive",
    )
    state_kind_policy: StateKindPolicy = Field(
        default_factory=StateKindPolicy,
        description="Permitted state kinds for inputs and targets",
    )

    @model_validator(mode="before")
    @classmethod
    def check_forbidden_task_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for k in FORBIDDEN_TASK_SPEC_KEYS:
                if k in values:
                    raise InvalidInputError(
                        f"TransitionTaskSpec must not define '{k}'. Task definition is decoupled from benchmark/model.",
                        details={"forbidden_key": k},
                    )
        return values
