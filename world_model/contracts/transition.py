"""TransitionSample training data unit contract.

Encapsulates one state transition sample for training and offline validation.
Single source of truth: history_times and target_times are strictly derived from
WorldState.timestamp, never duplicated as stored attributes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import Field, model_validator

from world_model.contracts.action import ActionSequence
from world_model.contracts.common import ContractBase, generate_id
from world_model.contracts.errors import InvalidInputError, TimeRangeInvalidError
from world_model.contracts.world_context import WorldContext
from world_model.contracts.world_state import WorldState


class TransitionSample(ContractBase):
    """Training data logic unit representing (W_history, Context, Actions) -> W_target.

    history_times and target_times are derived properties from WorldState.timestamp.
    """

    contract_type: str = Field(default="TransitionSample", description="Contract type identifier")
    sample_id: str = Field(
        default_factory=lambda: generate_id("sample"),
        description="Unique sample identifier",
    )
    task_spec_ref: str = Field(..., description="Reference to TransitionTaskSpec")
    state_history: list[WorldState] = Field(
        ...,
        description="Sequence of historical world states [W_{t - L + 1}, ..., W_t]",
    )
    context: WorldContext = Field(..., description="Physical context governing this transition")
    actions: Optional[ActionSequence] = Field(
        default=None,
        description="Robot control sequence (Phase 1: None)",
    )
    target_states: list[WorldState] = Field(
        ...,
        description="Sequence of future target states [W_{t + tau_1}, ..., W_{t + tau_H}]",
    )
    trajectory_id: str = Field(
        ...,
        description="Identifier of parent trajectory from dataset manifest",
    )
    dataset_manifest_ref: Optional[str] = Field(
        default=None,
        description="Reference to DatasetManifest providing this sample",
    )
    provenance: Optional[dict[str, Any]] = Field(
        default=None,
        description="Dataset slice indices, window offset, normalization params",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_duplicate_times_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for forbidden_key in ("history_times", "target_times"):
                if forbidden_key in values:
                    raise InvalidInputError(
                        f"TransitionSample must not store '{forbidden_key}'. "
                        f"Times must be derived dynamically from WorldState.timestamp.",
                        details={"forbidden_key": forbidden_key},
                    )
        return values

    @model_validator(mode="after")
    def validate_temporal_causality(self) -> TransitionSample:
        if not self.state_history:
            raise InvalidInputError("TransitionSample state_history must not be empty.")
        if not self.target_states:
            raise InvalidInputError("TransitionSample target_states must not be empty.")

        # Check history monotonic increasing
        for i in range(len(self.state_history) - 1):
            if self.state_history[i].timestamp >= self.state_history[i + 1].timestamp:
                raise TimeRangeInvalidError(
                    f"state_history timestamps must be strictly monotonic increasing: "
                    f"{self.state_history[i].timestamp} >= {self.state_history[i + 1].timestamp}"
                )

        # Check target monotonic increasing
        for i in range(len(self.target_states) - 1):
            if self.target_states[i].timestamp >= self.target_states[i + 1].timestamp:
                raise TimeRangeInvalidError(
                    f"target_states timestamps must be strictly monotonic increasing: "
                    f"{self.target_states[i].timestamp} >= {self.target_states[i + 1].timestamp}"
                )

        # Check causality: target states must be after the last history state
        last_hist_t = self.state_history[-1].timestamp
        first_target_t = self.target_states[0].timestamp
        if first_target_t <= last_hist_t:
            raise TimeRangeInvalidError(
                f"Causal violation: first target timestamp ({first_target_t}) "
                f"must be strictly after last history timestamp ({last_hist_t})."
            )

        return self

    @property
    def history_times(self) -> list[datetime]:
        """Derived authoritative history valid times."""
        return [s.timestamp for s in self.state_history]

    @property
    def target_times(self) -> list[datetime]:
        """Derived authoritative target valid times."""
        return [s.timestamp for s in self.target_states]
