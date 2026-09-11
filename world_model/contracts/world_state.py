"""WorldState domain contracts.

Represents an authoritative world state at a specific valid timestamp.
Supports Phase 1 environment.fluid_state, with reserved interfaces for robotics.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from world_model.contracts.common import (
    ContractBase,
    TimezoneAwareDatetime,
    generate_id,
)
from world_model.contracts.errors import (
    InvalidInputError,
    SchemaMismatchError,
)


class StateKind(str, Enum):
    """Categorical origin and epistemic status of a world state."""

    SIMULATED_GROUND_TRUTH = "simulated_ground_truth"
    OBSERVED = "observed"
    ESTIMATED = "estimated"
    PREDICTED = "predicted"


class FieldSpec(BaseModel):
    """Specification of a physical field in canonical representation."""

    canonical_name: str = Field(..., description="Canonical field identifier, e.g. velocity.x")
    units: str = Field(..., description="SI or physical units string, e.g. m/s, Pa")
    dtype: Optional[str] = Field(default="float32", description="Data type representation")
    components: Optional[int] = Field(default=1, description="Number of field components")
    role: str = Field(default="state", description="Role: state | auxiliary | diagnostic")

    model_config = ConfigDict(extra="forbid")


class FieldRef(BaseModel):
    """Reference or inline storage for a spatial field.

    Exactly one storage representation must be provided:
    - inline_value
    - tensor_ref
    - artifact_ref
    """

    field_spec_ref: Optional[str] = Field(default=None, description="Reference to canonical FieldSpec")
    inline_value: Optional[Any] = Field(default=None, description="Direct in-memory array/data")
    tensor_ref: Optional[str] = Field(default=None, description="In-memory or runtime tensor key")
    artifact_ref: Optional[str] = Field(default=None, description="Persistent file/HDF5/blob reference")

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_exactly_one_storage(self) -> FieldRef:
        count = sum(
            x is not None
            for x in (self.inline_value, self.tensor_ref, self.artifact_ref)
        )
        if count != 1:
            raise InvalidInputError(
                f"FieldRef requires exactly one of (inline_value, tensor_ref, artifact_ref), got {count}."
            )
        return self


class FluidState(BaseModel):
    """Fluid environment state representation for Phase 1.

    Fields are not hardcoded to 4 channels; any canonical fields mapping is permitted.
    """

    fields: Mapping[str, FieldRef] = Field(
        default_factory=dict,
        description="Canonical field name -> FieldRef mapping",
    )
    spatial_ref: str = Field(
        ...,
        description="Reference to GridSpec in WorldContext.static_conditions.spatial_domains",
    )
    mask_ref: Optional[str] = Field(
        default=None,
        description="Optional mask reference (e.g. solid boundaries, obstacle mask)",
    )

    model_config = ConfigDict(extra="forbid")


class EnvironmentComponents(BaseModel):
    """Environment subsystem components. Phase 1 activates fluid_state."""

    fluid_state: Optional[FluidState] = Field(
        default=None,
        description="Fluid field state component",
    )

    model_config = ConfigDict(extra="forbid")


class WorldComponents(BaseModel):
    """Multi-subsystem components collection for WorldState."""

    environment: EnvironmentComponents = Field(
        default_factory=EnvironmentComponents,
        description="Physical environment subsystem",
    )
    robot: Optional[Any] = Field(
        default=None,
        description="Reserved for robot state in future phases",
    )
    objects: Optional[Any] = Field(
        default=None,
        description="Reserved for discrete objects/obstacles in future phases",
    )
    relations: Optional[Any] = Field(
        default=None,
        description="Reserved for inter-object relations in future phases",
    )

    model_config = ConfigDict(extra="forbid")


class StateLineage(BaseModel):
    """Immutable provenance snapshot for a WorldState.

    All fields are frozen after construction. Container fields use tuples
    (not lists) to prevent in-place mutation via .append() / .extend().
    Pydantic non-strict mode coerces list inputs to tuples automatically.
    """

    source_state_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="IDs of previous states that produced this state",
    )
    source_observation_ids: tuple[str, ...] = Field(
        default_factory=tuple,
        description="IDs of raw ObservationFrames used if state was estimated",
    )
    parent_prediction_id: Optional[str] = Field(
        default=None,
        description="ID of WorldPrediction that generated this state (required if predicted)",
    )

    model_config = ConfigDict(extra="forbid", frozen=True)


class WorldState(ContractBase):
    """Top-level authoritative WorldState contract.

    timestamp is the single authoritative valid time of this state.
    state_kind explicitly separates ground truth, observations, estimations, and predictions.
    Predicted states never automatically promote to observed facts upon time passage.
    """

    contract_type: str = Field(default="WorldState", description="Contract type identifier")
    state_id: str = Field(
        default_factory=lambda: generate_id("ws"),
        description="Unique world state identifier",
    )
    state_kind: StateKind = Field(
        ...,
        description="simulated_ground_truth | observed | estimated | predicted",
    )
    timestamp: TimezoneAwareDatetime = Field(
        ...,
        description="Authoritative valid time of this state (must be timezone-aware)",
    )
    world_frame: str = Field(
        default="world",
        description="Default coordinate frame for spatial relations",
    )
    components: WorldComponents = Field(
        default_factory=WorldComponents,
        description="Subsystem components",
    )
    source: Optional[str] = Field(
        default=None,
        description="Origin description or sensor/simulation run tag",
    )
    quality: Optional[dict[str, Any]] = Field(
        default=None,
        description="Confidence, SNR, or simulation solver residual metadata",
    )
    lineage: StateLineage = Field(
        default_factory=StateLineage,
        description="Causal pedigree and parent prediction tracking",
    )
    provenance: Optional[dict[str, Any]] = Field(
        default=None,
        description="Dataset, experiment, or git run provenance",
    )

    @model_validator(mode="after")
    def validate_predicted_state_invariants(self) -> WorldState:
        """Enforce lineage invariants for predicted states."""
        if self.state_kind == StateKind.PREDICTED:
            if not self.lineage.parent_prediction_id:
                raise SchemaMismatchError(
                    "Predicted WorldState must have a non-empty lineage.parent_prediction_id",
                    details={"state_id": self.state_id, "state_kind": self.state_kind},
                )
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent in-place mutation of identity and semantic provenance fields."""
        if hasattr(self, "__pydantic_fields_set__") and name in (
            "state_id",
            "state_kind",
            "timestamp",
            "lineage",
        ):
            if name == "state_kind":
                current_kind = getattr(self, "state_kind", None)
                if current_kind == StateKind.PREDICTED:
                    raise InvalidInputError(
                        f"Predicted WorldState (id={getattr(self, 'state_id', 'unknown')}) cannot be "
                        f"mutated in-place to '{value}'. Real facts require separate observation/estimation states."
                    )
            raise InvalidInputError(
                f"WorldState semantic field '{name}' is immutable and cannot be reassigned in-place."
            )
        super().__setattr__(name, value)

    def assert_cannot_promote_to_fact(self, target_kind: StateKind) -> None:
        """Enforce that a predicted state cannot be automatically promoted to fact."""
        if self.state_kind == StateKind.PREDICTED and target_kind in (
            StateKind.OBSERVED,
            StateKind.ESTIMATED,
            StateKind.SIMULATED_GROUND_TRUTH,
        ):
            raise InvalidInputError(
                f"Predicted WorldState (id={self.state_id}) cannot be automatically promoted "
                f"to factual state kind '{target_kind}'. Real facts require separate observations."
            )
