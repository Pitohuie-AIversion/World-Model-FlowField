"""WorldContext domain contract.

Owns physical evolution conditions:
- Static conditions: geometry, physics parameters (Re, Sc), static boundaries, spatial domains (GridSpec)
- Temporal conditions: time-varying forcing and boundaries (ConditionSeries)

WorldContext strictly DOES NOT own:
- state timestamp
- history_times
- target_times
- dataset split / revision
- sampling seed
- num_trajectories
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from world_model.contracts.common import ContractBase, generate_id
from world_model.contracts.condition import ConditionSeries
from world_model.contracts.errors import InvalidInputError

FORBIDDEN_CONTEXT_KEYS = {
    "state_timestamp",
    "timestamp",
    "history_times",
    "target_times",
    "dataset_split",
    "split",
    "dataset_revision",
    "sampling_seed",
    "num_trajectories",
}


class GridSpec(BaseModel):
    """Component-level spatial domain and discretization specification.

    Grid dimensions (shape) are dataset/domain specific and not fixed globally.
    """

    grid_id: str = Field(..., description="Unique spatial domain identifier")
    coordinate_frame: str = Field(default="domain", description="Reference frame")
    axes: list[str] = Field(
        default_factory=lambda: ["y", "x"],
        description="Names and ordering of spatial axes, e.g. ['y', 'x'] or ['z', 'y', 'x']",
    )
    shape: Optional[list[int]] = Field(
        default=None,
        description="Discretization grid shape [Ny, Nx] or [Nz, Ny, Nx]. Not globally fixed.",
    )
    grid_type: str = Field(
        default="uniform_cartesian",
        description="uniform_cartesian | curvilinear | unstructured | spherical",
    )
    periodic_axes: list[str] = Field(
        default_factory=list,
        description="List of axes with periodic boundary conditions, e.g. ['x', 'y']",
    )
    coordinates_ref: Optional[str] = Field(
        default=None,
        description="Reference to coordinate arrays if non-uniform",
    )
    geometry_ref: Optional[str] = Field(
        default=None,
        description="Reference to domain geometry or CAD definition",
    )

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_grid_spec_invariants(self) -> GridSpec:
        if len(self.axes) != len(set(self.axes)):
            raise InvalidInputError(
                f"GridSpec axes must not contain duplicate names: {self.axes}"
            )

        if self.shape is not None:
            if any(dim <= 0 for dim in self.shape):
                raise InvalidInputError(
                    f"GridSpec shape dimensions must be strictly positive (> 0), got {self.shape}."
                )
            if len(self.shape) != len(self.axes):
                raise InvalidInputError(
                    f"GridSpec len(shape) ({len(self.shape)}) must match len(axes) ({len(self.axes)})."
                )

        if len(self.periodic_axes) != len(set(self.periodic_axes)):
            raise InvalidInputError(
                f"GridSpec periodic_axes must not contain duplicate names: {self.periodic_axes}"
            )

        axes_set = set(self.axes)
        for p_axis in self.periodic_axes:
            if p_axis not in axes_set:
                raise InvalidInputError(
                    f"GridSpec periodic axis '{p_axis}' is not declared in axes {self.axes}."
                )

        return self


class StaticConditions(BaseModel):
    """Time-invariant physical context driving dynamics."""

    spatial_domains: Mapping[str, GridSpec] = Field(
        default_factory=dict,
        description="spatial_ref -> GridSpec mapping for components",
    )
    geometry: Optional[dict[str, Any]] = Field(
        default=None,
        description="Domain bounds, bathymetry, or obstacle geometry",
    )
    physics_parameters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Dimensionless numbers (Re, Sc, Ro) or physical constants",
    )
    static_boundary_conditions: Optional[dict[str, Any]] = Field(
        default=None,
        description="Static Dirichlet / Neumann / Robin boundary conditions",
    )

    model_config = ConfigDict(extra="forbid")


class WorldContext(ContractBase):
    """Top-level physical environment context contract.

    Guarantees strict separation between static and temporal condition keys.
    """

    contract_type: str = Field(default="WorldContext", description="Contract type identifier")
    context_id: str = Field(
        default_factory=lambda: generate_id("wc"),
        description="Unique context identifier",
    )
    static_conditions: StaticConditions = Field(
        default_factory=StaticConditions,
        description="Invariant physical domain and boundary parameters",
    )
    temporal_conditions: Mapping[str, ConditionSeries] = Field(
        default_factory=dict,
        description="Canonical condition name -> ConditionSeries mapping",
    )
    scenario_id: Optional[str] = Field(
        default=None,
        description="Simulation scenario or real mission segment tag",
    )

    @model_validator(mode="before")
    @classmethod
    def check_forbidden_fields(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for k in FORBIDDEN_CONTEXT_KEYS:
                if k in values:
                    raise InvalidInputError(
                        f"WorldContext must not contain '{k}'. Physical evolution context only.",
                        details={"forbidden_key": k},
                    )
        return values

    @model_validator(mode="after")
    def validate_namespace_isolation(self) -> WorldContext:
        """Enforce that no condition name exists in both static and temporal conditions."""
        static_keys = set()
        if self.static_conditions.physics_parameters:
            static_keys.update(self.static_conditions.physics_parameters.keys())
        if self.static_conditions.static_boundary_conditions:
            static_keys.update(self.static_conditions.static_boundary_conditions.keys())

        temporal_keys = set(self.temporal_conditions.keys())
        overlap = static_keys.intersection(temporal_keys)
        if overlap:
            raise InvalidInputError(
                f"Condition namespace collision: key(s) {overlap} cannot appear in both "
                f"static_conditions and temporal_conditions.",
                details={"overlap_keys": list(overlap)},
            )
        return self
