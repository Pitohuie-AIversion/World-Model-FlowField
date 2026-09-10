"""DatasetManifest and ModelManifest contracts.

DatasetManifest describes dataset files, canonical field mappings, and splits.
ModelCapabilities and ModelManifest define model capabilities and compatibility requirements.
Grid shapes (Ny, Nx) and channel orders are never fixed globally.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field

from world_model.contracts.common import ContractBase
from world_model.contracts.world_state import StateKind


class DatasetManifest(ContractBase):
    """Manifest describing a structured dataset for world model training and benchmarking."""

    contract_type: str = Field(default="DatasetManifest", description="Contract type identifier")
    dataset_id: str = Field(..., description="Unique dataset identifier, e.g. the_well_shear_flow")
    revision: str = Field(..., description="Version/git/HDF5 revision tag")
    source: str = Field(..., description="Origin source or citation (e.g. The Well)")
    license: str = Field(..., description="License identifier (e.g. CC-BY-4.0, MIT)")
    file_list: list[dict[str, Any]] = Field(
        ...,
        description="List of files with path and cryptographic checksums",
    )
    field_mapping: Mapping[str, str] = Field(
        ...,
        description="Dataset source field name -> canonical field name (e.g. 'u' -> 'velocity.x')",
    )
    split_trajectory_ids: Mapping[str, list[str]] = Field(
        ...,
        description="Splits mapping, e.g. {'train': [...], 'valid': [...], 'test': [...]}. Split before window slicing.",
    )
    time_metadata: dict[str, Any] = Field(
        ...,
        description="Time sampling metadata: dt, time_unit, num_steps_per_trajectory",
    )
    spatial_metadata: dict[str, Any] = Field(
        ...,
        description="Spatial domain metadata: axes, resolution, boundary conditions. Never globally hardcoded.",
    )
    training_statistics_ref: Optional[str] = Field(
        default=None,
        description="Reference to normalizer statistics computed exclusively on training split",
    )


class ModelCapabilities(BaseModel):
    """Declarative capability contract of a WorldModelBackend."""

    input_components: list[str] = Field(
        default_factory=lambda: ["environment.fluid_state"],
        description="Input components accepted by model",
    )
    output_components: list[str] = Field(
        default_factory=lambda: ["environment.fluid_state"],
        description="Output components produced by model",
    )
    supported_state_kinds: list[StateKind] = Field(
        default_factory=lambda: [
            StateKind.SIMULATED_GROUND_TRUTH,
            StateKind.OBSERVED,
            StateKind.ESTIMATED,
        ],
        description="Permissible conditioning state kinds",
    )
    action_conditioning: bool = Field(
        default=False,
        description="Whether backend supports action conditioning (Phase 1: False)",
    )
    output_modes: list[str] = Field(
        default_factory=lambda: ["deterministic"],
        description="Supported output modes: deterministic | ensemble",
    )
    supported_rollout_modes: list[str] = Field(
        default_factory=lambda: ["one_step", "lead_time"],
        description="Rollout modes: one_step | lead_time | direct_multi_step | autoregressive",
    )
    history_policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Minimum and maximum history window constraints",
    )
    target_time_policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Supported target horizons and step policies",
    )
    supported_grid_types: list[str] = Field(
        default_factory=lambda: ["uniform_cartesian"],
        description="Supported grid discretizations",
    )
    periodic_axes_policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Support and requirements for periodic axes (e.g. SWVT requires periodic padding)",
    )
    padding_policy: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain padding requirements (e.g. divisible by patch/window size)",
    )
    supported_spatial_domain: Optional[dict[str, Any]] = Field(
        default=None,
        description="Boundary of validated physical parameter ranges",
    )

    model_config = ConfigDict(extra="forbid")


class ModelManifest(ContractBase):
    """Manifest accompanying a packaged model checkpoint."""

    contract_type: str = Field(default="ModelManifest", description="Contract type identifier")
    model_id: str = Field(..., description="Unique model architecture/instance identifier")
    version: str = Field(..., description="Model artifact semantic version")
    backend_type: str = Field(
        ...,
        description="Backend architecture category: latent_world_model | direct_transition",
    )
    compatible_schema_versions: list[str] = Field(
        default_factory=lambda: ["1.8.0"],
        description="List of compatible World Model schema versions",
    )
    capabilities: ModelCapabilities = Field(
        ...,
        description="Declarative capability specification",
    )
    required_context: list[str] = Field(
        default_factory=list,
        description="Required physical context keys (e.g. physics_parameters.Re)",
    )
    tensor_spec_ref: Optional[str] = Field(
        default=None,
        description="Reference to TensorSpec defining tensor layout and channel ordering",
    )
    validated_domain_ref: Optional[str] = Field(
        default=None,
        description="Reference to domain validation bounds",
    )
    uncertainty_mode: Optional[str] = Field(
        default=None,
        description="Uncertainty quantification mode if probabilistic backend",
    )
    checkpoint_hash: Optional[str] = Field(
        default=None,
        description="Cryptographic SHA256 checksum of model weights",
    )
    normalizer_ref: Optional[str] = Field(
        default=None,
        description="Reference to normalizer fitted on training data",
    )
    code_commit: Optional[str] = Field(
        default=None,
        description="Git commit hash corresponding to model code",
    )
