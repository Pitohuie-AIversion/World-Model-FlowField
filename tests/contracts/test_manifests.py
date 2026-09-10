import pytest

from world_model.contracts.errors import InvalidInputError
from world_model.contracts.manifests import (
    DatasetManifest,
    ModelCapabilities,
    ModelManifest,
)
from world_model.contracts.world_state import StateKind


def test_dataset_manifest_has_no_fixed_grid_requirement(
    sample_dataset_manifest: DatasetManifest,
) -> None:
    """DatasetManifest must accommodate arbitrary grid resolutions and shapes."""
    d = sample_dataset_manifest.to_dict()
    d["dataset_id"] = "the_well_rayleigh_benard_3d"
    d["spatial_metadata"] = {
        "axes": ["z", "y", "x"],
        "shape": [64, 512, 1024],
        "grid_type": "staggered_cartesian",
    }
    custom_manifest = DatasetManifest.from_dict(d)
    assert custom_manifest.spatial_metadata["shape"] == [64, 512, 1024]

    restored = DatasetManifest.from_dict(custom_manifest.to_dict())
    assert restored.spatial_metadata["shape"] == [64, 512, 1024]


def test_dataset_manifest_allows_variable_grid_shape(
    sample_dataset_manifest: DatasetManifest,
) -> None:
    """DatasetManifest accommodates arbitrary spatial domain shapes via proper dictionary validation."""
    d = sample_dataset_manifest.to_dict()
    d["spatial_metadata"] = {"axes": ["x"], "shape": [42]}
    tiny_manifest = DatasetManifest.from_dict(d)
    assert tiny_manifest.spatial_metadata["shape"] == [42]


def test_dataset_splits_are_disjoint(sample_dataset_manifest: DatasetManifest) -> None:
    """Trajectory IDs across dataset splits must be strictly pairwise disjoint."""
    d = sample_dataset_manifest.to_dict()
    # Introduce an overlap: traj_001 in both train and test
    d["split_trajectory_ids"] = {
        "train": ["traj_001", "traj_002"],
        "valid": ["traj_003"],
        "test": ["traj_001", "traj_004"],
    }
    with pytest.raises(InvalidInputError, match="split collision: trajectory 'traj_001' appears in multiple splits"):
        DatasetManifest.from_dict(d)


def test_model_capabilities_roundtrip() -> None:
    """ModelCapabilities must round-trip through dict serialization."""
    caps = ModelCapabilities(
        input_components=["environment.fluid_state"],
        output_components=["environment.fluid_state"],
        supported_state_kinds=[StateKind.SIMULATED_GROUND_TRUTH, StateKind.OBSERVED],
        action_conditioning=False,
        output_modes=["deterministic"],
        supported_rollout_modes=["one_step", "lead_time", "autoregressive"],
        history_policy={"min_steps": 4, "max_steps": 16},
        target_time_policy={"max_horizon": 64},
        supported_grid_types=["uniform_cartesian"],
        periodic_axes_policy={"supported": ["x", "y"]},
        padding_policy={"divisible_by": 4},
    )

    d = caps.model_dump(mode="json")
    assert d["output_modes"] == ["deterministic"]
    assert d["supported_rollout_modes"] == ["one_step", "lead_time", "autoregressive"]
    assert d["history_policy"]["min_steps"] == 4
    assert d["padding_policy"]["divisible_by"] == 4

    restored = ModelCapabilities.model_validate(d)
    assert restored.action_conditioning is False
    assert restored.periodic_axes_policy == {"supported": ["x", "y"]}


def test_model_manifest_roundtrip() -> None:
    """ModelManifest roundtrip with ModelCapabilities."""
    caps = ModelCapabilities(
        input_components=["environment.fluid_state"],
        output_components=["environment.fluid_state"],
        supported_state_kinds=[StateKind.SIMULATED_GROUND_TRUTH, StateKind.OBSERVED],
        action_conditioning=False,
        output_modes=["deterministic"],
        supported_rollout_modes=["one_step", "lead_time"],
        periodic_axes_policy={"supported": ["x", "y"]},
    )
    manifest = ModelManifest(
        model_id="swvt_shear_flow_baseline",
        version="0.1.0",
        backend_type="latent_world_model",
        capabilities=caps,
        checkpoint_hash="sha256:1234567890abcdef",
    )

    d = manifest.to_dict()
    assert d["contract_type"] == "ModelManifest"
    assert d["capabilities"]["output_modes"] == ["deterministic"]

    restored = ModelManifest.from_dict(d)
    assert restored.model_id == "swvt_shear_flow_baseline"
    assert restored.capabilities.supported_rollout_modes == ["one_step", "lead_time"]
