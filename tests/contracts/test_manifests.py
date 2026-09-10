"""Tests for DatasetManifest, ModelCapabilities, and ModelManifest."""

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
    # Custom 3D resolution or atypical shape
    custom_manifest = sample_dataset_manifest.model_copy(
        update={
            "dataset_id": "the_well_rayleigh_benard_3d",
            "spatial_metadata": {
                "axes": ["z", "y", "x"],
                "shape": [64, 512, 1024],
                "grid_type": "staggered_cartesian",
            },
        }
    )
    d = custom_manifest.to_dict()
    assert d["spatial_metadata"]["shape"] == [64, 512, 1024]

    restored = DatasetManifest.from_dict(d)
    assert restored.spatial_metadata["shape"] == [64, 512, 1024]

    # Verify no specific shape is hardcoded as a requirement
    # Single dim grid should also be accepted
    tiny_manifest = sample_dataset_manifest.model_copy(
        update={
            "dataset_id": "custom_1d_dataset",
            "spatial_metadata": {"axes": ["x"], "shape": [42]},
        }
    )
    assert tiny_manifest.spatial_metadata["shape"] == [42]


# Backward compat alias
test_dataset_manifest_does_not_require_fixed_grid_shape = test_dataset_manifest_has_no_fixed_grid_requirement


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
