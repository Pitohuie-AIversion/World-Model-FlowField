"""Tests for DatasetManifest, ModelCapabilities, and ModelManifest."""

from world_model.contracts.manifests import (
    DatasetManifest,
    ModelCapabilities,
    ModelManifest,
)
from world_model.contracts.world_state import StateKind


def test_dataset_manifest_does_not_require_fixed_grid_shape(
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
