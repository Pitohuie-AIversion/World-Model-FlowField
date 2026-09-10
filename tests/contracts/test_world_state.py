"""Tests for WorldState, FluidState, and component contracts."""

from datetime import datetime, timezone

import pytest

from world_model.contracts.errors import (
    InvalidInputError,
    SchemaMismatchError,
    TimeRangeInvalidError,
)
from world_model.contracts.world_state import (
    FieldRef,
    StateKind,
    StateLineage,
    WorldState,
)


def test_world_state_roundtrip(sample_ground_truth_world_state: WorldState) -> None:
    """Validate full serialization roundtrip via dict and JSON preserves all fields."""
    d = sample_ground_truth_world_state.to_dict()
    assert d["contract_type"] == "WorldState"
    assert d["schema_version"] == "1.8.0"
    assert d["state_id"] == "ws_sim_001"
    assert d["state_kind"] == "simulated_ground_truth"

    # Roundtrip from dict
    restored = WorldState.from_dict(d)
    assert restored.state_id == sample_ground_truth_world_state.state_id
    assert restored.state_kind == sample_ground_truth_world_state.state_kind
    assert restored.timestamp == sample_ground_truth_world_state.timestamp

    # Roundtrip from JSON string
    json_str = sample_ground_truth_world_state.to_json()
    restored_json = WorldState.from_json(json_str)
    assert restored_json.state_id == sample_ground_truth_world_state.state_id
    assert restored_json.state_kind == StateKind.SIMULATED_GROUND_TRUTH


def test_world_state_rejects_naive_timestamp(sample_fluid_state) -> None:
    """Naive datetime timestamps must be rejected immediately."""
    naive_time = datetime(2026, 9, 10, 12, 0, 0)  # No tzinfo!

    with pytest.raises((TimeRangeInvalidError, InvalidInputError)):
        WorldState(
            state_id="ws_invalid_time",
            state_kind=StateKind.OBSERVED,
            timestamp=naive_time,
        )


test_world_state_requires_timezone_aware_timestamp = test_world_state_rejects_naive_timestamp


def test_fluid_state_does_not_require_fixed_channels() -> None:
    """FluidState accepts arbitrary canonical channels, not hardcoded to four."""
    from world_model.contracts.world_state import FluidState

    # 2-channel fluid state (e.g. 2D velocity only)
    fluid_2ch = FluidState(
        fields={
            "velocity.x": FieldRef(inline_value=[1.0, 2.0]),
            "velocity.y": FieldRef(inline_value=[0.5, -0.5]),
        },
        spatial_ref="grid_2d",
    )
    assert len(fluid_2ch.fields) == 2
    assert "velocity.x" in fluid_2ch.fields
    assert "pressure" not in fluid_2ch.fields

    # Single-channel scalar state (e.g. pressure or tracer only)
    fluid_1ch = FluidState(
        fields={"pressure": FieldRef(inline_value=[101.3])},
        spatial_ref="grid_scalar",
    )
    assert len(fluid_1ch.fields) == 1

    # 5-channel state (e.g. 3D velocity + pressure + tracer)
    fluid_5ch = FluidState(
        fields={
            "velocity.x": FieldRef(inline_value=[1.0]),
            "velocity.y": FieldRef(inline_value=[2.0]),
            "velocity.z": FieldRef(inline_value=[3.0]),
            "pressure": FieldRef(inline_value=[100.0]),
            "tracer": FieldRef(inline_value=[0.1]),
        },
        spatial_ref="grid_3d",
    )
    assert len(fluid_5ch.fields) == 5


def test_predicted_state_remains_predicted(sample_predicted_world_state: WorldState) -> None:
    """Predicted states must not be automatically promoted to fact when time arrives."""
    assert sample_predicted_world_state.state_kind == StateKind.PREDICTED

    # Attempting promotion to observed/estimated fact must raise an error
    with pytest.raises(InvalidInputError, match="cannot be automatically promoted"):
        sample_predicted_world_state.assert_cannot_promote_to_fact(StateKind.OBSERVED)

    with pytest.raises(InvalidInputError, match="cannot be automatically promoted"):
        sample_predicted_world_state.assert_cannot_promote_to_fact(StateKind.ESTIMATED)


def test_world_state_lineage_roundtrip(sample_predicted_world_state: WorldState) -> None:
    """Test lineage and parent_prediction_id tracking and preservation."""
    assert sample_predicted_world_state.lineage.parent_prediction_id == "pred_test_001"

    data = sample_predicted_world_state.to_dict()
    restored = WorldState.from_dict(data)
    assert restored.lineage.parent_prediction_id == "pred_test_001"


def test_predicted_state_requires_parent_prediction_id(base_utc_time: datetime) -> None:
    """A predicted state must have lineage.parent_prediction_id."""
    with pytest.raises((SchemaMismatchError, InvalidInputError)):
        WorldState(
            state_id="ws_pred_no_lineage",
            state_kind=StateKind.PREDICTED,
            timestamp=base_utc_time,
            lineage=StateLineage(parent_prediction_id=None),
        )


def test_field_ref_exactly_one_storage() -> None:
    """FieldRef must provide exactly one of (inline_value, tensor_ref, artifact_ref)."""
    # Valid inline
    ref1 = FieldRef(inline_value=[1.0, 2.0])
    assert ref1.inline_value == [1.0, 2.0]

    # Valid tensor
    ref2 = FieldRef(tensor_ref="tensor_key_01")
    assert ref2.tensor_ref == "tensor_key_01"

    # Valid artifact
    ref3 = FieldRef(artifact_ref="hdfs://data/field.h5")
    assert ref3.artifact_ref == "hdfs://data/field.h5"

    # None provided -> error
    with pytest.raises(InvalidInputError):
        FieldRef()

    # Multiple provided -> error
    with pytest.raises(InvalidInputError):
        FieldRef(inline_value=[1.0], tensor_ref="tensor_key_01")
