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


def test_world_state_requires_timezone_aware_timestamp(sample_fluid_state) -> None:
    """Naive datetime timestamps must be rejected immediately."""
    naive_time = datetime(2026, 9, 10, 12, 0, 0)  # No tzinfo!

    with pytest.raises((TimeRangeInvalidError, InvalidInputError)):
        WorldState(
            state_id="ws_invalid_time",
            state_kind=StateKind.OBSERVED,
            timestamp=naive_time,
        )


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
