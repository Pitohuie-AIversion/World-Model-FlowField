"""Tests for contract serialization, schema versioning, and format integrity."""

from datetime import datetime, timezone

import pytest

from world_model.contracts.action import ActionSequence
from world_model.contracts.benchmark import BenchmarkProtocol
from world_model.contracts.errors import SchemaMismatchError
from world_model.contracts.manifests import DatasetManifest, ModelManifest
from world_model.contracts.prediction import PredictionRequest, WorldPrediction
from world_model.contracts.task_spec import TransitionTaskSpec
from world_model.contracts.transition import TransitionSample
from world_model.contracts.world_context import WorldContext
from world_model.contracts.world_state import StateKind, WorldState


def test_schema_version_is_serialized(sample_ground_truth_world_state: WorldState) -> None:
    """Every top-level contract must serialize contract_type and schema_version."""
    d = sample_ground_truth_world_state.to_dict()
    assert "contract_type" in d
    assert d["contract_type"] == "WorldState"
    assert "schema_version" in d
    assert d["schema_version"] == "1.8.0"


def test_mismatched_contract_type_raises_error() -> None:
    """Deserializing a payload with a mismatched contract_type must fail closed."""
    with pytest.raises(SchemaMismatchError, match="Contract type mismatch"):
        WorldState.from_dict({
            "contract_type": "WrongContractType",
            "schema_version": "1.8.0",
            "state_id": "ws_01",
            "state_kind": "observed",
            "timestamp": "2026-09-10T12:00:00Z",
        })


def test_timezone_preservation_in_roundtrip(sample_ground_truth_world_state: WorldState) -> None:
    """Timezone information must survive dict and JSON roundtrip without becoming naive."""
    json_str = sample_ground_truth_world_state.to_json()
    restored = WorldState.from_json(json_str)

    assert restored.timestamp.tzinfo is not None
    assert restored.timestamp == sample_ground_truth_world_state.timestamp


def test_enum_preservation_in_roundtrip(sample_ground_truth_world_state: WorldState) -> None:
    """Enum values must parse back to Enum members."""
    d = sample_ground_truth_world_state.to_dict()
    restored = WorldState.from_dict(d)
    assert isinstance(restored.state_kind, StateKind)
    assert restored.state_kind == StateKind.SIMULATED_GROUND_TRUTH
