"""Tests for contract serialization, schema versioning, and format integrity."""

from datetime import datetime, timedelta, timezone

import pytest

from world_model.contracts.action import ActionSequence
from world_model.contracts.benchmark import BenchmarkProtocol
from world_model.contracts.common import SCHEMA_VERSION_DEFAULT
from world_model.contracts.errors import InvalidInputError, SchemaMismatchError
from world_model.contracts.manifests import (
    DatasetManifest,
    ModelCapabilities,
    ModelManifest,
)
from world_model.contracts.prediction import PredictionRequest, WorldPrediction
from world_model.contracts.task_spec import (
    HistorySelection,
    TargetTimePolicy,
    TransitionTaskSpec,
)
from world_model.contracts.trajectory import PredictedTrajectory
from world_model.contracts.transition import TransitionSample
from world_model.contracts.world_context import WorldContext
from world_model.contracts.world_state import (
    StateKind,
    StateLineage,
    WorldComponents,
    WorldState,
)


def test_schema_version_is_serialized(sample_ground_truth_world_state: WorldState) -> None:
    """Every top-level contract must serialize contract_type and schema_version."""
    d = sample_ground_truth_world_state.to_dict()
    assert "contract_type" in d
    assert d["contract_type"] == "WorldState"
    assert "schema_version" in d
    assert d["schema_version"] == "1.8.0"


def test_schema_version_accepts_current_version(sample_ground_truth_world_state: WorldState) -> None:
    """Explicitly providing the current schema_version ('1.8.0') is accepted."""
    d = sample_ground_truth_world_state.to_dict()
    d["schema_version"] = "1.8.0"
    restored = WorldState.from_dict(d)
    assert restored.schema_version == "1.8.0"


def test_schema_version_rejects_unsupported_version(sample_ground_truth_world_state: WorldState) -> None:
    """Supplying an incompatible schema_version (e.g. 999.0.0) must fail closed with SchemaMismatchError."""
    d = sample_ground_truth_world_state.to_dict()
    d["schema_version"] = "999.0.0"
    with pytest.raises(SchemaMismatchError, match="Schema version mismatch"):
        WorldState.from_dict(d)


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


def test_json_rejects_nan(sample_ground_truth_world_state: WorldState) -> None:
    """NaN values in contract payloads must be rejected upon serialization or deserialization."""
    d = sample_ground_truth_world_state.to_dict()
    d["provenance"] = {"residual": float("nan")}
    with pytest.raises(InvalidInputError, match="NaN/Infinity"):
        WorldState.from_dict(d)

    bad_json = '{"contract_type": "WorldState", "schema_version": "1.8.0", "state_id": "ws_nan", "state_kind": "observed", "timestamp": "2026-09-10T12:00:00Z", "components": {}, "lineage": {}, "provenance": {"val": NaN}}'
    with pytest.raises(InvalidInputError, match="NaN/Infinity"):
        WorldState.from_json(bad_json)


def test_json_rejects_positive_inf(sample_ground_truth_world_state: WorldState) -> None:
    """Positive infinity float values must be rejected fail-closed."""
    d = sample_ground_truth_world_state.to_dict()
    d["provenance"] = {"divergence": float("inf")}
    with pytest.raises(InvalidInputError, match="NaN/Infinity"):
        WorldState.from_dict(d)

    bad_json = '{"contract_type": "WorldState", "schema_version": "1.8.0", "state_id": "ws_inf", "state_kind": "observed", "timestamp": "2026-09-10T12:00:00Z", "components": {}, "lineage": {}, "provenance": {"val": Infinity}}'
    with pytest.raises(InvalidInputError, match="NaN/Infinity"):
        WorldState.from_json(bad_json)


def test_json_rejects_negative_inf(sample_ground_truth_world_state: WorldState) -> None:
    """Negative infinity float values must be rejected fail-closed."""
    d = sample_ground_truth_world_state.to_dict()
    d["provenance"] = {"bound": float("-inf")}
    with pytest.raises(InvalidInputError, match="NaN/Infinity"):
        WorldState.from_dict(d)

    bad_json = '{"contract_type": "WorldState", "schema_version": "1.8.0", "state_id": "ws_ninf", "state_kind": "observed", "timestamp": "2026-09-10T12:00:00Z", "components": {}, "lineage": {}, "provenance": {"val": -Infinity}}'
    with pytest.raises(InvalidInputError, match="NaN/Infinity"):
        WorldState.from_json(bad_json)


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


def test_all_top_level_contracts_contain_schema_version(
    base_utc_time: datetime,
    sample_ground_truth_world_state: WorldState,
    sample_world_context: WorldContext,
    sample_task_spec: TransitionTaskSpec,
    sample_dataset_manifest: DatasetManifest,
) -> None:
    """Every top-level contract that inherits ContractBase must
    serialize both contract_type and schema_version into its dict output."""

    # WorldState
    ws_d = sample_ground_truth_world_state.to_dict()
    assert ws_d["contract_type"] == "WorldState"
    assert ws_d["schema_version"] == SCHEMA_VERSION_DEFAULT

    # WorldContext
    wc_d = sample_world_context.to_dict()
    assert wc_d["contract_type"] == "WorldContext"
    assert wc_d["schema_version"] == SCHEMA_VERSION_DEFAULT

    # TransitionTaskSpec
    ts_d = sample_task_spec.to_dict()
    assert ts_d["contract_type"] == "TransitionTaskSpec"
    assert ts_d["schema_version"] == SCHEMA_VERSION_DEFAULT

    # DatasetManifest
    dm_d = sample_dataset_manifest.to_dict()
    assert dm_d["contract_type"] == "DatasetManifest"
    assert dm_d["schema_version"] == SCHEMA_VERSION_DEFAULT

    # BenchmarkProtocol
    bp = BenchmarkProtocol(
        benchmark_id="bench_schema",
        task_spec_ref="task_001",
        dataset_manifest_ref="ds_001",
        rollout_horizons=[1],
    )
    bp_d = bp.to_dict()
    assert bp_d["contract_type"] == "BenchmarkProtocol"
    assert bp_d["schema_version"] == SCHEMA_VERSION_DEFAULT

    # ModelManifest
    caps = ModelCapabilities(
        input_components=["environment.fluid_state"],
        output_components=["environment.fluid_state"],
    )
    mm = ModelManifest(
        model_id="model_schema_test",
        version="0.0.1",
        backend_type="latent_world_model",
        capabilities=caps,
    )
    mm_d = mm.to_dict()
    assert mm_d["contract_type"] == "ModelManifest"
    assert mm_d["schema_version"] == SCHEMA_VERSION_DEFAULT

    # TransitionSample
    t_target = base_utc_time + timedelta(seconds=5)
    pred_state = WorldState(
        state_id="ws_target",
        state_kind=StateKind.SIMULATED_GROUND_TRUTH,
        timestamp=t_target,
        components=WorldComponents(),
    )
    ts = TransitionSample(
        sample_id="sample_schema",
        task_spec_ref="task_001",
        state_history=[sample_ground_truth_world_state],
        context=sample_world_context,
        target_states=[pred_state],
        trajectory_id="traj_schema",
    )
    ts_dict = ts.to_dict()
    assert ts_dict["contract_type"] == "TransitionSample"
    assert ts_dict["schema_version"] == SCHEMA_VERSION_DEFAULT

    # WorldPrediction
    predicted_st = WorldState(
        state_id="ws_pred_schema",
        state_kind=StateKind.PREDICTED,
        timestamp=t_target,
        components=WorldComponents(),
        lineage=StateLineage(parent_prediction_id="pred_schema"),
    )
    traj = PredictedTrajectory(trajectory_id="traj_schema", states=[predicted_st])
    wp = WorldPrediction(
        prediction_id="pred_schema",
        request_id="req_schema",
        generated_at=base_utc_time,
        trajectories=[traj],
    )
    wp_d = wp.to_dict()
    assert wp_d["contract_type"] == "WorldPrediction"
    assert wp_d["schema_version"] == SCHEMA_VERSION_DEFAULT
