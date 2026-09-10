"""Tests for PredictionRequest, PredictedTrajectory, and WorldPrediction."""

from datetime import datetime, timedelta, timezone

import pytest

from world_model.contracts.errors import (
    InvalidInputError,
    OutputModeUnsupportedError,
    SchemaMismatchError,
)
from world_model.contracts.prediction import (
    OutputMode,
    PredictionOptions,
    PredictionRequest,
    WorldPrediction,
)
from world_model.contracts.task_spec import TransitionTaskSpec
from world_model.contracts.trajectory import PredictedTrajectory
from world_model.contracts.world_context import WorldContext
from world_model.contracts.world_state import (
    StateKind,
    StateLineage,
    WorldComponents,
    WorldState,
)


def _make_predicted_state(t: datetime, state_id: str = "ws_p0", pred_id: str = "pred_01") -> WorldState:
    return WorldState(
        state_id=state_id,
        state_kind=StateKind.PREDICTED,
        timestamp=t,
        components=WorldComponents(),
        lineage=StateLineage(parent_prediction_id=pred_id),
    )


def test_prediction_request_exactly_one_state_source(base_utc_time: datetime) -> None:
    """PredictionRequest must enforce state_history_ref XOR state_history."""
    target_times = [base_utc_time + timedelta(seconds=1)]

    # Neither provided -> fail
    with pytest.raises(InvalidInputError, match="state_history_ref, state_history"):
        PredictionRequest(
            request_id="req_01",
            context_ref="wc_001",
            task_spec_ref="task_001",
            target_times=target_times,
        )

    # Both provided -> fail
    with pytest.raises(InvalidInputError, match="state_history_ref, state_history"):
        PredictionRequest(
            request_id="req_01",
            state_history_ref="hdfs://data/state_hist",
            state_history=[_make_predicted_state(base_utc_time)],
            context_ref="wc_001",
            task_spec_ref="task_001",
            target_times=target_times,
        )


def test_prediction_request_exactly_one_context_source(base_utc_time: datetime) -> None:
    """PredictionRequest must enforce context_ref XOR context."""
    target_times = [base_utc_time + timedelta(seconds=1)]

    # Neither provided -> fail
    with pytest.raises(InvalidInputError, match="context_ref, context"):
        PredictionRequest(
            request_id="req_02",
            state_history_ref="sh_001",
            task_spec_ref="task_001",
            target_times=target_times,
        )

    # Both provided -> fail
    with pytest.raises(InvalidInputError, match="context_ref, context"):
        PredictionRequest(
            request_id="req_02",
            state_history_ref="sh_001",
            context_ref="wc_001",
            context=WorldContext(context_id="wc_inline"),
            task_spec_ref="task_001",
            target_times=target_times,
        )


def test_prediction_request_exactly_one_task_spec_source(
    base_utc_time: datetime, sample_task_spec: TransitionTaskSpec
) -> None:
    """PredictionRequest must enforce task_spec_ref XOR task_spec."""
    target_times = [base_utc_time + timedelta(seconds=1)]

    # Neither provided -> fail
    with pytest.raises(InvalidInputError, match="task_spec_ref, task_spec"):
        PredictionRequest(
            request_id="req_03",
            state_history_ref="sh_001",
            context_ref="wc_001",
            target_times=target_times,
        )

    # Both provided -> fail
    with pytest.raises(InvalidInputError, match="task_spec_ref, task_spec"):
        PredictionRequest(
            request_id="req_03",
            state_history_ref="sh_001",
            context_ref="wc_001",
            task_spec_ref="task_001",
            task_spec=sample_task_spec,
            target_times=target_times,
        )


def test_deterministic_prediction_requires_single_trajectory() -> None:
    """Deterministic mode requires num_trajectories=1."""
    # Valid deterministic options
    opts = PredictionOptions(output_mode=OutputMode.DETERMINISTIC, num_trajectories=1)
    assert opts.num_trajectories == 1

    # num_trajectories > 1 with deterministic -> fail
    with pytest.raises(OutputModeUnsupportedError):
        PredictionOptions(output_mode=OutputMode.DETERMINISTIC, num_trajectories=5)


def test_predicted_trajectory_requires_predicted_states(base_utc_time: datetime) -> None:
    """All states inside PredictedTrajectory must have state_kind='predicted'."""
    pred_state = _make_predicted_state(base_utc_time)
    gt_state = WorldState(
        state_id="ws_gt_01",
        state_kind=StateKind.SIMULATED_GROUND_TRUTH,
        timestamp=base_utc_time + timedelta(seconds=1),
        components=WorldComponents(),
    )

    # Valid: all predicted
    traj = PredictedTrajectory(
        trajectory_id="traj_pred_01",
        states=[pred_state],
    )
    assert len(traj.states) == 1

    # Invalid: contains non-predicted state
    with pytest.raises(SchemaMismatchError, match="must strictly have state_kind='predicted'"):
        PredictedTrajectory(
            trajectory_id="traj_pred_bad",
            states=[pred_state, gt_state],
        )


def test_world_prediction_roundtrip(base_utc_time: datetime) -> None:
    """WorldPrediction serialization roundtrip."""
    pred_state = _make_predicted_state(base_utc_time + timedelta(seconds=1), pred_id="pred_001")
    traj = PredictedTrajectory(
        trajectory_id="traj_001",
        states=[pred_state],
    )

    pred = WorldPrediction(
        prediction_id="pred_001",
        request_id="req_001",
        generated_at=base_utc_time,
        base_state_ids=["ws_base_01"],
        trajectories=[traj],
        model_manifest_ref="model_swvt_v1",
        context_ref="wc_001",
        task_spec_ref="task_001",
        status="success",
    )

    d = pred.to_dict()
    assert d["contract_type"] == "WorldPrediction"
    assert d["prediction_id"] == "pred_001"
    assert len(d["trajectories"]) == 1

    restored = WorldPrediction.from_dict(d)
    assert restored.prediction_id == pred.prediction_id
    assert len(restored.trajectories) == 1
    assert restored.trajectories[0].states[0].state_kind == StateKind.PREDICTED
