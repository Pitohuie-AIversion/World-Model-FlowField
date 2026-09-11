"""Tests for PredictionRequest, PredictedTrajectory, and WorldPrediction."""

from datetime import datetime, timedelta, timezone

import pytest

from world_model.contracts.errors import (
    InvalidInputError,
    OutputModeUnsupportedError,
    SchemaMismatchError,
    TimeRangeInvalidError,
)
from world_model.contracts.prediction import (
    OutputMode,
    PredictionOptions,
    PredictionRequest,
    PredictionValidity,
    UncertaintySummary,
    WorldPrediction,
)
from world_model.contracts.task_spec import (
    ActionPolicy,
    HistorySelection,
    TargetTimePolicy,
    TransitionTaskSpec,
)
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


def test_prediction_request_rejects_non_monotonic_inline_history(base_utc_time: datetime) -> None:
    """Non-monotonic inline state_history timestamps must be rejected."""
    t0 = base_utc_time
    t1 = t0 + timedelta(seconds=1)
    target = [t0 + timedelta(seconds=10)]

    with pytest.raises(TimeRangeInvalidError, match="strictly monotonic"):
        PredictionRequest(
            request_id="req_nonmono",
            state_history=[_make_predicted_state(t1, "s1", "p"), _make_predicted_state(t0, "s0", "p")],
            context_ref="wc_001",
            task_spec_ref="task_001",
            target_times=target,
        )


def test_prediction_request_rejects_target_before_history(base_utc_time: datetime) -> None:
    """target_times[0] before last state_history timestamp must fail causal check."""
    t0 = base_utc_time
    t1 = t0 + timedelta(seconds=5)
    target_past = [t0 + timedelta(seconds=2)]

    with pytest.raises(TimeRangeInvalidError, match="Causal violation"):
        PredictionRequest(
            request_id="req_causal_past",
            state_history=[_make_predicted_state(t0, "s0", "p"), _make_predicted_state(t1, "s1", "p")],
            context_ref="wc_001",
            task_spec_ref="task_001",
            target_times=target_past,
        )


def test_prediction_request_rejects_target_equal_to_last_history(base_utc_time: datetime) -> None:
    """target_times[0] equal to last state_history timestamp must fail causal check."""
    t0 = base_utc_time
    t1 = t0 + timedelta(seconds=5)
    target_equal = [t1]

    with pytest.raises(TimeRangeInvalidError, match="Causal violation"):
        PredictionRequest(
            request_id="req_causal_eq",
            state_history=[_make_predicted_state(t0, "s0", "p"), _make_predicted_state(t1, "s1", "p")],
            context_ref="wc_001",
            task_spec_ref="task_001",
            target_times=target_equal,
        )


def test_action_policy_none_rejects_actions(
    base_utc_time: datetime, sample_task_spec: TransitionTaskSpec
) -> None:
    """When ActionPolicy is NONE, providing actions sequence must fail closed."""
    from world_model.contracts.action import ActionSequence

    assert sample_task_spec.action_policy == ActionPolicy.NONE

    act_seq = ActionSequence(
        robot_id="auv_01",
        timestamps_or_intervals=[base_utc_time],
        action_type="thrust",
        controls_or_commands=[10.0],
    )

    with pytest.raises(InvalidInputError, match="ActionPolicy.NONE, but an actions sequence was provided"):
        PredictionRequest(
            request_id="req_act_none",
            state_history_ref="sh_001",
            context_ref="wc_001",
            task_spec=sample_task_spec,
            target_times=[base_utc_time + timedelta(seconds=1)],
            actions=act_seq,
        )


def test_action_policy_required_requires_actions(base_utc_time: datetime) -> None:
    """When ActionPolicy is REQUIRED, omitting actions sequence must fail closed."""
    ts_required = TransitionTaskSpec(
        task_spec_id="task_required_act",
        history_selection=HistorySelection(window_size=1),
        target_time_policy=TargetTimePolicy(horizon=1),
        action_policy=ActionPolicy.REQUIRED,
    )

    with pytest.raises(InvalidInputError, match="ActionPolicy.REQUIRED, but no actions sequence was provided"):
        PredictionRequest(
            request_id="req_act_missing",
            state_history_ref="sh_001",
            context_ref="wc_001",
            task_spec=ts_required,
            target_times=[base_utc_time + timedelta(seconds=1)],
            actions=None,
        )


def test_action_policy_optional_accepts_none_and_actions(base_utc_time: datetime) -> None:
    """When ActionPolicy is OPTIONAL, both actions=None and actions=ActionSequence are valid."""
    from world_model.contracts.action import ActionSequence

    ts_optional = TransitionTaskSpec(
        task_spec_id="task_optional_act",
        history_selection=HistorySelection(window_size=1),
        target_time_policy=TargetTimePolicy(horizon=1),
        action_policy=ActionPolicy.OPTIONAL,
    )

    # actions = None -> success
    req1 = PredictionRequest(
        request_id="req_act_opt1",
        state_history_ref="sh_001",
        context_ref="wc_001",
        task_spec=ts_optional,
        target_times=[base_utc_time + timedelta(seconds=1)],
        actions=None,
    )
    assert req1.actions is None

    # actions = ActionSequence -> success
    act_seq = ActionSequence(
        robot_id="auv_01",
        timestamps_or_intervals=[base_utc_time],
        action_type="thrust",
        controls_or_commands=[10.0],
    )
    req2 = PredictionRequest(
        request_id="req_act_opt2",
        state_history_ref="sh_001",
        context_ref="wc_001",
        task_spec=ts_optional,
        target_times=[base_utc_time + timedelta(seconds=1)],
        actions=act_seq,
    )
    assert req2.actions is not None


def test_predicted_trajectory_rejects_non_monotonic_times(base_utc_time: datetime) -> None:
    """PredictedTrajectory states must have strictly increasing timestamps."""
    t0 = base_utc_time
    t1 = t0 + timedelta(seconds=1)
    with pytest.raises(TimeRangeInvalidError, match="strictly monotonic"):
        PredictedTrajectory(
            trajectory_id="traj_bad_mono",
            states=[_make_predicted_state(t1, "s1"), _make_predicted_state(t0, "s0")],
        )


def test_predicted_trajectory_rejects_duplicate_times(base_utc_time: datetime) -> None:
    """PredictedTrajectory states with duplicate timestamps must be rejected."""
    t0 = base_utc_time
    with pytest.raises(TimeRangeInvalidError, match="strictly monotonic"):
        PredictedTrajectory(
            trajectory_id="traj_dup",
            states=[_make_predicted_state(t0, "s0"), _make_predicted_state(t0, "s1")],
        )


def test_world_prediction_requires_matching_parent_prediction_id(base_utc_time: datetime) -> None:
    """WorldPrediction requires all state lineage parent_prediction_id to equal its prediction_id."""
    state_mismatch = _make_predicted_state(base_utc_time + timedelta(seconds=1), pred_id="wrong_pred_id")
    traj = PredictedTrajectory(trajectory_id="t1", states=[state_mismatch])

    with pytest.raises(SchemaMismatchError, match="parent_prediction_id 'wrong_pred_id' .* does not match"):
        WorldPrediction(
            prediction_id="pred_authoritative",
            request_id="req_001",
            trajectories=[traj],
        )


def test_prediction_states_match_requested_target_times(base_utc_time: datetime) -> None:
    """Predicted state timestamps must match requested target_times point-by-point."""
    from world_model.contracts.validation import validate_prediction_against_request

    t_req = [base_utc_time + timedelta(seconds=1), base_utc_time + timedelta(seconds=2)]
    req = PredictionRequest(
        request_id="req_align",
        state_history_ref="sh_01",
        context_ref="wc_01",
        task_spec_ref="ts_01",
        target_times=t_req,
    )

    wrong_state = _make_predicted_state(base_utc_time + timedelta(seconds=5), pred_id="pred_align")
    correct_state0 = _make_predicted_state(t_req[0], pred_id="pred_align")
    traj = PredictedTrajectory(trajectory_id="t_wrong", states=[correct_state0, wrong_state])
    pred = WorldPrediction(prediction_id="pred_align", request_id="req_align", trajectories=[traj])

    with pytest.raises(TimeRangeInvalidError, match="does not match requested target time"):
        validate_prediction_against_request(pred, req)


def test_prediction_state_count_matches_target_times(base_utc_time: datetime) -> None:
    """Predicted state count must match requested target_times count."""
    from world_model.contracts.validation import validate_prediction_against_request

    t_req = [base_utc_time + timedelta(seconds=1), base_utc_time + timedelta(seconds=2)]
    req = PredictionRequest(
        request_id="req_count",
        state_history_ref="sh_01",
        context_ref="wc_01",
        task_spec_ref="ts_01",
        target_times=t_req,
    )

    traj = PredictedTrajectory(
        trajectory_id="t_short",
        states=[_make_predicted_state(t_req[0], pred_id="pred_count")],
    )
    pred = WorldPrediction(prediction_id="pred_count", request_id="req_count", trajectories=[traj])

    with pytest.raises(TimeRangeInvalidError, match="state count .* does not match"):
        validate_prediction_against_request(pred, req)


def test_deterministic_request_requires_one_output_trajectory(base_utc_time: datetime) -> None:
    """Deterministic request requires exactly 1 trajectory in the output prediction."""
    from world_model.contracts.validation import validate_prediction_against_request

    t_req = [base_utc_time + timedelta(seconds=1)]
    req = PredictionRequest(
        request_id="req_det",
        state_history_ref="sh_01",
        context_ref="wc_01",
        task_spec_ref="ts_01",
        target_times=t_req,
        prediction_options=PredictionOptions(output_mode=OutputMode.DETERMINISTIC, num_trajectories=1),
    )

    traj1 = PredictedTrajectory(trajectory_id="t1", states=[_make_predicted_state(t_req[0], pred_id="pred_det")])
    traj2 = PredictedTrajectory(trajectory_id="t2", states=[_make_predicted_state(t_req[0], pred_id="pred_det")])
    pred = WorldPrediction(prediction_id="pred_det", request_id="req_det", trajectories=[traj1, traj2])

    with pytest.raises(OutputModeUnsupportedError, match="Deterministic prediction must contain exactly 1 trajectory"):
        validate_prediction_against_request(pred, req)


def test_ensemble_request_requires_requested_trajectory_count(base_utc_time: datetime) -> None:
    """Ensemble request requires len(trajectories) == num_trajectories."""
    from world_model.contracts.validation import validate_prediction_against_request

    t_req = [base_utc_time + timedelta(seconds=1)]
    req = PredictionRequest(
        request_id="req_ens",
        state_history_ref="sh_01",
        context_ref="wc_01",
        task_spec_ref="ts_01",
        target_times=t_req,
        prediction_options=PredictionOptions(output_mode=OutputMode.ENSEMBLE, num_trajectories=3),
    )

    t1 = PredictedTrajectory(trajectory_id="t1", states=[_make_predicted_state(t_req[0], pred_id="pred_ens")])
    t2 = PredictedTrajectory(trajectory_id="t2", states=[_make_predicted_state(t_req[0], pred_id="pred_ens")])
    pred = WorldPrediction(prediction_id="pred_ens", request_id="req_ens", trajectories=[t1, t2])

    with pytest.raises(InvalidInputError, match="Ensemble prediction expected 3 trajectories, got 2"):
        validate_prediction_against_request(pred, req)


def test_prediction_validity_roundtrip() -> None:
    """PredictionValidity serialize and deserialize roundtrip."""
    v = PredictionValidity(
        status="valid",
        horizon_supported=True,
        spatial_domain_supported=True,
        condition_coverage_ok=True,
        validated_range_checks={"Re": True, "Sc": True},
        warnings=[],
    )
    d = v.model_dump(mode="json")
    restored = PredictionValidity.model_validate(d)
    assert restored.status == "valid"
    assert restored.validated_range_checks["Re"] is True


def test_uncertainty_summary_roundtrip() -> None:
    """UncertaintySummary serialize and deserialize roundtrip."""
    u = UncertaintySummary(
        method="ensemble",
        calibrated=False,
        num_trajectories=5,
        statistics={"variance": 0.042},
    )
    d = u.model_dump(mode="json")
    restored = UncertaintySummary.model_validate(d)
    assert restored.method == "ensemble"
    assert restored.calibrated is False
    assert restored.num_trajectories == 5


def test_uncertainty_summary_not_calibrated_by_default() -> None:
    """Phase 1 deterministic default is uncalibrated and method is 'none'."""
    u = UncertaintySummary()
    assert u.calibrated is False
    assert u.method == "none"
    assert u.num_trajectories == 1


def test_world_prediction_lineage_immutable_through_inner_state(base_utc_time: datetime) -> None:
    """Integration regression: lineage of states inside WorldPrediction cannot be tampered.

    Steps:
    1. Create a valid WorldPrediction with matching lineage.
    2. Attempt to reassign parent_prediction_id on an inner state's lineage.
    3. Verify the attempt raises ValidationError (frozen_instance).
    4. Confirm the prediction's parent-child lineage integrity is still valid.
    """
    from pydantic import ValidationError
    from world_model.contracts.validation import validate_prediction_against_request

    pred_id = "pred_integration_lineage"
    t1 = base_utc_time + timedelta(seconds=1)

    state = _make_predicted_state(t1, state_id="ws_int", pred_id=pred_id)
    traj = PredictedTrajectory(trajectory_id="traj_int", states=[state])
    prediction = WorldPrediction(
        prediction_id=pred_id,
        request_id="req_int",
        trajectories=[traj],
    )

    # Attempt to tamper with the lineage through the prediction's inner state
    inner_state = prediction.trajectories[0].states[0]
    with pytest.raises(ValidationError, match="frozen_instance"):
        inner_state.lineage.parent_prediction_id = "pred_tampered"

    # After failed tampering, lineage is still consistent
    assert inner_state.lineage.parent_prediction_id == pred_id

    # Validate the prediction against its request to confirm integrity
    req = PredictionRequest(
        request_id="req_int",
        state_history_ref="sh_01",
        context_ref="wc_01",
        task_spec_ref="ts_01",
        target_times=[t1],
    )
    # This should succeed — lineage was not corrupted
    validate_prediction_against_request(prediction, req)
