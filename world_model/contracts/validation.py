"""Resolved Semantic Validation.

Separates structural contract validation (performed on individual objects)
from cross-object semantic consistency checks requiring resolved references or
multi-entity alignment (Request <-> TaskSpec, Prediction <-> Request).
"""

from __future__ import annotations

from typing import Optional

from world_model.contracts.errors import (
    InvalidInputError,
    OutputModeUnsupportedError,
    SchemaMismatchError,
    TimeRangeInvalidError,
)
from world_model.contracts.prediction import (
    OutputMode,
    PredictionRequest,
    WorldPrediction,
)
from world_model.contracts.task_spec import ActionPolicy, TransitionTaskSpec
from world_model.contracts.world_context import WorldContext
from world_model.contracts.world_state import WorldState


def validate_resolved_request(
    request: PredictionRequest,
    task_spec: TransitionTaskSpec,
    state_history: Optional[list[WorldState]] = None,
    context: Optional[WorldContext] = None,
) -> None:
    """Validate semantic consistency across resolved request components."""
    # 1. ActionPolicy consistency
    if task_spec.action_policy == ActionPolicy.NONE and request.actions is not None:
        raise InvalidInputError(
            "TaskSpec specifies ActionPolicy.NONE, but actions sequence was provided in PredictionRequest."
        )
    if task_spec.action_policy == ActionPolicy.REQUIRED and request.actions is None:
        raise InvalidInputError(
            "TaskSpec specifies ActionPolicy.REQUIRED, but actions sequence was not provided in PredictionRequest."
        )

    # 2. History validation if provided externally
    effective_history = state_history if state_history is not None else request.state_history
    if effective_history is not None:
        if not effective_history:
            raise InvalidInputError("Effective state_history must not be empty.")

        for i in range(len(effective_history) - 1):
            if effective_history[i].timestamp >= effective_history[i + 1].timestamp:
                raise TimeRangeInvalidError(
                    f"Resolved state_history timestamps must be strictly monotonic increasing: "
                    f"{effective_history[i].timestamp} >= {effective_history[i + 1].timestamp}"
                )

        last_hist_t = effective_history[-1].timestamp
        first_target_t = request.target_times[0]
        if first_target_t <= last_hist_t:
            raise TimeRangeInvalidError(
                f"Causal violation: first target timestamp ({first_target_t}) "
                f"must be strictly after last history timestamp ({last_hist_t})."
            )

        if task_spec.history_selection.window_size is not None:
            if len(effective_history) < task_spec.history_selection.window_size:
                raise InvalidInputError(
                    f"History length ({len(effective_history)}) is less than required window_size "
                    f"({task_spec.history_selection.window_size})."
                )

    # 3. Target time policy horizon check if applicable
    if task_spec.target_time_policy.horizon is not None:
        if len(request.target_times) != task_spec.target_time_policy.horizon:
            raise InvalidInputError(
                f"Target times count ({len(request.target_times)}) does not match "
                f"TaskSpec horizon ({task_spec.target_time_policy.horizon})."
            )


def validate_prediction_against_request(
    prediction: WorldPrediction,
    request: PredictionRequest,
) -> None:
    """Validate that WorldPrediction fulfills the corresponding PredictionRequest."""
    # 1. Request ID matching
    if prediction.request_id != request.request_id:
        raise SchemaMismatchError(
            f"Prediction request_id '{prediction.request_id}' does not match "
            f"request request_id '{request.request_id}'.",
            details={"prediction_request_id": prediction.request_id, "request_id": request.request_id},
        )

    # 2. Output mode & trajectory count (K invariant)
    opts = request.prediction_options
    if opts.output_mode == OutputMode.DETERMINISTIC:
        if len(prediction.trajectories) != 1:
            raise OutputModeUnsupportedError(
                f"Deterministic prediction must contain exactly 1 trajectory, "
                f"got {len(prediction.trajectories)}."
            )
    elif opts.output_mode == OutputMode.ENSEMBLE:
        if prediction.status == "success":
            if len(prediction.trajectories) != opts.num_trajectories:
                raise InvalidInputError(
                    f"Ensemble prediction expected {opts.num_trajectories} trajectories, "
                    f"got {len(prediction.trajectories)}."
                )

    # 3. Target times alignment across all trajectories
    for traj_idx, traj in enumerate(prediction.trajectories):
        if len(traj.states) != len(request.target_times):
            raise TimeRangeInvalidError(
                f"Trajectory {traj_idx} state count ({len(traj.states)}) does not match "
                f"requested target_times count ({len(request.target_times)})."
            )

        for state_idx, (state, target_t) in enumerate(zip(traj.states, request.target_times)):
            if state.timestamp != target_t:
                raise TimeRangeInvalidError(
                    f"Trajectory {traj_idx} state {state_idx} timestamp ({state.timestamp}) "
                    f"does not match requested target time ({target_t})."
                )
