"""Tests for TransitionSample contract."""

from datetime import datetime, timedelta, timezone

import pytest

from world_model.contracts.errors import (
    InvalidInputError,
    TimeRangeInvalidError,
)
from world_model.contracts.transition import TransitionSample
from world_model.contracts.world_context import WorldContext
from world_model.contracts.world_state import (
    StateKind,
    StateLineage,
    WorldComponents,
    WorldState,
)


def _make_state(t: datetime, state_id: str, kind: StateKind = StateKind.SIMULATED_GROUND_TRUTH) -> WorldState:
    lineage = StateLineage(parent_prediction_id="pred_0" if kind == StateKind.PREDICTED else None)
    return WorldState(
        state_id=state_id,
        state_kind=kind,
        timestamp=t,
        components=WorldComponents(),
        lineage=lineage,
    )


def test_transition_sample_uses_state_timestamps(
    base_utc_time: datetime, sample_world_context: WorldContext
) -> None:
    """history_times and target_times must be derived properties from WorldState.timestamp."""
    t0 = base_utc_time
    t1 = t0 + timedelta(seconds=1)
    t2 = t0 + timedelta(seconds=2)
    t_target = t0 + timedelta(seconds=3)

    hist = [_make_state(t0, "ws_h0"), _make_state(t1, "ws_h1"), _make_state(t2, "ws_h2")]
    target = [_make_state(t_target, "ws_t0")]

    sample = TransitionSample(
        sample_id="sample_001",
        task_spec_ref="task_001",
        state_history=hist,
        context=sample_world_context,
        target_states=target,
        trajectory_id="traj_001",
    )

    # Derived properties check
    assert sample.history_times == [t0, t1, t2]
    assert sample.target_times == [t_target]

    # Serialization roundtrip
    d = sample.to_dict()
    assert "history_times" not in d
    assert "target_times" not in d

    restored = TransitionSample.from_dict(d)
    assert restored.history_times == [t0, t1, t2]
    assert restored.target_times == [t_target]


def test_transition_sample_forbids_stored_history_times(
    base_utc_time: datetime, sample_world_context: WorldContext
) -> None:
    """TransitionSample must reject passing history_times or target_times directly."""
    t0 = base_utc_time
    hist = [_make_state(t0, "ws_h0")]
    target = [_make_state(t0 + timedelta(seconds=1), "ws_t0")]

    with pytest.raises(InvalidInputError, match="must not store 'history_times'"):
        TransitionSample.from_dict({
            "sample_id": "sample_bad",
            "task_spec_ref": "task_001",
            "state_history": [s.to_dict() for s in hist],
            "context": sample_world_context.to_dict(),
            "target_states": [s.to_dict() for s in target],
            "trajectory_id": "traj_001",
            "history_times": [t0.isoformat()],
        })


def test_transition_sample_causality_validation(
    base_utc_time: datetime, sample_world_context: WorldContext
) -> None:
    """Target state must be strictly after the latest history state."""
    t0 = base_utc_time
    t1 = t0 + timedelta(seconds=1)

    hist = [_make_state(t0, "ws_h0"), _make_state(t1, "ws_h1")]

    # Target in the past -> must fail
    with pytest.raises(TimeRangeInvalidError, match="Causal violation"):
        TransitionSample(
            sample_id="sample_non_causal",
            task_spec_ref="task_001",
            state_history=hist,
            context=sample_world_context,
            target_states=[_make_state(t0, "ws_t_past")],  # t0 <= t1!
            trajectory_id="traj_001",
        )
