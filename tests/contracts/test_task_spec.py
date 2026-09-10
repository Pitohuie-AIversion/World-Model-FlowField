"""Tests for TransitionTaskSpec Single Source of Truth."""

import pytest

from world_model.contracts.errors import InvalidInputError
from world_model.contracts.task_spec import (
    ActionPolicy,
    HistorySelection,
    RolloutPolicy,
    TargetTimePolicy,
    TransitionTaskSpec,
)


def test_transition_task_spec_roundtrip(sample_task_spec: TransitionTaskSpec) -> None:
    """TransitionTaskSpec serialize / deserialize roundtrip."""
    d = sample_task_spec.to_dict()
    assert d["contract_type"] == "TransitionTaskSpec"
    assert d["action_policy"] == "none"
    assert d["rollout_policy"] == "one_step"

    restored = TransitionTaskSpec.from_dict(d)
    assert restored.task_spec_id == sample_task_spec.task_spec_id
    assert restored.action_policy == ActionPolicy.NONE
    assert restored.rollout_policy == RolloutPolicy.ONE_STEP
    assert restored.history_selection.window_size == 4


def test_transition_task_spec_has_no_benchmark_settings() -> None:
    """TransitionTaskSpec must not own dataset split, metrics, or compute budget."""
    with pytest.raises(InvalidInputError, match="must not define 'dataset_split'"):
        TransitionTaskSpec.from_dict({
            "task_spec_id": "task_invalid",
            "dataset_split": "train",
            "history_selection": {"window_size": 2},
            "target_time_policy": {"horizon": 1},
        })

    with pytest.raises(InvalidInputError, match="must not define 'metric_suite'"):
        TransitionTaskSpec.from_dict({
            "task_spec_id": "task_invalid",
            "metric_suite": ["relative_l2"],
            "history_selection": {"window_size": 2},
            "target_time_policy": {"horizon": 1},
        })


# Backward compat alias
test_task_spec_forbids_split_and_metrics = test_transition_task_spec_has_no_benchmark_settings
