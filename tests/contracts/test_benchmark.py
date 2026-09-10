"""Tests for BenchmarkProtocol contract."""

import pytest

from world_model.contracts.benchmark import BenchmarkProtocol
from world_model.contracts.errors import InvalidInputError
from world_model.contracts.task_spec import TransitionTaskSpec


def test_benchmark_protocol_is_independent_from_task_spec(
    sample_task_spec: TransitionTaskSpec,
) -> None:
    """BenchmarkProtocol and TransitionTaskSpec must maintain strict separation of concerns."""
    # BenchmarkProtocol references task_spec_ref, does not duplicate task fields
    proto = BenchmarkProtocol(
        benchmark_id="bench_shear_flow_lead_time",
        task_spec_ref=sample_task_spec.task_spec_id,
        dataset_manifest_ref="the_well_shear_flow",
        split_policy_ref="test",
        rollout_horizons=[1, 2, 4, 8],
        metric_suite=["relative_l2", "rmse"],
        random_seeds=[42, 100],
    )

    # BenchmarkProtocol cannot define task_spec fields like input_components
    with pytest.raises(InvalidInputError, match="must not define 'input_components'"):
        BenchmarkProtocol.from_dict({
            "benchmark_id": "bench_bad",
            "task_spec_ref": "task_01",
            "dataset_manifest_ref": "ds_01",
            "rollout_horizons": [1],
            "input_components": ["environment.fluid_state"],
        })

    # TransitionTaskSpec cannot define benchmark fields like metric_suite
    with pytest.raises(InvalidInputError, match="must not define 'metric_suite'"):
        TransitionTaskSpec.from_dict({
            "task_spec_id": "task_bad",
            "history_selection": {"window_size": 2},
            "target_time_policy": {"horizon": 1},
            "metric_suite": ["relative_l2"],
        })




def test_benchmark_protocol_roundtrip() -> None:
    """BenchmarkProtocol serialization roundtrip."""
    proto = BenchmarkProtocol(
        benchmark_id="bench_001",
        task_spec_ref="task_001",
        dataset_manifest_ref="ds_001",
        rollout_horizons=[1, 4],
        metric_suite=["relative_l2", "transition_consistency"],
        random_seeds=[42],
    )
    d = proto.to_dict()
    assert d["contract_type"] == "BenchmarkProtocol"
    assert d["rollout_horizons"] == [1, 4]

    restored = BenchmarkProtocol.from_dict(d)
    assert restored.benchmark_id == "bench_001"
    assert restored.metric_suite == ["relative_l2", "transition_consistency"]
