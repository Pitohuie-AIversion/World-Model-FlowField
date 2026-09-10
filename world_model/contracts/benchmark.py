"""BenchmarkProtocol domain contract.

Defines fair, reproducible evaluation protocols across world models.
Strictly decoupled from TransitionTaskSpec:
- TransitionTaskSpec defines WHAT state transition is being studied.
- BenchmarkProtocol defines HOW different backends are evaluated and compared.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field, model_validator

from world_model.contracts.common import ContractBase, generate_id
from world_model.contracts.errors import InvalidInputError

FORBIDDEN_BENCHMARK_KEYS = {
    "input_components",
    "target_components",
    "history_selection",
    "target_time_policy",
    "action_policy",
}


class BenchmarkProtocol(ContractBase):
    """Protocol defining standardized evaluation rules, metrics, and compute budgets."""

    contract_type: str = Field(default="BenchmarkProtocol", description="Contract type identifier")
    benchmark_id: str = Field(
        default_factory=lambda: generate_id("bench"),
        description="Unique benchmark protocol identifier",
    )
    task_spec_ref: str = Field(
        ...,
        description="Reference to TransitionTaskSpec defining the physical transition task",
    )
    dataset_manifest_ref: str = Field(
        ...,
        description="Reference to DatasetManifest providing test trajectories",
    )
    split_policy_ref: str = Field(
        default="test",
        description="Dataset split targeted for evaluation (e.g. 'test', 'out_of_distribution')",
    )
    rollout_horizons: list[int] = Field(
        ...,
        description="Evaluation rollout horizons H (e.g. [1, 4, 8, 16])",
    )
    metric_suite: list[str] = Field(
        default_factory=lambda: [
            "relative_l2",
            "rmse",
            "vorticity_error",
            "transition_consistency",
        ],
        description="List of evaluation metrics to compute",
    )
    random_seeds: list[int] = Field(
        default_factory=lambda: [42],
        description="RNG seeds for stochastic evaluation or multi-run benchmarking",
    )
    compute_budget_policy: Optional[dict[str, Any]] = Field(
        default=None,
        description="Max inference latency, VRAM limit, or FLOP budget",
    )
    reporting_policy: Optional[dict[str, Any]] = Field(
        default=None,
        description="Format and aggregation rules for benchmark reporting",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_task_spec_overlap(cls, values: Any) -> Any:
        if isinstance(values, dict):
            for k in FORBIDDEN_BENCHMARK_KEYS:
                if k in values:
                    raise InvalidInputError(
                        f"BenchmarkProtocol must not define '{k}'. "
                        f"State transition semantics belong exclusively to TransitionTaskSpec.",
                        details={"forbidden_key": k},
                    )
        return values
