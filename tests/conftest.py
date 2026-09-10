"""Pytest fixtures for world_model contract testing."""

from datetime import datetime, timedelta, timezone

import pytest

from world_model.contracts.common import generate_id
from world_model.contracts.condition import (
    ConditionSeries,
    ExtrapolationPolicy,
    InterpolationPolicy,
)
from world_model.contracts.manifests import DatasetManifest
from world_model.contracts.task_spec import (
    ActionPolicy,
    HistorySelection,
    RolloutPolicy,
    TargetTimePolicy,
    TransitionTaskSpec,
)
from world_model.contracts.world_context import (
    GridSpec,
    StaticConditions,
    WorldContext,
)
from world_model.contracts.world_state import (
    EnvironmentComponents,
    FieldRef,
    FluidState,
    StateKind,
    StateLineage,
    WorldComponents,
    WorldState,
)


@pytest.fixture
def base_utc_time() -> datetime:
    return datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def sample_fluid_state() -> FluidState:
    return FluidState(
        fields={
            "velocity.x": FieldRef(inline_value=[1.0, 2.0, 3.0]),
            "velocity.y": FieldRef(inline_value=[0.1, 0.2, 0.3]),
            "pressure": FieldRef(inline_value=[101.3, 101.5, 101.2]),
        },
        spatial_ref="grid_shear_flow_128x256",
        mask_ref=None,
    )


@pytest.fixture
def sample_ground_truth_world_state(
    base_utc_time: datetime, sample_fluid_state: FluidState
) -> WorldState:
    return WorldState(
        state_id="ws_sim_001",
        state_kind=StateKind.SIMULATED_GROUND_TRUTH,
        timestamp=base_utc_time,
        components=WorldComponents(
            environment=EnvironmentComponents(fluid_state=sample_fluid_state)
        ),
        lineage=StateLineage(source_state_ids=["ws_sim_000"]),
        provenance={"dataset": "the_well_shear_flow", "trajectory_id": "traj_01"},
    )


@pytest.fixture
def sample_predicted_world_state(
    base_utc_time: datetime, sample_fluid_state: FluidState
) -> WorldState:
    return WorldState(
        state_id="ws_pred_001",
        state_kind=StateKind.PREDICTED,
        timestamp=base_utc_time + timedelta(seconds=1),
        components=WorldComponents(
            environment=EnvironmentComponents(fluid_state=sample_fluid_state)
        ),
        lineage=StateLineage(parent_prediction_id="pred_test_001"),
    )


@pytest.fixture
def sample_world_context(base_utc_time: datetime) -> WorldContext:
    grid = GridSpec(
        grid_id="grid_shear_flow_128x256",
        axes=["y", "x"],
        shape=[128, 256],
        grid_type="uniform_cartesian",
        periodic_axes=["x", "y"],
    )
    forcing = ConditionSeries(
        condition_spec_ref="spec_surface_wind",
        timestamps=[base_utc_time, base_utc_time + timedelta(seconds=10)],
        values=[1.0, 1.5],
        interpolation_policy=InterpolationPolicy.LINEAR,
        extrapolation_policy=ExtrapolationPolicy.FORBIDDEN,
    )
    return WorldContext(
        context_id="wc_001",
        static_conditions=StaticConditions(
            spatial_domains={"grid_shear_flow_128x256": grid},
            physics_parameters={"Re": 1000.0, "Sc": 1.0},
        ),
        temporal_conditions={"surface_wind": forcing},
        scenario_id="scenario_shear_flow_re1000",
    )


@pytest.fixture
def sample_task_spec() -> TransitionTaskSpec:
    return TransitionTaskSpec(
        task_spec_id="task_phase1_shear_flow_onestep",
        input_components=["environment.fluid_state"],
        target_components=["environment.fluid_state"],
        history_selection=HistorySelection(window_size=4, step=1),
        target_time_policy=TargetTimePolicy(horizon=1, step=1, lead_times=[1.0]),
        required_context_keys=["physics_parameters.Re"],
        action_policy=ActionPolicy.NONE,
        rollout_policy=RolloutPolicy.ONE_STEP,
    )


@pytest.fixture
def sample_dataset_manifest() -> DatasetManifest:
    return DatasetManifest(
        dataset_id="the_well_shear_flow",
        revision="v1.0.0",
        source="The Well: A Benchmark for Physical Dynamics",
        license="CC-BY-4.0",
        file_list=[
            {"path": "train/shear_flow_001.h5", "checksum": "sha256:abc123"},
            {"path": "test/shear_flow_test.h5", "checksum": "sha256:def456"},
        ],
        field_mapping={
            "velocity_x": "velocity.x",
            "velocity_y": "velocity.y",
            "p": "pressure",
        },
        split_trajectory_ids={
            "train": ["traj_001", "traj_002"],
            "valid": ["traj_003"],
            "test": ["traj_004"],
        },
        time_metadata={"dt": 0.1, "time_unit": "second", "steps": 100},
        spatial_metadata={
            "axes": ["y", "x"],
            "shape": [128, 256],
            "domain_extent": [[0.0, 1.0], [0.0, 2.0]],
            "boundary_conditions": "periodic",
        },
        training_statistics_ref="stats/train_norm_stats.json",
    )
