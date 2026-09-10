"""Tests for WorldContext and physical condition constraints."""

import pytest

from world_model.contracts.errors import InvalidInputError
from world_model.contracts.world_context import (
    GridSpec,
    StaticConditions,
    WorldContext,
)


def test_world_context_has_no_state_or_target_time(sample_world_context: WorldContext) -> None:
    """WorldContext must not contain state_timestamp, target_times, history_times, or dataset splits."""
    # Attribute check
    assert not hasattr(sample_world_context, "target_times")
    assert not hasattr(sample_world_context, "history_times")
    assert not hasattr(sample_world_context, "state_timestamp")
    assert not hasattr(sample_world_context, "timestamp")
    assert not hasattr(sample_world_context, "dataset_split")

    # Initialization with forbidden keys must fail
    with pytest.raises(InvalidInputError, match="must not contain 'target_times'"):
        WorldContext.from_dict({
            "context_id": "wc_invalid",
            "target_times": ["2026-09-10T12:00:00Z"],
            "static_conditions": {},
            "temporal_conditions": {},
        })

    with pytest.raises(InvalidInputError, match="must not contain 'state_timestamp'"):
        WorldContext.from_dict({
            "context_id": "wc_invalid",
            "state_timestamp": "2026-09-10T12:00:00Z",
            "static_conditions": {},
            "temporal_conditions": {},
        })

    with pytest.raises(InvalidInputError, match="must not contain 'history_times'"):
        WorldContext.from_dict({
            "context_id": "wc_invalid",
            "history_times": ["2026-09-10T12:00:00Z"],
            "static_conditions": {},
            "temporal_conditions": {},
        })

    with pytest.raises(InvalidInputError, match="must not contain 'dataset_split'"):
        WorldContext.from_dict({
            "context_id": "wc_invalid",
            "dataset_split": "train",
            "static_conditions": {},
            "temporal_conditions": {},
        })


def test_grid_spec_shape_matches_axes() -> None:
    """GridSpec shape length must equal axes length, dimensions must be > 0, and axes must not repeat."""
    # Mismatched lengths: 2 axes, 3 shape dims
    with pytest.raises(InvalidInputError, match="len\\(shape\\) .* must match len\\(axes\\)"):
        GridSpec(grid_id="g1", axes=["y", "x"], shape=[128, 256, 10])

    # Non-positive shape dimension
    with pytest.raises(InvalidInputError, match="shape dimensions must be strictly positive"):
        GridSpec(grid_id="g2", axes=["y", "x"], shape=[128, 0])

    with pytest.raises(InvalidInputError, match="shape dimensions must be strictly positive"):
        GridSpec(grid_id="g3", axes=["y", "x"], shape=[-128, 256])

    # Duplicate axis names
    with pytest.raises(InvalidInputError, match="axes must not contain duplicate names"):
        GridSpec(grid_id="g4", axes=["x", "x"], shape=[10, 10])


def test_grid_spec_periodic_axes_are_valid() -> None:
    """GridSpec periodic_axes must be a non-duplicate subset of axes."""
    # Periodic axis not in axes
    with pytest.raises(InvalidInputError, match="periodic axis 'z' is not declared in axes"):
        GridSpec(grid_id="g5", axes=["y", "x"], periodic_axes=["x", "z"])

    # Duplicate periodic axis
    with pytest.raises(InvalidInputError, match="periodic_axes must not contain duplicate names"):
        GridSpec(grid_id="g6", axes=["y", "x"], periodic_axes=["x", "x"])

    # Valid subset
    grid = GridSpec(grid_id="g7", axes=["z", "y", "x"], shape=[32, 64, 128], periodic_axes=["x", "y"])
    assert grid.periodic_axes == ["x", "y"]


def test_world_context_roundtrip(sample_world_context: WorldContext) -> None:
    """WorldContext serialization roundtrip."""
    d = sample_world_context.to_dict()
    assert d["contract_type"] == "WorldContext"
    assert "surface_wind" in d["temporal_conditions"]

    restored = WorldContext.from_dict(d)
    assert restored.context_id == sample_world_context.context_id
    assert "surface_wind" in restored.temporal_conditions
    assert "grid_shear_flow_128x256" in restored.static_conditions.spatial_domains


def test_world_context_namespace_isolation() -> None:
    """Same canonical condition name cannot appear in both static and temporal conditions."""
    from datetime import datetime, timezone
    from world_model.contracts.condition import ConditionSeries

    t0 = datetime(2026, 9, 10, 12, 0, 0, tzinfo=timezone.utc)
    series = ConditionSeries(
        condition_spec_ref="spec_pressure_grad",
        timestamps=[t0],
        values=[0.5],
    )

    with pytest.raises(InvalidInputError, match="Condition namespace collision"):
        WorldContext(
            context_id="wc_collision",
            static_conditions=StaticConditions(
                physics_parameters={"pressure_grad": 0.5}
            ),
            temporal_conditions={"pressure_grad": series},
        )
