"""Tests for WorldContext and physical condition constraints."""

import pytest

from world_model.contracts.errors import InvalidInputError
from world_model.contracts.world_context import (
    GridSpec,
    StaticConditions,
    WorldContext,
)


def test_world_context_has_no_target_times(sample_world_context: WorldContext) -> None:
    """WorldContext must not contain target_times, history_times, or dataset splits."""
    # Attribute check
    assert not hasattr(sample_world_context, "target_times")
    assert not hasattr(sample_world_context, "history_times")
    assert not hasattr(sample_world_context, "dataset_split")

    # Initialization with forbidden keys must fail
    with pytest.raises(InvalidInputError, match="must not contain 'target_times'"):
        WorldContext.from_dict({
            "context_id": "wc_invalid",
            "target_times": ["2026-09-10T12:00:00Z"],
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
