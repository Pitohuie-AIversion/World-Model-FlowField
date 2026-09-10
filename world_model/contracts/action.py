"""ActionSequence domain contract.

Reserved for robot control commands and actions.
Phase 1 defaults to actions=None, but the interface contract is strictly defined.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from world_model.contracts.common import ContractBase, generate_id


class ActionSequence(ContractBase):
    """Sequence of robot control commands/actuation inputs over time."""

    contract_type: str = Field(default="ActionSequence", description="Contract type identifier")
    action_sequence_id: str = Field(
        default_factory=lambda: generate_id("act_seq"),
        description="Unique action sequence identifier",
    )
    robot_id: str = Field(..., description="Target robot platform identifier")
    timestamps_or_intervals: list[Any] = Field(
        ...,
        description="Execution time points or intervals [t_start, t_end]",
    )
    action_type: str = Field(
        ...,
        description="Action semantic type (e.g. thrust_vector, rpm, trajectory_waypoint)",
    )
    controls_or_commands: list[Any] = Field(
        ...,
        description="Control input array or command payloads",
    )
    interpolation_or_hold_policy: str = Field(
        default="zero_order_hold",
        description="Hold / interpolation policy across control steps",
    )
    validity: Optional[dict[str, Any]] = Field(
        default=None,
        description="Validity bounds or safety verification flags",
    )
