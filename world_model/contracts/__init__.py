"""World Model Domain Contracts and Interfaces."""

from world_model.contracts.action import ActionSequence
from world_model.contracts.benchmark import BenchmarkProtocol
from world_model.contracts.common import (
    SCHEMA_VERSION_DEFAULT,
    ContractBase,
    TimezoneAwareDatetime,
    current_utc_time,
    generate_id,
    validate_timezone_aware,
)
from world_model.contracts.condition import (
    ConditionSeries,
    ConditionSpec,
    ExtrapolationPolicy,
    InterpolationPolicy,
)
from world_model.contracts.errors import (
    BoundaryPolicyUnsupportedError,
    ConditionCoverageInvalidError,
    ContractError,
    InsufficientStateError,
    InvalidInputError,
    ModelIncompatibleError,
    OutOfValidatedRangeError,
    OutputModeUnsupportedError,
    PredictionFailedError,
    SchemaMismatchError,
    TaskSpecIncompatibleError,
    TimeRangeInvalidError,
    WorldModelError,
)
from world_model.contracts.manifests import (
    DatasetManifest,
    ModelCapabilities,
    ModelManifest,
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
    RolloutPolicy,
    StateKindPolicy,
    TargetTimePolicy,
    TransitionTaskSpec,
)
from world_model.contracts.trajectory import PredictedTrajectory
from world_model.contracts.transition import TransitionSample
from world_model.contracts.validation import (
    validate_prediction_against_request,
    validate_resolved_request,
)
from world_model.contracts.world_context import (
    GridSpec,
    StaticConditions,
    WorldContext,
)
from world_model.contracts.world_state import (
    EnvironmentComponents,
    FieldRef,
    FieldSpec,
    FluidState,
    StateKind,
    StateLineage,
    WorldComponents,
    WorldState,
)

__all__ = [
    # Base & Common
    "ContractBase",
    "TimezoneAwareDatetime",
    "validate_timezone_aware",
    "generate_id",
    "current_utc_time",
    "SCHEMA_VERSION_DEFAULT",
    # Errors
    "WorldModelError",
    "ContractError",
    "InvalidInputError",
    "SchemaMismatchError",
    "TimeRangeInvalidError",
    "ConditionCoverageInvalidError",
    "TaskSpecIncompatibleError",
    "ModelIncompatibleError",
    "BoundaryPolicyUnsupportedError",
    "OutputModeUnsupportedError",
    "InsufficientStateError",
    "OutOfValidatedRangeError",
    "PredictionFailedError",
    # WorldState
    "StateKind",
    "FieldSpec",
    "FieldRef",
    "FluidState",
    "EnvironmentComponents",
    "WorldComponents",
    "StateLineage",
    "WorldState",
    # Conditions
    "InterpolationPolicy",
    "ExtrapolationPolicy",
    "ConditionSpec",
    "ConditionSeries",
    # WorldContext
    "GridSpec",
    "StaticConditions",
    "WorldContext",
    # Action
    "ActionSequence",
    # TaskSpec
    "ActionPolicy",
    "RolloutPolicy",
    "HistorySelection",
    "TargetTimePolicy",
    "StateKindPolicy",
    "TransitionTaskSpec",
    # Transition
    "TransitionSample",
    # Trajectory
    "PredictedTrajectory",
    # Prediction
    "OutputMode",
    "PredictionOptions",
    "PredictionRequest",
    "PredictionValidity",
    "UncertaintySummary",
    "WorldPrediction",
    # Validation
    "validate_resolved_request",
    "validate_prediction_against_request",
    # Manifests
    "DatasetManifest",
    "ModelCapabilities",
    "ModelManifest",
    # Benchmark
    "BenchmarkProtocol",
]
