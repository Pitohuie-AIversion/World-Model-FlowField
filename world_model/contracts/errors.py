"""World Model Contract and Domain Error Hierarchy.

All contract-level validation errors must raise specific subclasses of ContractError.
Silent fallbacks or generic unhandled ValueError/Exception catches are forbidden.
"""

from typing import Any, Optional


class WorldModelError(Exception):
    """Base class for all World Model domain exceptions."""

    def __init__(
        self,
        message: str,
        error_code: str = "WORLD_MODEL_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code}, message={self.message!r}, details={self.details})"


class ContractError(WorldModelError):
    """Base class for contract and schema validation errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "CONTRACT_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code=error_code, details=details)


class InvalidInputError(ContractError):
    """Raised when an input fails structural or semantic validation."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="INVALID_INPUT", details=details)


class SchemaMismatchError(ContractError):
    """Raised when contract_type, schema_version, or field typing is incompatible."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="SCHEMA_MISMATCH", details=details)


class TimeRangeInvalidError(ContractError):
    """Raised when timestamps are non-monotonic, missing timezones, or out of causal sequence."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="TIME_RANGE_INVALID", details=details)


class ConditionCoverageInvalidError(ContractError):
    """Raised when conditions are evaluated outside their valid coverage interval with forbidden extrapolation."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="CONDITION_COVERAGE_INVALID", details=details)


class TaskSpecIncompatibleError(ContractError):
    """Raised when a TransitionTaskSpec is incompatible with ModelCapabilities or DatasetManifest."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="TASK_SPEC_INCOMPATIBLE", details=details)


class ModelIncompatibleError(ContractError):
    """Raised when model manifest or checkpoint cannot fulfill the required capabilities."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="MODEL_INCOMPATIBLE", details=details)


class BoundaryPolicyUnsupportedError(ContractError):
    """Raised when periodic/padding boundary requirements cannot be satisfied by backend."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="BOUNDARY_POLICY_UNSUPPORTED", details=details)


class OutputModeUnsupportedError(ContractError):
    """Raised when requested output mode (e.g. ensemble) is unsupported by model capabilities."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="OUTPUT_MODE_UNSUPPORTED", details=details)


class InsufficientStateError(ContractError):
    """Raised when state history length is shorter than required by task specification."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="INSUFFICIENT_STATE", details=details)


class OutOfValidatedRangeError(ContractError):
    """Raised when request conditions or states lie outside model validated domain."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="OUT_OF_VALIDATED_RANGE", details=details)


class PredictionFailedError(ContractError):
    """Raised when execution of prediction fails without silent fallback."""

    def __init__(
        self,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, error_code="PREDICTION_FAILED", details=details)
