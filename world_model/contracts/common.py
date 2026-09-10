"""Common base contracts, utilities, and validation primitives.

Provides:
- ContractBase: Self-describing top-level contract with contract_type and schema_version
- Timezone-aware datetime serialization and verification
- Standard ID generation
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, TypeVar

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from world_model.contracts.errors import (
    InvalidInputError,
    SchemaMismatchError,
    TimeRangeInvalidError,
)

SCHEMA_VERSION_DEFAULT = "1.8.0"


def _reject_constant(c: str) -> None:
    raise InvalidInputError(f"Non-finite JSON constant '{c}' (NaN/Infinity) is forbidden in contract data.")


def assert_finite_values(val: Any) -> None:
    """Recursively ensure that no float value is NaN, +Inf, or -Inf."""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            raise InvalidInputError(
                f"Non-finite float value '{val}' (NaN/Infinity) is forbidden in contract data."
            )
    elif isinstance(val, dict):
        for v in val.values():
            assert_finite_values(v)
    elif isinstance(val, (list, tuple, set)):
        for v in val:
            assert_finite_values(v)


def validate_timezone_aware(dt: datetime) -> datetime:
    """Ensure datetime has an explicit timezone (e.g. UTC, +08:00)."""
    if not isinstance(dt, datetime):
        raise TimeRangeInvalidError(f"Expected datetime instance, got {type(dt)}")
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        raise TimeRangeInvalidError(
            f"Timestamp {dt.isoformat()} is naive. Timestamps must be timezone-aware (ISO-8601 with timezone)."
        )
    return dt


TimezoneAwareDatetime = Annotated[datetime, AfterValidator(validate_timezone_aware)]

T = TypeVar("T", bound="ContractBase")


def generate_id(prefix: str) -> str:
    """Generate a unique ID with human-readable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def current_utc_time() -> datetime:
    """Get current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class ContractBase(BaseModel):
    """Top-level self-describing contract base class.

    All top-level persistent/inter-process payloads inherit from ContractBase.
    Requires contract_type and schema_version, and forbids arbitrary extra attributes.
    """

    contract_type: str = Field(..., description="Canonical contract type identifier")
    schema_version: str = Field(
        default=SCHEMA_VERSION_DEFAULT,
        description="Semantic schema version (e.g. 1.8.0)",
    )

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_contract_header(cls, values: Any) -> Any:
        """Verify contract_type matches this class name if provided, and check schema_version."""
        if not isinstance(values, dict):
            return values

        field_info = cls.model_fields.get("contract_type")
        if field_info is not None and isinstance(field_info.default, str):
            expected_type = field_info.default
        else:
            expected_type = cls.__name__

        provided_type = values.get("contract_type")
        if provided_type is not None and provided_type != expected_type:
            raise SchemaMismatchError(
                f"Contract type mismatch: expected '{expected_type}', got '{provided_type}'",
                details={"expected": expected_type, "provided": provided_type},
            )

        # Set default contract_type if not provided
        if "contract_type" not in values:
            values["contract_type"] = expected_type

        # Check schema_version: fail closed on unsupported/mismatched version
        provided_version = values.get("schema_version")
        if provided_version is not None:
            if provided_version != SCHEMA_VERSION_DEFAULT:
                raise SchemaMismatchError(
                    f"Schema version mismatch: expected '{SCHEMA_VERSION_DEFAULT}', got '{provided_version}'",
                    details={"expected": SCHEMA_VERSION_DEFAULT, "provided": provided_version},
                )
        else:
            values["schema_version"] = SCHEMA_VERSION_DEFAULT

        return values

    def to_dict(self) -> dict[str, Any]:
        """Serialize contract to a JSON-compatible dictionary."""
        d = self.model_dump(mode="json")
        assert_finite_values(d)
        return d

    def to_json(self, indent: int | None = None) -> str:
        """Serialize contract to a JSON formatted string."""
        d = self.to_dict()
        try:
            return json.dumps(d, indent=indent, allow_nan=False)
        except ValueError as e:
            raise InvalidInputError(f"Failed to serialize {self.__class__.__name__} to JSON: {e}") from e

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Construct contract from a dictionary with strict validation."""
        if not isinstance(data, dict):
            raise InvalidInputError(f"Expected dict input for {cls.__name__}, got {type(data)}")
        assert_finite_values(data)
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise InvalidInputError(f"Validation failed for {cls.__name__}: {e}") from e

    @classmethod
    def from_json(cls: type[T], json_str: str) -> T:
        """Construct contract from a JSON string with strict validation."""
        if not isinstance(json_str, (str, bytes)):
            raise InvalidInputError(f"Expected JSON str/bytes for {cls.__name__}, got {type(json_str)}")
        try:
            data = json.loads(json_str, parse_constant=_reject_constant)
        except InvalidInputError:
            raise
        except Exception as e:
            raise InvalidInputError(f"Malformed JSON string for {cls.__name__}: {e}") from e
        return cls.from_dict(data)
