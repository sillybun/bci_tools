# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/bci_decoder_api/specs.py
# original git blob: 635d1a7602e8841caab5cd98b922da4f535445c7
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Caller-defined tensor descriptions and structural decoder contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import ClassVar, Mapping

import numpy as np

from .errors import SpecValidationError


def _validate_id(value: str, *, field_name: str) -> None:
    if not value or value.strip() != value:
        raise SpecValidationError(f"{field_name} must be a non-empty, trimmed string")


def _freeze_string_mapping(values: Mapping[str, str]) -> Mapping[str, str]:
    copied: dict[str, str] = {}
    for key, value in values.items():
        key_string = str(key)
        value_string = str(value)
        if not key_string or key_string.strip() != key_string or not value_string:
            raise SpecValidationError("schema attributes require non-empty trimmed keys and values")
        copied[key_string] = value_string
    return MappingProxyType(copied)


def _freeze_axis_labels(
    values: Mapping[str, tuple[str, ...]],
) -> Mapping[str, tuple[str, ...]]:
    copied: dict[str, tuple[str, ...]] = {}
    for axis, labels in values.items():
        axis_name = str(axis)
        frozen_labels = tuple(str(label) for label in labels)
        if not axis_name or axis_name.strip() != axis_name:
            raise SpecValidationError("axis-label keys must be non-empty and trimmed")
        if any(not label or label.strip() != label for label in frozen_labels):
            raise SpecValidationError("axis labels must be non-empty and trimmed")
        if len(set(frozen_labels)) != len(frozen_labels):
            raise SpecValidationError(f"axis labels for {axis_name!r} must be unique")
        copied[axis_name] = frozen_labels
    return MappingProxyType(copied)


@dataclass(frozen=True)
class SchemaDefinition:
    """Caller-defined tensor structure with optional descriptive metadata.

    No catalog registration is required. Description, units and attributes do
    not select preprocessing or impose numeric-domain rules in the SDK.
    """

    schema_id: str
    tensor_role: str
    dtype: str
    axes: tuple[str, ...]
    description: str = ""
    value_unit: str | None = None
    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        _validate_id(self.schema_id, field_name="schema_id")
        if self.tensor_role not in {"feature", "action", "feedback"}:
            raise SpecValidationError("tensor_role must be feature, action, or feedback")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise SpecValidationError(f"unsupported dtype: {self.dtype}") from exc
        if dtype != np.dtype("float32"):
            raise SpecValidationError("public decoder tensors must use float32")
        if not axes or len(set(axes)) != len(axes):
            raise SpecValidationError("schema axes must be non-empty and unique")
        if any(not axis or axis.strip() != axis for axis in axes):
            raise SpecValidationError("axis names must be non-empty and trimmed")
        if not isinstance(self.description, str):
            raise SpecValidationError("schema description must be a string")
        if self.value_unit is not None and not isinstance(self.value_unit, str):
            raise SpecValidationError("value_unit must be None or a string")
        object.__setattr__(self, "dtype", dtype.name)
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "attributes", _freeze_string_mapping(self.attributes))


@dataclass(frozen=True)
class TensorLayout:
    """Concrete shape and element ordering for one deployed tensor contract."""

    layout_id: str
    shape: tuple[int, ...]
    axis_labels: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        _validate_id(self.layout_id, field_name="layout_id")
        if not shape or any(not isinstance(size, int) or size <= 0 for size in shape):
            raise SpecValidationError("all layout dimensions must be positive integers")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "axis_labels", _freeze_axis_labels(self.axis_labels))


@dataclass(frozen=True)
class ArraySpec:
    """A semantic schema bound to one concrete layout."""

    _expected_role: ClassVar[str]
    schema: SchemaDefinition
    layout: TensorLayout

    def __post_init__(self) -> None:
        if self.schema.tensor_role != self._expected_role:
            raise SpecValidationError(
                f"{type(self).__name__} requires a {self._expected_role!r} schema"
            )
        if len(self.schema.axes) != len(self.layout.shape):
            raise SpecValidationError("schema axes and layout shape must have the same rank")
        for axis, labels in self.layout.axis_labels.items():
            if axis not in self.schema.axes:
                raise SpecValidationError(f"layout labels refer to unknown axis {axis!r}")
            axis_index = self.schema.axes.index(axis)
            if len(labels) != self.layout.shape[axis_index]:
                raise SpecValidationError(
                    f"layout labels for {axis!r} must match that axis dimension"
                )

    @property
    def schema_id(self) -> str:
        return self.schema.schema_id

    @property
    def layout_id(self) -> str:
        return self.layout.layout_id

    @property
    def dtype(self) -> str:
        return self.schema.dtype

    @property
    def axes(self) -> tuple[str, ...]:
        return self.schema.axes

    @property
    def shape(self) -> tuple[int, ...]:
        return self.layout.shape

    @property
    def rank(self) -> int:
        return len(self.layout.shape)


@dataclass(frozen=True)
class FeatureSpec(ArraySpec):
    """Concrete feature contract for one logical decoder update."""

    _expected_role: ClassVar[str] = "feature"
    time_axis: int | None = None
    sample_period_ns: int | None = None
    window_alignment: str = "right"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.time_axis is None:
            if self.sample_period_ns is not None:
                raise SpecValidationError("sample_period_ns requires a time_axis")
        else:
            if not 0 <= self.time_axis < self.rank:
                raise SpecValidationError("time_axis is outside the tensor rank")
            if self.axes[self.time_axis] != "time":
                raise SpecValidationError("the time_axis must point to an axis named 'time'")
            if self.sample_period_ns is None or self.sample_period_ns <= 0:
                raise SpecValidationError("a time axis requires a positive sample_period_ns")
        if self.window_alignment != "right":
            raise SpecValidationError("v2 only supports right-aligned feature windows")


@dataclass(frozen=True)
class ActionSpec(ArraySpec):
    """Concrete decoder-action contract."""

    _expected_role: ClassVar[str] = "action"


@dataclass(frozen=True)
class FeedbackSpec(ArraySpec):
    """Concrete external-feedback contract."""

    _expected_role: ClassVar[str] = "feedback"


def validate_values_against_spec(
    values: np.ndarray,
    spec: ArraySpec,
    *,
    field_name: str,
) -> None:
    """Raise SpecValidationError when an array violates a concrete layout."""

    if not isinstance(values, np.ndarray):
        raise SpecValidationError(f"{field_name} must be a NumPy array")
    if values.dtype != np.dtype(spec.dtype):
        raise SpecValidationError(
            f"{field_name} dtype {values.dtype} does not match {spec.dtype}"
        )
    if values.shape != spec.shape:
        raise SpecValidationError(
            f"{field_name} shape {values.shape} does not match {spec.shape}"
        )
