# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/bci_decoder_api/artifact.py
# original git blob: 53f8d01582e279d2bd59ce77b46b403ed73ce040
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Immutable model artifacts and safe directory persistence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .contracts import JsonValue, TrainingProvenance, frozen_arrays
from .errors import (
    ArtifactSerializationError,
    ArtifactValidationError,
    UnsupportedArtifactVersionError,
)
from .specs import (
    ActionSpec,
    FeedbackSpec,
    FeatureSpec,
    SchemaDefinition,
    TensorLayout,
)


SUPPORTED_API_MAJOR = 2
SUPPORTED_ARTIFACT_FORMAT_MAJOR = 2
_MODEL_TYPE_RE = re.compile(r"^[a-z0-9_]+$")
_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")


def _major(version: str) -> int:
    if not isinstance(version, str) or not _VERSION_RE.fullmatch(version):
        raise ArtifactValidationError(f"invalid version string: {version!r}")
    return int(version.split(".", 1)[0])


def _freeze_json(value: Any) -> JsonValue | tuple[Any, ...] | Mapping[str, Any]:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not np.isfinite(value):
            raise ArtifactValidationError("JSON configuration values must be finite")
        return value
    raise ArtifactValidationError(f"configuration value is not JSON-compatible: {type(value)!r}")


def _thaw_json(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def json_sha256(value: Mapping[str, Any]) -> str:
    """Stable hash for a JSON-compatible configuration mapping."""

    frozen = _freeze_json(value)
    raw = json.dumps(_thaw_json(frozen), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelArtifact:
    decoder_api_version: str
    artifact_format_version: str
    model_type: str
    model_version: str
    implementation_version: str
    created_at_ns: int
    feature_spec: FeatureSpec
    action_spec: ActionSpec
    feedback_spec: FeedbackSpec | None
    parameters: Mapping[str, np.ndarray]
    training_config: Mapping[str, JsonValue]
    runtime_config: Mapping[str, JsonValue]
    provenance: TrainingProvenance

    def __post_init__(self) -> None:
        if _major(self.decoder_api_version) != SUPPORTED_API_MAJOR:
            raise ArtifactValidationError(
                f"unsupported decoder API version: {self.decoder_api_version}"
            )
        if _major(self.artifact_format_version) != SUPPORTED_ARTIFACT_FORMAT_MAJOR:
            raise UnsupportedArtifactVersionError(self.artifact_format_version)
        _major(self.implementation_version)
        if not _MODEL_TYPE_RE.fullmatch(self.model_type):
            raise ArtifactValidationError(
                "model_type must contain only lowercase letters, digits, and underscores"
            )
        if not self.model_version or self.created_at_ns < 0:
            raise ArtifactValidationError("model_version is required and created_at_ns must be non-negative")
        object.__setattr__(self, "parameters", frozen_arrays(self.parameters))
        object.__setattr__(self, "training_config", _freeze_json(dict(self.training_config)))
        object.__setattr__(self, "runtime_config", _freeze_json(dict(self.runtime_config)))


def _spec_from_dict(kind: str, raw: Mapping[str, Any]):
    schema_raw = raw["schema"]
    layout_raw = raw["layout"]
    schema = SchemaDefinition(
        schema_id=str(schema_raw["schema_id"]),
        tensor_role=str(schema_raw["tensor_role"]),
        dtype=str(schema_raw["dtype"]),
        axes=tuple(str(axis) for axis in schema_raw["axes"]),
        description=str(schema_raw["description"]),
        value_unit=(
            None if schema_raw.get("value_unit") is None else str(schema_raw["value_unit"])
        ),
        attributes={str(key): str(value) for key, value in schema_raw.get("attributes", {}).items()},
    )
    layout = TensorLayout(
        layout_id=str(layout_raw["layout_id"]),
        shape=tuple(int(size) for size in layout_raw["shape"]),
        axis_labels={
            str(axis): tuple(str(label) for label in labels)
            for axis, labels in layout_raw.get("axis_labels", {}).items()
        },
    )
    common = {"schema": schema, "layout": layout}
    if kind == "feature":
        return FeatureSpec(
            **common,
            time_axis=None if raw.get("time_axis") is None else int(raw["time_axis"]),
            sample_period_ns=(
                None if raw.get("sample_period_ns") is None else int(raw["sample_period_ns"])
            ),
            window_alignment=str(raw.get("window_alignment", "right")),
        )
    if kind == "action":
        return ActionSpec(**common)
    if kind == "feedback":
        return FeedbackSpec(**common)
    raise ArtifactSerializationError(f"unknown spec kind: {kind}")


def _spec_to_dict(spec: FeatureSpec | ActionSpec | FeedbackSpec) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema": {
            "schema_id": spec.schema.schema_id,
            "tensor_role": spec.schema.tensor_role,
            "dtype": spec.schema.dtype,
            "axes": list(spec.schema.axes),
            "description": spec.schema.description,
            "value_unit": spec.schema.value_unit,
            "attributes": dict(spec.schema.attributes),
        },
        "layout": {
            "layout_id": spec.layout.layout_id,
            "shape": list(spec.layout.shape),
            "axis_labels": {
                axis: list(labels) for axis, labels in spec.layout.axis_labels.items()
            },
        },
    }
    if isinstance(spec, FeatureSpec):
        result.update(
            {
                "time_axis": spec.time_axis,
                "sample_period_ns": spec.sample_period_ns,
                "window_alignment": spec.window_alignment,
            }
        )
    return result


def _manifest(parameters: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "sha256": _array_sha256(array),
        }
        for name, array in sorted(parameters.items())
    }


def _metadata(artifact: ModelArtifact) -> dict[str, Any]:
    return {
        "decoder_api_version": artifact.decoder_api_version,
        "artifact_format_version": artifact.artifact_format_version,
        "model_type": artifact.model_type,
        "model_version": artifact.model_version,
        "implementation_version": artifact.implementation_version,
        "created_at_ns": artifact.created_at_ns,
        "feature_spec": _spec_to_dict(artifact.feature_spec),
        "action_spec": _spec_to_dict(artifact.action_spec),
        "feedback_spec": (
            None if artifact.feedback_spec is None else _spec_to_dict(artifact.feedback_spec)
        ),
        "training_config": _thaw_json(artifact.training_config),
        "runtime_config": _thaw_json(artifact.runtime_config),
        "provenance": asdict(artifact.provenance),
        "parameter_manifest": _manifest(artifact.parameters),
    }


def _artifact_format_version_from_metadata(metadata: Mapping[str, Any]) -> str:
    """Read the current key while preserving load compatibility with draft v2 files."""

    current = metadata.get("artifact_format_version")
    legacy = metadata.get("artifact_schema_version")
    if current is not None and legacy is not None and str(current) != str(legacy):
        raise ArtifactSerializationError("conflicting artifact format version fields")
    if current is not None:
        return str(current)
    if legacy is not None:
        return str(legacy)
    raise ArtifactSerializationError("artifact_format_version is required")


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def save_artifact(artifact: ModelArtifact, directory: str | Path) -> Path:
    """Validate and atomically publish an artifact directory."""

    from .registry import validate_registered_artifact

    validate_registered_artifact(artifact)
    target = Path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    backup: Path | None = None
    try:
        parameters_path = temporary / "parameters.npz"
        with parameters_path.open("wb") as handle:
            np.savez_compressed(handle, **artifact.parameters)
            handle.flush()
            os.fsync(handle.fileno())

        metadata_path = temporary / "artifact.json"
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(_metadata(artifact), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        if target.exists() or target.is_symlink():
            backup = target.parent / f".{target.name}.backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(temporary, target)
        except Exception:
            if backup is not None and backup.exists():
                os.replace(backup, target)
            raise
        if backup is not None:
            _remove_path(backup)
        return target
    except Exception as exc:
        _remove_path(temporary)
        if isinstance(exc, (ArtifactValidationError, ArtifactSerializationError)):
            raise
        raise ArtifactSerializationError(str(exc)) from exc


def load_artifact(directory: str | Path) -> ModelArtifact:
    """Load, verify, and model-validate a published artifact directory."""

    target = Path(directory)
    try:
        metadata = json.loads((target / "artifact.json").read_text(encoding="utf-8"))
        artifact_format_version = _artifact_format_version_from_metadata(metadata)
        if _major(artifact_format_version) != SUPPORTED_ARTIFACT_FORMAT_MAJOR:
            raise UnsupportedArtifactVersionError(artifact_format_version)

        manifest = metadata["parameter_manifest"]
        with np.load(target / "parameters.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(manifest):
                raise ArtifactSerializationError("parameter manifest does not match parameters.npz")
            parameters: dict[str, np.ndarray] = {}
            for name in archive.files:
                array = np.array(archive[name], copy=True)
                entry = manifest[name]
                if list(array.shape) != list(entry["shape"]):
                    raise ArtifactSerializationError(f"parameter {name} has the wrong shape")
                if array.dtype.str != str(entry["dtype"]):
                    raise ArtifactSerializationError(f"parameter {name} has the wrong dtype")
                if _array_sha256(array) != str(entry["sha256"]):
                    raise ArtifactSerializationError(f"parameter {name} checksum mismatch")
                parameters[name] = array

        provenance = metadata["provenance"]
        artifact = ModelArtifact(
            decoder_api_version=str(metadata["decoder_api_version"]),
            artifact_format_version=artifact_format_version,
            model_type=str(metadata["model_type"]),
            model_version=str(metadata["model_version"]),
            implementation_version=str(metadata["implementation_version"]),
            created_at_ns=int(metadata["created_at_ns"]),
            feature_spec=_spec_from_dict("feature", metadata["feature_spec"]),
            action_spec=_spec_from_dict("action", metadata["action_spec"]),
            feedback_spec=(
                None
                if metadata["feedback_spec"] is None
                else _spec_from_dict("feedback", metadata["feedback_spec"])
            ),
            parameters=parameters,
            training_config=metadata["training_config"],
            runtime_config=metadata["runtime_config"],
            provenance=TrainingProvenance(
                source_ids=tuple(str(item) for item in provenance["source_ids"]),
                sample_count=int(provenance["sample_count"]),
                sequence_count=int(provenance["sequence_count"]),
                training_code_version=str(provenance["training_code_version"]),
                training_config_hash=str(provenance["training_config_hash"]),
                parent_model_version=(
                    None
                    if provenance.get("parent_model_version") is None
                    else str(provenance["parent_model_version"])
                ),
            ),
        )
        from .registry import validate_registered_artifact

        validate_registered_artifact(artifact)
        return artifact
    except (UnsupportedArtifactVersionError, ArtifactSerializationError, ArtifactValidationError):
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactSerializationError(str(exc)) from exc
