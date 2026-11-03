# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/bci_decoder_api/registry.py
# original git blob: 06cf62a80e7f3db84fc3ee1d577e4c507b9996e5
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Explicit DecoderCore registration and SDK-owned Runtime construction."""

from __future__ import annotations

import re
from threading import RLock
from typing import TypeVar

from .artifact import ModelArtifact
from .base import DecoderCore
from .errors import DuplicateModelTypeError, RuntimeRegistrationError, UnknownModelTypeError
from .runtime import DecoderRuntime


CoreT = TypeVar("CoreT", bound=type[DecoderCore])
_MODEL_TYPE_RE = re.compile(r"^[a-z0-9_]+$")
_registry: dict[str, type[DecoderCore]] = {}
_registry_lock = RLock()


def register_decoder_core(core_type: CoreT) -> CoreT:
    """Register a model-specific Core class and reject silent replacement."""

    if not isinstance(core_type, type) or not issubclass(core_type, DecoderCore):
        raise RuntimeRegistrationError("registered core must subclass DecoderCore")
    model_type = getattr(core_type, "model_type", "")
    implementation_version = getattr(core_type, "implementation_version", "")
    if not isinstance(model_type, str) or not _MODEL_TYPE_RE.fullmatch(model_type):
        raise RuntimeRegistrationError("core model_type is invalid")
    if not isinstance(implementation_version, str) or not implementation_version:
        raise RuntimeRegistrationError("core implementation_version is required")
    with _registry_lock:
        if model_type in _registry:
            raise DuplicateModelTypeError(model_type)
        _registry[model_type] = core_type
    return core_type


def get_core_type(model_type: str) -> type[DecoderCore]:
    with _registry_lock:
        try:
            return _registry[model_type]
        except KeyError as exc:
            raise UnknownModelTypeError(model_type) from exc


def registered_model_types() -> tuple[str, ...]:
    with _registry_lock:
        return tuple(sorted(_registry))


def validate_registered_artifact(artifact: ModelArtifact) -> None:
    core_type = get_core_type(artifact.model_type)
    core_type.validate_artifact(artifact)


def create_runtime(artifact: ModelArtifact) -> DecoderRuntime:
    """Create one public Runtime around a fresh model-specific Core."""

    core_type = get_core_type(artifact.model_type)
    core_type.validate_artifact(artifact)
    core = core_type.from_artifact(artifact)
    return DecoderRuntime(artifact, core)
