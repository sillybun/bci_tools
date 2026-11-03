# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: 8d79df0475b85ca859a02a8268f5262358a22afa
# original path: src/bci_decoder_api/errors.py
# original git blob: a1aa10f0edcb84b022e447e489f396d4c83001a2
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Typed errors for the decoder SDK."""


class DecoderError(Exception):
    """Base class for all SDK-defined errors."""


class SpecValidationError(DecoderError):
    """A tensor or semantic specification is invalid."""


class ArtifactError(DecoderError):
    """Base class for artifact construction and persistence errors."""


class ArtifactSerializationError(ArtifactError):
    """An artifact cannot be written or read safely."""


class ArtifactValidationError(ArtifactError):
    """An artifact does not satisfy its declared model contract."""


class UnsupportedArtifactVersionError(ArtifactError):
    """The artifact envelope version is not supported."""


class ModelTypeMismatchError(ArtifactValidationError):
    """The artifact model type does not match a runtime implementation."""


class ParameterValidationError(ArtifactValidationError):
    """Model-specific parameter validation failed."""


class RuntimeRegistrationError(DecoderError):
    """Base class for runtime registry errors."""


class DuplicateModelTypeError(RuntimeRegistrationError):
    """A model type is already registered."""


class UnknownModelTypeError(RuntimeRegistrationError):
    """No runtime implementation is registered for a model type."""


class RuntimeStateError(DecoderError):
    """A checkpoint cannot be restored into a runtime."""


class TrainingError(DecoderError):
    """Base class for expected offline-training failures."""


class TrainingDataError(TrainingError):
    """Training data violates a model-specific contract."""


class TrainingConfigError(TrainingError):
    """Training configuration is invalid or incomplete."""


class TrainingSchemaError(TrainingError):
    """Training data and declared schemas are incompatible."""


class InsufficientDataError(TrainingError):
    """There is not enough valid data to fit or adapt a model."""


class TrainingNumericalError(TrainingError):
    """Training failed because a numerical result was unusable."""
