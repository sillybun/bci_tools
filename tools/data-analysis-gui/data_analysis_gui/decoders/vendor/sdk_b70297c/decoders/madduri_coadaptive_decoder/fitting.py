# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/madduri_coadaptive_decoder/fitting.py
# original git blob: 8dc43374a3ff3638de24d76b3ee9b6fe3ab09478
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Random seed decoders, SmoothBatch ridge fitting, and the Artifact validator.

Online inference uses the Artifact-compatible map ``v_hat = D u + b``. New
training defaults to the paper's linear decoder with ``b = 0``; the affine form
remains available for legacy replay. The co-adaptive part lives in the offline
SDK ``adapt()`` path: every ~20 s the optimal decoder ``D*`` is the ridge
minimizer of the decoder cost (eq. 11) and the next decoder is the SmoothBatch
blend (eq. 9). Because the cost is a convex quadratic in ``D``, ``D*`` has a
closed form and needs only NumPy.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np

from ...bci_decoder_api import ModelArtifact

from .config import INIT_SIGNS as INIT_SIGNS
from .config import MadduriFitConfig, positive_integer, finite_real

MODEL_TYPE = "madduri_smoothbatch"
IMPLEMENTATION_VERSION = "2.0.0"

__all__ = [
    "MODEL_TYPE", "IMPLEMENTATION_VERSION", "INIT_SIGNS", "random_seed_decoder",
    "fit_optimal_decoder", "smoothbatch", "limit_speed", "parameter_fingerprint",
    "validate_madduri_artifact",
]


def random_seed_decoder(
        *,
        channel_count: int,
        init_sign: str,
        scale: float,
        random_seed: int,
        action_dimension: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a randomly initialized decoder (paper: per-trial random D, eq. Methods).

    ``positive`` draws ``D`` uniformly in ``[0, scale]``; ``negative`` in
    ``[-scale, 0]``. The Artifact-compatible offset ``b`` starts at zero and
    remains zero under the default no-intercept adaptation mode.
    """

    MadduriFitConfig(random_seed=random_seed, init_sign=init_sign, scale=scale)
    positive_integer(channel_count, name="channel_count")
    positive_integer(action_dimension, name="action_dimension")
    rng = np.random.default_rng(random_seed)
    magnitude = scale * rng.random((action_dimension, channel_count))
    weights = magnitude if init_sign == "positive" else -magnitude
    intercept = np.zeros(action_dimension, dtype=np.float64)
    return weights.astype(np.float64), intercept


def fit_optimal_decoder(
        features: np.ndarray,
        desired_velocity: np.ndarray,
        *,
        penalty: float,
        fit_intercept: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge minimizer of the decoder cost.

    By default this minimizes ``||D u - v||^2 + penalty ||D||^2`` and returns
    an exactly-zero intercept, matching the paper's linear decoder. Setting
    ``fit_intercept=True`` preserves the legacy affine fit: the design and
    target are centered, and the unpenalized offset is recovered afterward.
    """

    design = np.asarray(features, dtype=np.float64)
    target = np.asarray(desired_velocity, dtype=np.float64)
    channel_count = design.shape[1]

    if fit_intercept:
        feature_mean = np.mean(design, axis=0)
        target_mean = np.mean(target, axis=0)
        fit_design = design - feature_mean
        fit_target = target - target_mean
    else:
        feature_mean = np.zeros(channel_count, dtype=np.float64)
        target_mean = np.zeros(target.shape[1], dtype=np.float64)
        fit_design = design
        fit_target = target

    gram = fit_design.T @ fit_design + penalty * np.eye(channel_count)
    cross = fit_design.T @ fit_target
    try:
        coefficients = np.linalg.solve(gram, cross)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(gram) @ cross
    weights = coefficients.T
    intercept = target_mean - weights @ feature_mean
    return weights, intercept


def smoothbatch(
        prior_weights: np.ndarray,
        prior_intercept: np.ndarray,
        optimal_weights: np.ndarray,
        optimal_intercept: np.ndarray,
        *,
        learning_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    """SmoothBatch blend (eq. 9): ``alpha`` weights the prior decoder."""

    alpha = float(learning_rate)
    weights = alpha * prior_weights + (1.0 - alpha) * optimal_weights
    intercept = alpha * prior_intercept + (1.0 - alpha) * optimal_intercept
    return weights, intercept


def limit_speed(velocity: np.ndarray, maximum_speed: float) -> np.ndarray:
    """Clamp each velocity's magnitude to ``maximum_speed``, preserving direction."""

    values = np.asarray(velocity, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("velocity must contain finite, non-empty vectors")
    finite_real(maximum_speed, name="maximum_speed", positive=True)
    # Normalize before computing the norm to avoid overflow for large vectors.
    largest = np.max(np.abs(values), axis=-1, keepdims=True)
    direction = values / np.where(largest == 0, 1.0, largest)
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    allowed = maximum_speed / np.where(norm == 0, 1.0, norm)
    return direction * np.minimum(largest, allowed)


def parameter_fingerprint(
        parameters: Mapping[str, np.ndarray],
        *,
        payload: Mapping[str, object],
) -> str:
    """Short deterministic digest of parameters plus salient training payload."""

    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, array in sorted(parameters.items()):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()[:12]


def validate_madduri_artifact(artifact: ModelArtifact) -> None:
    """Compatibility entry point for the shared lifecycle validator."""
    from .validation import validate_madduri_artifact as validate

    validate(artifact)
