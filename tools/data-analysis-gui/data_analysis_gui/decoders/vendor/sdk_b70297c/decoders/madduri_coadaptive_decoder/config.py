# Vendored from github.com/sillybun/bci-decoder-sdk
# commit: b70297c092157a0627fa8d5f97f1694d6043e2bc
# original path: src/decoders/madduri_coadaptive_decoder/config.py
# original git blob: 24b69bac3a0f83b4238e113f527e0f51203d3e61
# Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> `from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.
# --- end vendor header ---
"""Explicit algorithm settings, independent of descriptive tensor metadata."""

from dataclasses import dataclass

import numpy as np

STATE_UPDATE_PERIOD_NS = 20_000_000
INIT_SIGNS = ("positive", "negative")


def positive_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def finite_real(value: object, *, name: str, positive: bool = False) -> float:
    if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or (value <= 0 if positive else value < 0)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return float(value)


@dataclass(frozen=True)
class MadduriFitConfig:
    """Random initialization; the seed fit consumes an empty interaction block."""

    random_seed: int = 0
    init_sign: str = "positive"
    scale: float = 1e-2

    def __post_init__(self) -> None:
        positive_integer(self.random_seed, name="random_seed", minimum=0)
        if self.init_sign not in INIT_SIGNS:
            raise ValueError(f"init_sign must be one of {INIT_SIGNS}")
        finite_real(self.scale, name="scale", positive=True)


@dataclass(frozen=True)
class MadduriAdaptConfig:
    """Ridge fit and blend; learning_rate is the retained parent weight alpha."""

    learning_rate: float = 0.5
    penalty: float = 1e2
    minimum_samples: int = 4
    fit_intercept: bool = False

    def __post_init__(self) -> None:
        alpha = finite_real(self.learning_rate, name="learning_rate", positive=True)
        if alpha >= 1:
            raise ValueError("learning_rate must lie in the open interval (0, 1)")
        finite_real(self.penalty, name="penalty")
        positive_integer(self.minimum_samples, name="minimum_samples", minimum=2)
        if not isinstance(self.fit_intercept, bool):
            raise ValueError("fit_intercept must be a bool")
