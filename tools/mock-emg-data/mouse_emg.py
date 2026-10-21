"""Mouse motion and direction-coded AC amplitude source for Bipolar-32."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

import numpy as np

from packet import PACKET_SAMPLES
from montage import Bipolar32Montage, SAGA_64_CHANNEL_LABELS

CARRIER_HZ = 80.0
REST_AMPLITUDE = 5.0
SAMPLE_RATE_HZ = 2000.0


@dataclass(frozen=True)
class MouseSourceConfig:
    channels: int = 64
    mapping_norm: float = 100.0
    noise_level: float = 5.0
    return_tau: float = 1.0
    seed: int = 42
    source_id: str = "mouse_mock_emg_raw"

    def validate(self):
        if isinstance(self.channels, bool) or not isinstance(self.channels, int) or self.channels != 64:
            raise ValueError("Bipolar-32 mouse source requires exactly 64 channels")
        for name in ("mapping_norm", "noise_level", "return_tau"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.return_tau == 0:
            raise ValueError("return_tau must be > 0")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        return self


class SpringBall:
    """Thread-safe normalized displacement; free motion is an exact exponential.

    Right/up are positive. During a drag, position is held even if the mouse
    stops moving. Release/focus loss starts decay from the current position.
    """

    def __init__(self, tau=1.0, *, clock=time.monotonic):
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and > 0")
        self.tau = tau
        self.clock = clock
        self._lock = threading.Lock()
        self._v = np.zeros(2)
        self._time = clock()
        self._held = False

    def _at(self, now):
        if self._held:
            return self._v.copy()
        return self._v * math.exp(-max(0.0, now - self._time) / self.tau)

    def sample(self):
        with self._lock:
            return self._at(self.clock())

    def set_tau(self, tau):
        """Change the return speed without jumping to a different position."""
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be finite and > 0")
        with self._lock:
            now = self.clock()
            self._v = self._at(now)
            self._time = now
            self.tau = tau

    @property
    def held(self):
        with self._lock:
            return self._held

    def drag(self, position):
        v = np.asarray(position, dtype=float)
        if v.shape != (2,) or not np.isfinite(v).all():
            raise ValueError("position must contain two finite values")
        radius = float(np.linalg.norm(v))
        v = v / max(1.0, radius)
        with self._lock:
            self._v = v.copy()
            self._time = self.clock()
            self._held = True

    def release(self):
        with self._lock:
            now = self.clock()
            self._v = self._at(now)
            self._time = now
            self._held = False

    def center(self):
        with self._lock:
            self._v = np.zeros(2)
            self._time = self.clock()
            self._held = False


class AmplitudeMouseEmg:
    """Four directional groups modulate continuous 80 Hz bipolar carriers.

    The nonnegative 32x4 amplitude map has Frobenius norm mapping_norm.
    Each anatomical pair belongs to right/left/up/down in repeating order.
    Electrode noise is independent; baseline carriers remain at zero intention.
    """

    def __init__(self, config: MouseSourceConfig):
        self.config = config.validate()
        phase_seed, noise_seed = np.random.SeedSequence(config.seed).spawn(2)
        matrix = np.eye(4)[np.arange(32) % 4]
        self.matrix = matrix * (config.mapping_norm / np.linalg.norm(matrix, "fro"))
        self.matrix.setflags(write=False)
        self.phases = np.random.default_rng(phase_seed).uniform(0, 2 * np.pi, 32)
        self.montage = Bipolar32Montage.from_channel_labels(SAGA_64_CHANNEL_LABELS)
        self.rng = np.random.default_rng(noise_seed)
        self.sample_index = 0

    def packet(self, displacement, *, sample_index=None):
        v = np.asarray(displacement, dtype=float)
        if v.shape != (2,) or not np.isfinite(v).all():
            raise ValueError("displacement must contain two finite values")
        if sample_index is None:
            sample_index = self.sample_index
        if (isinstance(sample_index, bool) or not isinstance(sample_index, (int, np.integer))
                or sample_index < self.sample_index):
            raise ValueError("sample_index must be an integer at or after the next sample")
        x, y = v
        activation = np.maximum([x, -x, y, -y], 0)
        amplitude = REST_AMPLITUDE + self.matrix @ activation
        # Use the sender's sample grid, including skipped slots. Never restart
        # the carrier at a packet boundary or fabricate a missing raw packet.
        indices = sample_index + np.arange(PACKET_SAMPLES)
        phase = 2 * np.pi * CARRIER_HZ * (indices / SAMPLE_RATE_HZ)
        pair_signal = amplitude[:, None] * np.sin(phase[None, :] + self.phases[:, None])
        noise = self.rng.standard_normal((self.config.channels, PACKET_SAMPLES))
        with np.errstate(over="ignore", invalid="ignore"):
            raw = self.config.noise_level * noise
            raw[list(self.montage.positive_indices)] += pair_signal / 2
            raw[list(self.montage.negative_indices)] -= pair_signal / 2
            raw = raw.astype(np.float32)
        if not np.isfinite(raw).all():
            raise ValueError("signal exceeds float32 range; reduce mapping norm or noise level")
        self.sample_index = int(sample_index) + PACKET_SAMPLES
        return raw
