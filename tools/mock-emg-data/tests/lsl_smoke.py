"""Check a real local LSL loopback with a unique, short-lived source."""

from pathlib import Path
import sys
import time
import uuid

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import MousePublisher
from mouse_emg import MouseSourceConfig, SpringBall
from packet import PacketDecoder


def main():
    config = MouseSourceConfig(source_id=f"bci-tools-smoke-{uuid.uuid4()}")
    worker = MousePublisher(config, SpringBall())
    inlet = None
    failures = []
    # Direct connection makes publisher errors observable without a Qt event loop.
    from PySide6.QtCore import Qt
    worker.failed.connect(failures.append, Qt.ConnectionType.DirectConnection)
    worker.start()
    try:
        streams = resolve_byprop("source_id", config.source_id, timeout=5)
        assert streams, f"Test stream not discovered: {failures}"
        info = streams[0]
        assert info.name() == "emg_raw_signed_2khz"
        assert info.type() == "EMG"
        assert info.channel_count() == 2560
        assert info.nominal_srate() == 50.0
        inlet = StreamInlet(info)
        decoder = PacketDecoder(64)
        timestamps, ages = [], []
        deadline = time.monotonic() + 10
        while len(timestamps) < 100 and time.monotonic() < deadline:
            sample, timestamp = inlet.pull_sample(timeout=0.25)
            if sample is None:
                continue
            packet = decoder.decode(sample, timestamp)
            assert packet.raw_ct.shape == (64, 40)
            assert packet.raw_ct.dtype == np.float32
            timestamps.append(timestamp)
            ages.append(local_clock() - timestamp)
        assert not failures, failures
        assert len(timestamps) == 100, len(timestamps)
        intervals = np.diff(timestamps)
        assert np.all(intervals > 0)
        np.testing.assert_allclose(intervals / 0.020, np.rint(intervals / 0.020), atol=1e-5)
        assert abs(float(np.median(intervals)) - 0.020) < 1e-5
        assert max(ages) < 1.0, max(ages)
        print(f"LSL loopback passed: 100 packets, width=2560, median interval={np.median(intervals):.3f}s, "
              f"maximum age={max(ages):.3f}s")
    finally:
        if inlet is not None:
            inlet.close_stream()
        worker.request_stop()
        assert worker.wait(3000), "Publisher did not stop"


if __name__ == "__main__":
    main()
