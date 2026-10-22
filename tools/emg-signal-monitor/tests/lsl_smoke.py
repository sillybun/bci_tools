"""Real LSL loopback through the monitor worker, with an isolated test stream."""

from contextlib import contextmanager
from pathlib import Path
import sys
import threading
import time
import uuid

import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raw_stream_monitor import MonitorConfig, RawStreamWorker


def wait_until(qt, predicate, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "Timed out waiting for monitor"


@contextmanager
def sample_source():
    identity = f"emg-monitor-smoke-{uuid.uuid4()}"
    config = MonitorConfig(stream_name=identity, source_id=identity).validated()
    stop = threading.Event()
    ready = threading.Event()
    errors = []

    def publish():
        try:
            outlet = StreamOutlet(StreamInfo(identity, "EMG", 2560, 50, "float32", identity))
            ready.set()
            started = local_clock()
            index = 0
            while not stop.is_set():
                timestamp = started + index * 0.02
                offsets = (np.arange(40) - 39) / 2000
                carrier = np.sin(2 * np.pi * 80 * (timestamp + offsets))
                values = np.arange(1, 65, dtype=np.float32)[:, None] * carrier
                outlet.push_sample(values.astype(np.float32).ravel(order="C").tolist(), timestamp)
                index += 1
                stop.wait(max(0, started + index * 0.02 - local_clock()))
        except Exception as error:
            errors.append(error)
            ready.set()

    thread = threading.Thread(target=publish, daemon=True)
    thread.start()
    try:
        assert ready.wait(5), "Publisher startup timed out"
        assert not errors, errors
        yield config
        assert not errors, errors
    finally:
        stop.set()
        thread.join(3)
        assert not thread.is_alive(), "Publisher did not stop"


def main():
    qt = QApplication.instance() or QApplication([])
    with sample_source() as config:
        worker = RawStreamWorker(config)
        packets, failures = [], []
        worker.batch_received.connect(packets.extend)
        worker.failed.connect(failures.append)
        worker.start()
        try:
            wait_until(qt, lambda: len(packets) >= 100 or failures)
            assert not failures, failures
            assert len(packets) >= 100
            timestamps = np.array([packet.packet_timestamp for packet in packets])
            np.testing.assert_allclose(np.diff(timestamps), .02, atol=1e-6)
            for packet in packets:
                assert packet.raw_ct.shape == (64, 40)
                np.testing.assert_allclose(packet.raw_ct[63], packet.raw_ct[0] * 64)
            assert worker.invalid_total == worker.out_of_order_total == worker.gap_total == 0
            print(f"LSL monitor passed: {len(packets)} packets, width=2560, 20 ms intervals; no invalid/order/gap errors")
        finally:
            worker.request_stop()
            assert worker.wait(5000), "Monitor did not stop"
            qt.processEvents()


if __name__ == "__main__":
    main()
