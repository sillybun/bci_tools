"""Native desktop check; optionally save Qt-rendered screenshots with --output."""

import argparse
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from pylsl import StreamInlet, local_clock, resolve_byprop
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import MockMouseEmgWindow
from mouse_emg import MouseSourceConfig


def wait_until(qt, predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        qt.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    assert predicate(), "Native UI check timed out"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    qt = QApplication.instance() or QApplication([])
    window = MockMouseEmgWindow(MouseSourceConfig(source_id=f"mock-emg-ui-{uuid.uuid4()}"))
    inlet = None
    try:
        window.show()
        wait_until(qt, lambda: window._status_name == "Streaming"
                   and int(window.count_label.text().replace(",", "")) >= 10)
        for key, value in (("mapping_norm", "250"), ("noise_level", "2"),
                           ("return_tau", "0.4"), ("seed", "123")):
            window.parameter_inputs[key].setText(value)
        applied_at = local_clock()
        window.apply_button.click()
        wait_until(qt, lambda: window.config.mapping_norm == 250
                   and window._status_name == "Streaming"
                   and int(window.count_label.text().replace(",", "")) >= 10)
        assert window.config.noise_level == 2 and window.config.seed == 123
        assert window.ball.tau == 0.4
        streams = resolve_byprop("source_id", window.config.source_id, timeout=5)
        assert streams, "Restarted LSL stream not discovered"
        inlet = StreamInlet(streams[0])
        received = []

        def receive():
            sample, timestamp = inlet.pull_sample(timeout=0.0)
            if sample is not None:
                assert len(sample) == 2560 and np.isfinite(sample).all()
                assert timestamp > applied_at
                assert local_clock() - timestamp < 1.0
                received.append(timestamp)
            return len(received) >= 20

        wait_until(qt, receive)
        assert np.all(np.diff(received) > 0)
        if args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            assert window.grab().save(str(args.output / "mock-emg-expanded.png"))
        window.set_sidebar_visible(False)
        window.resize(300, 340)
        qt.processEvents()
        window.ball.drag([0.4, 0.3])
        window.refresh_motion()
        qt.processEvents()
        assert window.canvas.compact
        assert window.width() == 300 and window.height() == 340
        if args.output:
            assert window.grab().save(str(args.output / "mock-emg-compact.png"))
        window.set_sidebar_visible(True)
        qt.processEvents()
        assert not window.canvas.compact
        assert window.publisher.isRunning()
        assert window.parameter_inputs["mapping_norm"].text() == "250"
        print(f"Native Mock EMG UI passed: parameter apply/restart, 20 fresh LSL packets, "
              f"compact mode, and restore; environment={sys.prefix}",
              flush=True)
    finally:
        if inlet is not None:
            inlet.close_stream()
        window.close()
        qt.processEvents()
        if window.publisher is not None:
            assert not window.publisher.isRunning()


if __name__ == "__main__":
    main()
