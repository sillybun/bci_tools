from __future__ import annotations

import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from mouse_emg import AmplitudeMouseEmg, MouseSourceConfig, SpringBall
from app import MockMouseEmgWindow, MousePublisher, parse_args
from packet import PacketDecoder


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def test_hold_release_monotonic_return_and_center():
    now = [0.0]
    ball = SpringBall(1.0, clock=lambda: now[0])
    ball.drag((0.4, -0.3))
    now[0] = 100.0
    np.testing.assert_array_equal(ball.sample(), [0.4, -0.3])
    ball.release()
    for elapsed in (0.1, 0.5, 1.0, 3.0, 10.0):
        now[0] = 100 + elapsed
        np.testing.assert_allclose(ball.sample(), np.array([0.4, -0.3]) * np.exp(-elapsed))
        assert ball.sample()[0] > 0 and ball.sample()[1] < 0
    ball.drag((3, 4))
    np.testing.assert_allclose(ball.sample(), [0.6, 0.8])
    ball.center()
    np.testing.assert_array_equal(ball.sample(), [0, 0])
    assert not ball.held


def test_changing_return_time_preserves_position_and_hold():
    now = [0.0]
    ball = SpringBall(1.0, clock=lambda: now[0])
    ball.drag([0.4, 0.3])
    ball.set_tau(2.0)
    now[0] = 10.0
    np.testing.assert_array_equal(ball.sample(), [0.4, 0.3])
    assert ball.held
    ball.release()
    now[0] = 12.0
    before = ball.sample()
    ball.set_tau(0.5)
    np.testing.assert_array_equal(ball.sample(), before)
    now[0] = 12.5
    np.testing.assert_allclose(ball.sample(), before * np.exp(-1))
    with pytest.raises(ValueError):
        ball.set_tau(0)
    assert ball.tau == 0.5


def test_continuous_carrier_bipolar_mapping_and_wire_roundtrip():
    generator = AmplitudeMouseEmg(MouseSourceConfig(mapping_norm=80, noise_level=0))
    assert np.linalg.norm(generator.matrix, "fro") == pytest.approx(80)
    assert np.linalg.matrix_rank(generator.matrix) == 4
    raw = generator.packet([0.4, 0.3])
    assert raw.shape == (64, 40) and raw.dtype == np.float32
    decoded = PacketDecoder(64).decode(raw.ravel(order="C"), 100)
    np.testing.assert_array_equal(decoded.raw_ct, raw)
    np.testing.assert_allclose(np.diff(decoded.sample_timestamps), 0.0005)
    # A full second assembled from 50 packets must be a continuous 80 Hz
    # carrier; resetting phase every 40 points produces a different spectrum.
    packets = [raw] + [generator.packet([0.4, 0.3]) for _ in range(49)]
    pairs = generator.montage.difference(np.concatenate(packets, axis=1))
    spectrum = np.abs(np.fft.rfft(pairs, axis=1))
    np.testing.assert_array_equal(spectrum.argmax(axis=1), np.full(32, 80))
    amplitude = spectrum[:, 80] / 1000
    expected = 5 + generator.matrix @ [0.4, 0, 0.3, 0]
    np.testing.assert_allclose(amplitude, expected, rtol=1e-6)
    positive, negative = generator.montage.positive_indices, generator.montage.negative_indices
    np.testing.assert_array_equal(raw[list(positive)], -raw[list(negative)])


def test_noise_statistics_independence_and_reproducibility():
    config = MouseSourceConfig(noise_level=5)
    a, b = AmplitudeMouseEmg(config), AmplitudeMouseEmg(config)
    noiseless = AmplitudeMouseEmg(MouseSourceConfig(noise_level=0))
    samples = []
    for _ in range(100):
        packet = a.packet([0, 0])
        np.testing.assert_array_equal(packet, b.packet([0, 0]))
        samples.append(packet - noiseless.packet([0, 0]))
    noise = np.concatenate(samples, axis=1)
    assert abs(noise.mean()) < 0.05
    assert noise.std() == pytest.approx(5, abs=0.05)
    assert abs(np.corrcoef(noise[0], noise[1])[0, 1]) < 0.06
    assert abs(np.corrcoef(noise[0, :-1], noise[0, 1:])[0, 1]) < 0.06
    zero_map = AmplitudeMouseEmg(MouseSourceConfig(mapping_norm=0, noise_level=0))
    rest_only = AmplitudeMouseEmg(MouseSourceConfig(mapping_norm=0, noise_level=0))
    rest = zero_map.packet([1, 0])
    np.testing.assert_array_equal(rest, rest_only.packet([0, 0]))
    assert np.std(rest) > 0


def test_carrier_keeps_physical_phase_across_skipped_slots():
    complete, skipped = [AmplitudeMouseEmg(MouseSourceConfig(noise_level=0)) for _ in range(2)]
    complete.packet([1, 0])
    complete.packet([1, 0])
    expected = complete.packet([1, 0])
    skipped.packet([1, 0])
    np.testing.assert_array_equal(skipped.packet([1, 0], sample_index=80), expected)
    with pytest.raises(ValueError, match="sample_index"):
        skipped.packet([1, 0], sample_index=40)


@pytest.mark.parametrize("args", [
    ["--noise-level", "-1"], ["--noise-level", "nan"],
    ["--mapping-norm", "inf"], ["--mapping-norm", "-1"],
    ["--return-tau", "0"], ["--channels", "1"], ["--seed", "-2"],
    ["--source-id", " "], ["--channels", "32"],
])
def test_invalid_cli_rejected(args):
    with pytest.raises(SystemExit) as exc:
        parse_args(args)
    assert exc.value.code == 2


class Outlet:
    def __init__(self, delay=0, fail=False):
        self.samples = []
        self.closed = False
        self.delay = delay
        self.fail = fail

    def push_sample(self, sample, timestamp):
        if self.fail:
            raise RuntimeError("test outlet failure")
        self.samples.append((sample, timestamp))
        time.sleep(self.delay)

    def close(self):
        self.closed = True


def advancing_lsl_clock():
    start = time.monotonic()
    return lambda: 1000 + time.monotonic() - start


def test_publisher_sleep_clock_jump_keeps_new_packets_fresh():
    # LSL advances across sleep while Python's scheduling clock may pause.
    now = [1000.0]
    outlet = Outlet()
    received_at = []

    class SleepEvent:
        def is_set(self):
            return len(outlet.samples) >= 6

        def wait(self, timeout):
            now[0] += timeout
            if len(outlet.samples) == 3:
                now[0] += 5260.8
            received_at.append(now[0])
            return False

    worker = MousePublisher(MouseSourceConfig(noise_level=0), SpringBall(),
                            outlet_factory=lambda _: outlet, lsl_clock=lambda: now[0])
    worker.stop_event = SleepEvent()
    worker.run()
    assert outlet.closed
    assert len(outlet.samples) == 6
    ticks = np.array([timestamp for _, timestamp in outlet.samples])
    ages = np.array(received_at) - ticks
    assert np.all(ages < 0.020 + 1e-8)
    assert np.all(ages >= -1e-8)
    assert ticks[3] - ticks[2] >= 5260.8
    assert np.all(np.diff(ticks) >= 0.020 - 1e-8)
    reference = AmplitudeMouseEmg(MouseSourceConfig(noise_level=0))
    for sample, timestamp in outlet.samples:
        slot = round((timestamp - 1000) / 0.02)
        expected = reference.packet([0, 0], sample_index=(slot - 1) * 40)
        np.testing.assert_array_equal(np.asarray(sample).reshape(64, 40), expected)


def test_publisher_timing_payload_and_shutdown(app):
    outlet = Outlet()
    ball = SpringBall()
    ball.drag([0.4, 0.3])
    config = MouseSourceConfig(noise_level=0)
    clock = advancing_lsl_clock()
    epoch = []
    def record_clock():
        now = clock()
        if not epoch:
            epoch.append(now)
        return now
    worker = MousePublisher(config, ball, outlet_factory=lambda _: outlet, lsl_clock=record_clock)
    worker.start()
    try:
        wait_until(app, lambda: len(outlet.samples) >= 10)
    finally:
        worker.request_stop()
        assert worker.wait(2000)
    assert outlet.closed
    reference = AmplitudeMouseEmg(config)
    decoder = PacketDecoder(64)
    for sample, timestamp in outlet.samples:
        slot = round((timestamp - epoch[0]) / 0.02)
        expected = reference.packet([0.4, 0.3], sample_index=(slot - 1) * 40)
        np.testing.assert_array_equal(decoder.decode(sample, timestamp).raw_ct, expected)
    ticks = np.array([timestamp for _, timestamp in outlet.samples])
    assert np.all(np.diff(ticks) > 0)
    # Grid remains exact even if a loaded CI machine skips a slot.
    np.testing.assert_allclose((ticks - epoch[0]) / 0.02, np.rint((ticks - epoch[0]) / 0.02), atol=1e-8)


def test_delayed_sender_exposes_gaps_instead_of_bursting(app):
    outlet = Outlet(delay=0.07)
    reports = []
    worker = MousePublisher(MouseSourceConfig(), SpringBall(), outlet_factory=lambda _: outlet, lsl_clock=advancing_lsl_clock())
    worker.progress.connect(lambda *args: reports.append(args))
    worker.start()
    try:
        wait_until(app, lambda: len(outlet.samples) >= 5)
    finally:
        worker.request_stop()
        assert worker.wait(2000)
        app.processEvents()
    assert reports[-1][2] > 0
    assert np.max(np.diff([timestamp for _, timestamp in outlet.samples])) >= 0.06 - 1e-8


def test_qt_drag_release_focus_center_and_layout(app):
    window = MockMouseEmgWindow(MouseSourceConfig(), auto_start=False)
    window.show()
    app.processEvents()
    try:
        canvas = window.canvas
        center, radius = canvas.geometry_values()
        QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=center.toPoint())
        destination = center.toPoint() + QPoint(round(0.4 * radius), round(-0.3 * radius))
        QTest.mouseMove(canvas, destination)
        assert window.ball.held
        np.testing.assert_allclose(window.ball.sample(), [0.4, 0.3], atol=0.006)
        QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=destination)
        before = np.linalg.norm(window.ball.sample())
        QTest.qWait(80)
        assert np.linalg.norm(window.ball.sample()) < before
        window.ball.drag([0.4, 0.3])
        app.sendEvent(canvas, QEvent(QEvent.Type.FocusOut))
        assert not window.ball.held
        QTest.mouseClick(window.center_button, Qt.MouseButton.LeftButton)
        np.testing.assert_array_equal(window.ball.sample(), [0, 0])
        for widget in window.displacement_labels:
            assert widget.width() >= widget.fontMetrics().horizontalAdvance(widget.text())
        window.resize(1040, 760)
        app.processEvents()
        assert window.center_button.height() >= window.center_button.fontMetrics().height() + 16
        button_bottom = window.stop_button.mapTo(window, window.stop_button.rect().bottomRight())
        assert 0 < button_bottom.y() < window.height()
    finally:
        window.close()


def test_window_error_stop_restart_and_close(app):
    outlets = []

    def factory(config, ball):
        outlet = Outlet(fail=not outlets)
        outlets.append(outlet)
        return MousePublisher(config, ball, outlet_factory=lambda _: outlet, lsl_clock=advancing_lsl_clock())

    window = MockMouseEmgWindow(MouseSourceConfig(), publisher_factory=factory)
    window.show()
    try:
        wait_until(app, lambda: window.stop_button.text() == "Start Stream")
        assert "test outlet failure" in window.footer_status.text()
        assert outlets[0].closed
        QTest.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
        wait_until(app, lambda: len(outlets) == 2 and len(outlets[1].samples) >= 2)
        QTest.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
        wait_until(app, lambda: window.stop_button.text() == "Start Stream")
        assert "Stopped" in window.header_status.text()
        QTest.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
        wait_until(app, lambda: len(outlets) == 3 and len(outlets[2].samples) >= 2)
    finally:
        window.close()
    assert not window.publisher.isRunning()
    assert all(outlet.closed for outlet in outlets)


def edit_parameters(window, **values):
    for key, value in values.items():
        window.parameter_inputs[key].setText(str(value))


def test_apply_while_stopped_updates_next_start_only(app):
    configs = []
    outlet = Outlet()

    def factory(config, ball):
        configs.append(config)
        return MousePublisher(config, ball, outlet_factory=lambda _: outlet, lsl_clock=advancing_lsl_clock())

    initial = MouseSourceConfig(mapping_norm=80, noise_level=2, return_tau=3, seed=7, source_id="initial")
    window = MockMouseEmgWindow(initial, auto_start=False, publisher_factory=factory)
    try:
        for key, field in window.parameter_inputs.items():
            assert field.text() == str(getattr(initial, key))
        field = window.parameter_inputs["mapping_norm"]
        field.selectAll()
        QTest.keyClicks(field, "200")
        assert "Unapplied changes" in window.parameters_note.text()
        assert window.config == initial
        edit_parameters(window, mapping_norm=200, noise_level=0, return_tau=0.25, seed=123,
                        source_id="  changed  ")
        window.apply_button.click()
        expected = MouseSourceConfig(mapping_norm=200, noise_level=0, return_tau=0.25,
                                     seed=123, source_id="changed")
        assert window.config == expected
        assert window.ball.tau == 0.25
        assert window.publisher is None and not configs
        assert window._status_name == "Stopped"
        assert window.parameter_inputs["source_id"].text() == "changed"
        window.stop_button.click()
        wait_until(app, lambda: len(outlet.samples) >= 2)
        assert configs == [expected]
    finally:
        window.close()
    assert outlet.closed


@pytest.mark.parametrize("key,value", [
    ("mapping_norm", "-1"), ("mapping_norm", "inf"),
    ("noise_level", "nan"), ("noise_level", "abc"),
    ("return_tau", "0"), ("return_tau", "-2"), ("return_tau", ""),
    ("seed", "1.5"), ("seed", "-1"), ("source_id", " "),
])
def test_invalid_parameter_edits_leave_applied_config_unchanged(app, key, value):
    config = MouseSourceConfig()
    window = MockMouseEmgWindow(config, auto_start=False)
    try:
        edit_parameters(window, **{key: value})
        window.apply_button.click()
        assert window.config == config
        assert window.ball.tau == config.return_tau
        assert window.publisher is None
        assert "#cf2639" in window.parameters_note.styleSheet()
        assert window.apply_button.isEnabled()
    finally:
        window.close()


def test_apply_while_streaming_replaces_outlet_and_changes_payload(app):
    outlets, configs, epochs = [], [], []

    def factory(config, ball):
        assert all(outlet.closed for outlet in outlets), "The previous outlet must close first"
        outlet = Outlet()
        epoch = []
        clock = advancing_lsl_clock()

        def record_clock():
            now = clock()
            if not epoch:
                epoch.append(now)
            return now

        outlets.append(outlet)
        configs.append(config)
        epochs.append(epoch)
        return MousePublisher(config, ball, outlet_factory=lambda _: outlet, lsl_clock=record_clock)

    window = MockMouseEmgWindow(MouseSourceConfig(), publisher_factory=factory)
    window.ball.drag([0.4, 0.3])
    try:
        wait_until(app, lambda: outlets and len(outlets[0].samples) >= 2)
        window.apply_button.click()  # No change should not interrupt the stream.
        assert window._pending_config is None and not window.publisher.stop_event.is_set()
        edit_parameters(window, noise_level="invalid")
        window.apply_button.click()
        assert not window.publisher.stop_event.is_set()
        assert window._status_name == "Streaming"
        edit_parameters(window, mapping_norm=240, noise_level=2, return_tau=0.2, seed=88,
                        source_id="new-source")
        window.apply_button.click()
        assert not window.apply_button.isEnabled() and not window.stop_button.isEnabled()
        window.start_stream()  # No competing worker while shutdown is pending.
        wait_until(app, lambda: len(outlets) == 2 and len(outlets[1].samples) >= 3)
        assert outlets[0].closed
        assert window.config == MouseSourceConfig(mapping_norm=240, noise_level=2,
                                                  return_tau=0.2, seed=88, source_id="new-source")
        assert configs[1] == window.config and window.ball.tau == 0.2
        assert window.apply_button.isEnabled() and window.stop_button.isEnabled()
        assert window._status_name == "Streaming"
        window.stop_button.click()
        wait_until(app, lambda: window._status_name == "Stopped")
        reference = AmplitudeMouseEmg(window.config)
        for sample, timestamp in outlets[1].samples:
            slot = round((timestamp - epochs[1][0]) / 0.02)
            expected = reference.packet([0.4, 0.3], sample_index=(slot - 1) * 40)
            np.testing.assert_array_equal(np.asarray(sample).reshape(64, 40), expected)
    finally:
        window.close()
    assert len(outlets) == 2 and all(outlet.closed for outlet in outlets)


def test_close_during_apply_does_not_restart_stream(app):
    outlets = []

    def factory(config, ball):
        outlet = Outlet()
        outlets.append(outlet)
        return MousePublisher(config, ball, outlet_factory=lambda _: outlet, lsl_clock=advancing_lsl_clock())

    window = MockMouseEmgWindow(MouseSourceConfig(), publisher_factory=factory)
    try:
        wait_until(app, lambda: outlets and len(outlets[0].samples) >= 2)
        edit_parameters(window, noise_level=0)
        window.apply_button.click()
        assert window._pending_config is not None
    finally:
        window.close()
        app.processEvents()
    assert len(outlets) == 1 and outlets[0].closed
    assert not window.publisher.isRunning()
