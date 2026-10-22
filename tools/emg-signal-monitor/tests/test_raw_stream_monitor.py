from __future__ import annotations

import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QThread
from PySide6.QtWidgets import QApplication, QScrollArea

from raw_stream_monitor import (
    DEFAULT_BANDPASS_HIGH_HZ,
    DEFAULT_BANDPASS_LOW_HZ,
    DISPLAY_MODE_BIPOLAR32,
    DISPLAY_MODE_RAW,
    DisplayBandpassConfig,
    MAX_HISTORY_BYTES,
    MonitorConfig,
    OnlineDisplayBandpass,
    RawEmgHistory,
    RawStreamWorker,
    bipolar32_grid_position,
    decimate_minmax,
    parse_channel_labels,
    raw_channel_statistics,
    transform_display_values,
    validate_stream_info,
)
from montage import (
    SAGA_64_CHANNEL_LABELS,
    fixed_pair_labels,
)
from packet import (
    PACKET_SAMPLES,
    PHYSICAL_SAMPLE_RATE_HZ,
    DecodedPacket,
    PacketDecoder,
)


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "app.py"
_SPEC = spec_from_file_location("monitor_raw_emg_stream", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
RawStreamMonitorWindow = _MODULE.RawStreamMonitorWindow
parse_args = _MODULE.parse_args


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeInfo:
    def __init__(
        self, *, channels: int, name: str = "emg_raw_signed_2khz",
        stream_type: str = "EMG", source_id: str = "fake-source",
        channel_format: int = 1, nominal_srate: float = 50.0,
    ) -> None:
        self._channels = channels
        self._name = name
        self._type = stream_type
        self._source_id = source_id
        self._format = channel_format
        self._rate = nominal_srate

    def name(self): return self._name
    def type(self): return self._type
    def source_id(self): return self._source_id
    def channel_count(self): return self._channels
    def channel_format(self): return self._format
    def nominal_srate(self): return self._rate
    def uid(self): return "fake-uid"
    def hostname(self): return "localhost"


def decoded(channel_count: int, packet_timestamp: float, start: float = 0.0):
    values = np.arange(start, start + channel_count * PACKET_SAMPLES, dtype=np.float32)
    return PacketDecoder(channel_count).decode(values, packet_timestamp)


def packet_from_ct(values: np.ndarray, packet_timestamp: float) -> DecodedPacket:
    raw = np.ascontiguousarray(values, dtype=np.float32)
    offsets = np.arange(PACKET_SAMPLES - 1, -1, -1, dtype=np.float64)
    stamps = packet_timestamp - offsets / PHYSICAL_SAMPLE_RATE_HZ
    return DecodedPacket(raw, packet_timestamp, stamps)


def test_config_generates_local_labels_and_strict_contract():
    config = MonitorConfig(channel_count=3).validated()
    assert config.channel_labels == ("ch001", "ch002", "ch003")
    assert config.expected_lsl_width == 120
    assert config.contract.channel_count == 120
    assert config.contract.channel_format == "float32"
    assert config.contract.nominal_srate == 50.0
    assert parse_channel_labels("A, B, C", 3) == ("A", "B", "C")
    with pytest.raises(ValueError, match="unique"):
        MonitorConfig(channel_count=2, channel_labels=("A", "A")).validated()
    with pytest.raises(ValueError, match="128 MiB"):
        MonitorConfig(channel_count=512, history_seconds=120).validated()
    assert MAX_HISTORY_BYTES == 128 * 1024 * 1024
    assert MonitorConfig(channel_count=64).validated().channel_labels == SAGA_64_CHANNEL_LABELS
    assert parse_channel_labels("", 64) == SAGA_64_CHANNEL_LABELS
    assert parse_args(["--channels", "64"]).config.channel_labels == SAGA_64_CHANNEL_LABELS


def test_signed_bipolar_display_transform_reuses_fixed_pair_order_and_direction():
    sample_count = 3
    raw = np.empty((64, sample_count), dtype=np.float32)
    value_by_label = {}
    for index, label in enumerate(SAGA_64_CHANNEL_LABELS):
        value_by_label[label] = np.float32(index + 1)
        raw[index] = value_by_label[label]

    labels, bipolar = transform_display_values(
        raw, SAGA_64_CHANNEL_LABELS, DISPLAY_MODE_BIPOLAR32
    )
    pairs = fixed_pair_labels()
    assert labels == tuple(pair for pair, _positive, _negative in pairs)
    assert bipolar.shape == (32, sample_count)
    for index, (_pair, positive, negative) in enumerate(pairs):
        np.testing.assert_array_equal(
            bipolar[index],
            np.full(sample_count, value_by_label[positive] - value_by_label[negative]),
        )
    raw_labels, raw_display = transform_display_values(
        raw, SAGA_64_CHANNEL_LABELS, DISPLAY_MODE_RAW
    )
    assert raw_labels == SAGA_64_CHANNEL_LABELS
    assert raw_display is raw


def test_bipolar_display_requires_explicit_saga_labels_and_has_qc_grid_order():
    with pytest.raises(ValueError, match="exactly 64"):
        transform_display_values(
            np.empty((3, 10), dtype=np.float32),
            ("A", "B", "C"),
            DISPLAY_MODE_BIPOLAR32,
        )
    with pytest.raises(ValueError, match="missing labels"):
        transform_display_values(
            np.empty((64, 10), dtype=np.float32),
            tuple(f"ch{index + 1:03d}" for index in range(64)),
            DISPLAY_MODE_BIPOLAR32,
        )
    assert [bipolar32_grid_position(index) for index in range(8)] == [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
    ]


def test_header_mismatch_is_blocking():
    config = MonitorConfig(channel_count=2, source_id="wanted").validated()
    assert validate_stream_info(FakeInfo(channels=80, source_id="wanted"), config)["channel_count"] == 80
    with pytest.raises(RuntimeError, match="channel_count"):
        validate_stream_info(FakeInfo(channels=2, source_id="wanted"), config)
    with pytest.raises(RuntimeError, match="source_id"):
        validate_stream_info(FakeInfo(channels=80, source_id="wrong"), config)


def test_history_full_resolution_statistics_and_decimation():
    history = RawEmgHistory(2, history_seconds=0.05)
    first = decoded(2, 1.0)
    second = decoded(2, 1.02, start=80)
    history.append_many((first, second))
    timestamps, values = history.window(0.05)
    assert values.shape == (2, 80)
    np.testing.assert_array_equal(values[:, :40], first.raw_ct)
    assert np.all(np.diff(timestamps) > 0)
    rms, p2p, mean = history.channel_statistics(0.05)
    np.testing.assert_allclose(rms, np.sqrt(np.mean(values.astype(float) ** 2, axis=1)))
    np.testing.assert_allclose(p2p, np.ptp(values, axis=1))
    np.testing.assert_allclose(mean, np.mean(values, axis=1))
    display_ts = np.r_[np.arange(20) / 2000, 1.0 + np.arange(20) / 2000]
    display_values = np.sin(np.arange(40))
    display_values[7], display_values[31] = 99.0, -88.0
    segments = decimate_minmax(display_ts, display_values, max_points=10)
    assert len(segments) == 2
    displayed = np.concatenate([part[:, 1] for part in segments])
    assert 99.0 in displayed and -88.0 in displayed


def test_display_bandpass_validation_and_chunked_sos_equivalence():
    from scipy.signal import sosfilt, sosfilt_zi

    for low, high in ((0.0, 500.0), (20.0, 20.0), (500.0, 20.0), (20.0, 1000.0)):
        with pytest.raises(ValueError, match="0 < low < high < Nyquist"):
            DisplayBandpassConfig(low_hz=low, high_hz=high).validated()
    with pytest.raises(ValueError, match="finite"):
        DisplayBandpassConfig(low_hz=float("nan"), high_hz=500.0).validated()

    rng = np.random.default_rng(17)
    raw = rng.normal(size=(3, PACKET_SAMPLES * 5)).astype(np.float32)
    processor = OnlineDisplayBandpass(3, DisplayBandpassConfig(20.0, 500.0))
    actual = []
    for index in range(5):
        right = (index + 1) * PACKET_SAMPLES
        packet = packet_from_ct(raw[:, right - PACKET_SAMPLES:right], right / 2000.0)
        actual.append(processor.process_packet(packet).raw_ct)
    base_zi = sosfilt_zi(processor.sos)[:, None, :] * raw[:, 0][None, :, None]
    expected, _state = sosfilt(processor.sos, raw.astype(np.float64), axis=1, zi=base_zi)
    np.testing.assert_allclose(
        np.concatenate(actual, axis=1), expected.astype(np.float32), rtol=1e-6, atol=1e-6
    )
    assert processor.has_state


def test_display_bandpass_resets_state_at_packet_gap():
    rng = np.random.default_rng(23)
    first = packet_from_ct(rng.normal(size=(2, 40)), 1.0)
    after_gap = packet_from_ct(rng.normal(size=(2, 40)), 1.04)
    streaming = OnlineDisplayBandpass(2)
    streaming.process_packet(first)
    gap_output = streaming.process_packet(after_gap).raw_ct
    fresh_output = OnlineDisplayBandpass(2).process_packet(after_gap).raw_ct
    np.testing.assert_array_equal(gap_output, fresh_output)

    # A positive packet timestamp can still reconstruct overlapping physical
    # sample times; this is also a discontinuity and must not carry zi across.
    overlapping = packet_from_ct(rng.normal(size=(2, 40)), 1.045)
    overlap_output = streaming.process_packet(overlapping).raw_ct
    fresh_overlap = OnlineDisplayBandpass(2).process_packet(overlapping).raw_ct
    np.testing.assert_array_equal(overlap_output, fresh_overlap)


def test_display_bandpass_frequency_selectivity_after_transient():
    duration_s = 10.0
    times = np.arange(int(duration_s * PHYSICAL_SAMPLE_RATE_HZ)) / PHYSICAL_SAMPLE_RATE_HZ
    frequencies = (0.2, 100.0, 800.0)
    raw = np.stack([np.sin(2 * np.pi * frequency * times) for frequency in frequencies]).astype(np.float32)
    processor = OnlineDisplayBandpass(
        3, DisplayBandpassConfig(DEFAULT_BANDPASS_LOW_HZ, DEFAULT_BANDPASS_HIGH_HZ)
    )
    output = []
    for left in range(0, raw.shape[1], PACKET_SAMPLES):
        right = left + PACKET_SAMPLES
        output.append(
            processor.process_packet(packet_from_ct(raw[:, left:right], right / 2000.0)).raw_ct
        )
    filtered = np.concatenate(output, axis=1)[:, int(3 * PHYSICAL_SAMPLE_RATE_HZ):]
    rms = np.sqrt(np.mean(filtered.astype(np.float64) ** 2, axis=1))
    assert rms[1] > 0.65
    assert rms[0] < 0.02 * rms[1]
    assert rms[2] < 0.08 * rms[1]


class SleepingWorker(RawStreamWorker):
    def __init__(self, config):
        super().__init__(config, resolver=lambda **_kwargs: None)

    def run(self):
        while not self.isInterruptionRequested():
            QThread.msleep(5)


def process_until(app, predicate, timeout_s=1.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_ui_settings_are_separate_and_running_config_locks(app):
    window = RawStreamMonitorWindow(MonitorConfig(channel_count=2), worker_factory=SleepingWorker)
    assert not window.settings_dialog.isVisible()
    window._show_settings()
    assert window.settings_dialog.isVisible()
    window.connect_stream()
    assert process_until(app, lambda: window.worker is not None and window.worker.isRunning())
    assert not window.name_edit.isEnabled()
    assert window.bandpass_enable_check.isEnabled()
    assert window.bandpass_low_spin.isEnabled()
    assert window.bandpass_high_spin.isEnabled()
    assert window.bandpass_apply_button.isEnabled()
    assert window.disconnect_button.isEnabled()
    assert window.disconnect_stream()
    assert window.name_edit.isEnabled()
    window.close()
    app.processEvents()


def test_ui_contract_labels_scale_and_structured_health(app):
    window = RawStreamMonitorWindow(MonitorConfig(channel_count=64))
    assert window.sample_rate_header.value_label.text() == "2000 Hz"
    assert window.packet_rate_header.value_label.text() == "-- pkt/s"
    assert window.age_header.toolTip().startswith("校时后的最近有效样本年龄")
    assert "延迟" not in window.age_header.value_label.text()
    assert set(window.metric_cards) == {"rms", "p2p", "mean"}
    assert "std" not in window.metric_cards
    assert "µV" not in window.styleSheet()
    assert window.gain_mode_combo.currentText() == "Auto"
    assert not window.fixed_gain_spin.isVisible()
    window.gain_mode_combo.setCurrentText("Custom")
    assert not window.fixed_gain_spin.isHidden()
    assert set(window.health_cells) == {
        "gap", "out_of_order", "invalid", "disconnect", "packet_rate", "age", "correction"
    }
    window.close()
    app.processEvents()


def test_ui_default_32_per_page_and_online_bandpass(app):
    generic_labels = tuple(f"ch{index + 1:03d}" for index in range(64))
    window = RawStreamMonitorWindow(
        MonitorConfig(channel_count=64, channel_labels=generic_labels)
    )
    assert window.page_size_combo.currentText() == "32"
    assert window.page_label.text() == "通道 1–32 / 64 · 第 1/2 页"
    assert window.bandpass_enable_check.isChecked()
    assert window._bandpass_enabled
    assert "在线带通 1–200 Hz" in window.bandpass_status.text()
    assert "在线带通 1–200 Hz" in window.waveform_title.text()
    assert len(window.channel_buttons) == 64
    button_ids = [id(button) for button in window.channel_buttons]
    window._on_batch((decoded(64, 3.0),))
    window._select_channel(63)
    assert window._page == 1
    assert window.canvas.labels == tuple(f"ch{index:03d}" for index in range(33, 65))
    assert window.detail_canvas.labels == ("ch064",)
    assert window.channel_buttons[63].isChecked()
    assert [id(button) for button in window.channel_buttons] == button_ids
    window._select_channel(0)
    assert [id(button) for button in window.channel_buttons] == button_ids
    window.page_size_combo.setCurrentText("16")
    assert [id(button) for button in window.channel_buttons] == button_ids
    window.close()
    app.processEvents()


def test_ui_bipolar_switch_preserves_raw_history_and_updates_values_stats_and_grid(app):
    window = RawStreamMonitorWindow(MonitorConfig(channel_count=64))
    assert window.display_mode_combo.currentText() == DISPLAY_MODE_BIPOLAR32
    assert window.channel_header.value_label.text() == "32 pairs"
    window.bandpass_enable_check.setChecked(False)
    assert window._apply_bandpass_settings()
    window.display_mode_combo.setCurrentText(DISPLAY_MODE_RAW)
    first = decoded(64, 3.0)
    window._on_batch((first,))
    history_id = id(window.history)
    sample_count = window.history.sample_count
    window._select_channel(31)
    assert window._page == 0

    window.display_mode_combo.setCurrentText(DISPLAY_MODE_BIPOLAR32)
    assert id(window.history) == history_id
    assert window.history.sample_count == sample_count
    assert window.selected_channel == 0
    assert window._page == 0
    assert window._labels == tuple(pair for pair, _positive, _negative in fixed_pair_labels())
    assert len(window.channel_buttons) == 32
    assert window.channel_header.value_label.text() == "32 pairs"
    assert "Bipolar-32" in window.waveform_title.text()

    _timestamps, raw_values = window.history.window(5.0)
    expected_labels, expected_values = transform_display_values(
        raw_values, SAGA_64_CHANNEL_LABELS, DISPLAY_MODE_BIPOLAR32
    )
    assert window.canvas.labels == expected_labels
    np.testing.assert_array_equal(window.canvas.values, expected_values)
    expected_stats = raw_channel_statistics(expected_values)
    for actual, expected in zip(window._cached_stats, expected_stats):
        np.testing.assert_allclose(actual, expected)
    assert "正端 − 负端" in window.channel_buttons[0].toolTip()
    for index in range(32):
        item_index = window.channel_matrix_layout.indexOf(window.channel_buttons[index])
        row, column, row_span, column_span = window.channel_matrix_layout.getItemPosition(
            item_index
        )
        expected_row, expected_column = bipolar32_grid_position(index)
        assert (row, column, row_span, column_span) == (
            expected_row + 1, expected_column + 1, 1, 1
        )
        assert window.channel_buttons[index].accessibleName() == expected_labels[index]

    window.display_mode_combo.setCurrentText(DISPLAY_MODE_RAW)
    assert id(window.history) == history_id
    assert window.history.sample_count == sample_count
    assert len(window.channel_buttons) == 64
    np.testing.assert_array_equal(window.canvas.values, raw_values[:32])
    window.close()
    app.processEvents()


def test_ui_bandpass_uses_new_data_only_and_disabling_restores_full_raw(app):
    window = RawStreamMonitorWindow(
        MonitorConfig(channel_count=2, channel_labels=("FCR", "ECR"))
    )
    assert window.bandpass_enable_check.isChecked()
    assert window._bandpass_enabled
    window.bandpass_enable_check.setChecked(False)
    assert window._apply_bandpass_settings()
    first = decoded(2, 2.0)
    window._on_batch((first,))
    raw_history_id = id(window.history)
    raw_samples = window.history.sample_count

    window.bandpass_enable_check.setChecked(True)
    assert window._apply_bandpass_settings()
    assert id(window.history) == raw_history_id
    assert window.history.sample_count == raw_samples
    assert window.filtered_history.sample_count == 0
    assert "新参数仅作用于应用后的新数据" in window.bandpass_status.text()
    assert "在线带通 1–200 Hz" in window.waveform_title.text()

    second = decoded(2, 2.02, start=80)
    window._on_batch((second,))
    window._refresh_display(force=True)
    assert window.history.sample_count == raw_samples + PACKET_SAMPLES
    assert window.filtered_history.sample_count == PACKET_SAMPLES
    assert window.canvas.timestamps[0] == pytest.approx(second.sample_timestamps[0])
    assert not np.array_equal(window.canvas.values, second.raw_ct)
    displayed_filtered = window.canvas.values.copy()
    expected_stats = raw_channel_statistics(displayed_filtered)
    for actual, expected in zip(window._cached_stats, expected_stats):
        np.testing.assert_allclose(actual, expected)

    window.bandpass_enable_check.setChecked(False)
    assert window._apply_bandpass_settings()
    assert window.filtered_history.sample_count == 0
    assert window.canvas.timestamps[0] == pytest.approx(first.sample_timestamps[0])
    np.testing.assert_array_equal(
        window.canvas.values, np.concatenate((first.raw_ct, second.raw_ct), axis=1)
    )
    assert "未滤波" in window.waveform_title.text()
    assert not np.array_equal(window.canvas.values[:, -PACKET_SAMPLES:], displayed_filtered)
    window.close()
    app.processEvents()


def test_ui_bandpass_invalid_parameters_are_atomic_and_mode_switch_does_not_reset(app):
    window = RawStreamMonitorWindow(MonitorConfig(channel_count=64))
    assert window.display_mode_combo.currentText() == DISPLAY_MODE_BIPOLAR32
    window.display_mode_combo.setCurrentText(DISPLAY_MODE_RAW)
    window.bandpass_enable_check.setChecked(True)
    assert window._apply_bandpass_settings()
    window._on_batch((decoded(64, 3.0),))
    filtered_history_id = id(window.filtered_history)
    filtered_samples = window.filtered_history.sample_count
    active_config = window._bandpass_config

    window.display_mode_combo.setCurrentText(DISPLAY_MODE_BIPOLAR32)
    assert id(window.filtered_history) == filtered_history_id
    assert window.filtered_history.sample_count == filtered_samples
    assert "在线带通" in window.detail_meta.text()
    assert "在线带通" in window.channel_buttons[0].toolTip()

    window.bandpass_low_spin.setValue(600.0)
    window.bandpass_high_spin.setValue(500.0)
    assert not window._apply_bandpass_settings()
    assert window._bandpass_enabled
    assert window._bandpass_config == active_config
    assert id(window.filtered_history) == filtered_history_id
    assert window.filtered_history.sample_count == filtered_samples
    assert "当前滤波配置未改变" in window.bandpass_status.text()

    # Turning filtering off is always safe and immediate, even while the
    # unapplied frequency fields contain an invalid relationship.
    window.bandpass_enable_check.setChecked(False)
    assert window._apply_bandpass_settings()
    assert not window._bandpass_enabled
    assert window._bandpass_config == active_config
    assert window.filtered_history.sample_count == 0
    assert window.canvas.timestamps[0] == pytest.approx(3.0 - 39 / 2000.0)
    assert "未滤波" in window.waveform_title.text()
    window.close()
    app.processEvents()


def test_ui_pause_freezes_filtered_display_but_filter_history_keeps_receiving(app):
    window = RawStreamMonitorWindow(
        MonitorConfig(channel_count=2, channel_labels=("FCR", "ECR"))
    )
    window.bandpass_enable_check.setChecked(True)
    assert window._apply_bandpass_settings()
    window._on_batch((decoded(2, 2.0),))
    window._refresh_display(force=True)
    canvas_latest = float(window.canvas.timestamps[-1])
    window.pause_check.setChecked(True)
    window._on_batch((decoded(2, 2.02, start=80),))
    window._refresh_display()
    assert window.history.latest_timestamp == pytest.approx(2.02)
    assert window.filtered_history.latest_timestamp == pytest.approx(2.02)
    assert float(window.canvas.timestamps[-1]) == canvas_latest
    window.pause_check.setChecked(False)
    window._refresh_display(force=True)
    assert float(window.canvas.timestamps[-1]) == pytest.approx(2.02)
    window.close()
    app.processEvents()


def test_ui_non_saga_config_keeps_raw_mode_and_disables_bipolar(app):
    window = RawStreamMonitorWindow(
        MonitorConfig(channel_count=3, channel_labels=("FCR", "ECR", "TRI"))
    )
    bipolar_item = window.display_mode_combo.model().item(1)
    assert bipolar_item is not None
    assert not bipolar_item.isEnabled()
    assert window.display_mode_combo.currentText() == DISPLAY_MODE_RAW
    assert "requires exactly 64" in window.display_mode_combo.toolTip()
    window._on_batch((decoded(3, 2.0),))
    window._refresh_display(force=True)
    assert window.canvas.labels == ("FCR", "ECR", "TRI")
    assert window.canvas.values.shape == (3, PACKET_SAMPLES)
    window.close()
    app.processEvents()


def test_reconnect_config_signal_safely_falls_back_from_bipolar_to_generic_raw(app):
    window = RawStreamMonitorWindow(
        MonitorConfig(channel_count=64), worker_factory=SleepingWorker
    )
    window.display_mode_combo.setCurrentText(DISPLAY_MODE_BIPOLAR32)
    window.channel_spin.setValue(3)
    window.labels_edit.setText("FCR,ECR,TRI")
    window.connect_stream()
    assert process_until(app, lambda: window.worker is not None and window.worker.isRunning())
    assert window.display_mode_combo.currentText() == DISPLAY_MODE_RAW
    assert window._raw_labels == ("FCR", "ECR", "TRI")
    assert window._labels == ("FCR", "ECR", "TRI")
    assert window.history.channel_count == 3
    assert not window.display_mode_combo.model().item(1).isEnabled()
    assert window.disconnect_stream()
    window.close()
    app.processEvents()


def test_ui_pause_freezes_display_but_history_keeps_receiving(app):
    window = RawStreamMonitorWindow(MonitorConfig(channel_count=2, channel_labels=("FCR", "ECR")))
    window.bandpass_enable_check.setChecked(False)
    assert window._apply_bandpass_settings()
    window._on_batch((decoded(2, 2.0),))
    window._refresh_display(force=True)
    canvas_latest = float(window.canvas.timestamps[-1])
    window.pause_check.setChecked(True)
    window._on_batch((decoded(2, 2.02, start=80),))
    window._refresh_display()
    assert window.history.latest_timestamp == pytest.approx(2.02)
    assert float(window.canvas.timestamps[-1]) == canvas_latest
    window._select_channel(1)
    assert float(window.canvas.timestamps[-1]) == pytest.approx(2.02)
    assert "ECR" in window.detail_label.text()
    window.close()
    app.processEvents()


def test_ui_health_uses_packet_rate_data_age_and_real_failure(app):
    window = RawStreamMonitorWindow(MonitorConfig(channel_count=2))
    window._on_health({
        "rx_total": 10, "actual_rate_hz": 49.8, "last_sample_age_ms": 8.0,
        "gap_total": 1, "out_of_order_total": 2, "invalid_total": 3, "lost_total": 0,
    })
    assert window.health_state_label.text() == "接收中"
    assert window.packet_rate_header.value_label.text() == "49.8 pkt/s"
    assert window.health_cells["age"].value_label.text() == "8 ms"
    assert window.health_cells["gap"].value_label.text() == "1"
    window._on_failed("simulated inlet failure")
    assert window.health_cells["disconnect"].value_label.text() == "1"
    assert window.connection_label.text() == "连接/接收错误"
    window._on_health({
        "rx_total": 0, "actual_rate_hz": 0.0, "last_sample_age_ms": None,
        "gap_total": 0, "out_of_order_total": 0, "invalid_total": 0, "lost_total": 0,
    })
    assert window.health_state_label.text() == "等待首包"
    window.close()
    app.processEvents()


@pytest.mark.parametrize("width,height", [(1440, 900), (1100, 720), (980, 680)])
def test_ui_resizes_without_global_scroll_or_empty_grab(app, width, height):
    window = RawStreamMonitorWindow(MonitorConfig(channel_count=64))
    window.bandpass_enable_check.setChecked(True)
    assert window._apply_bandpass_settings()
    window.resize(width, height)
    window.show()
    app.processEvents()
    assert not isinstance(window.centralWidget(), QScrollArea)
    image = window.grab().toImage()
    assert not image.isNull()
    assert image.width() >= width and image.height() >= height
    window.close()
    app.processEvents()


def test_queued_old_worker_signals_do_not_change_new_connection(app):
    window = RawStreamMonitorWindow(MonitorConfig(channel_count=2))
    old = RawStreamWorker(MonitorConfig(channel_count=2))
    new = RawStreamWorker(MonitorConfig(channel_count=2))
    window.worker = old
    window._bind_worker(old)
    old.finished.emit()
    old.batch_received.emit((decoded(2, 10),))
    old.failed.emit("old connection error")
    window.worker = new
    window._bind_worker(new)
    window._set_config_locked(True)
    window._set_connection_state("discovering", "正在连接")
    app.processEvents()
    assert window.history.sample_count == 0
    assert window.connection_label.text() == "正在连接"
    assert window.disconnect_button.isEnabled()
    assert not window._worker_failed
    window.close()
    app.processEvents()
