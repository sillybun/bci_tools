#!/usr/bin/env python3
"""Standalone Qt console for the packetized signed raw EMG LSL stream."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import math
import sys
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt
from PySide6.QtGui import QColor, QCloseEvent, QFont, QFontMetrics, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QFrame,
    QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from raw_stream_monitor import (
    DEFAULT_BANDPASS_HIGH_HZ, DEFAULT_BANDPASS_LOW_HZ, DEFAULT_HISTORY_SECONDS,
    DEFAULT_STREAM_NAME, DEFAULT_STREAM_TYPE, DISPLAY_BANDPASS_ORDER,
    DISPLAY_MODE_BIPOLAR32, DISPLAY_MODE_RAW, MAX_CHANNEL_COUNT,
    DisplayBandpassConfig, MonitorConfig, OnlineDisplayBandpass, RawEmgHistory,
    RawStreamWorker, bipolar32_grid_position, decimate_minmax,
    default_channel_labels, parse_channel_labels, raw_channel_statistics,
    transform_display_values,
)
from packet import (
    PACKET_DURATION_SECONDS, PHYSICAL_SAMPLE_RATE_HZ, DecodedPacket,
)

BACKGROUND = "#F4F6F8"
CARD_BORDER = "#D9DEE7"
PRIMARY = "#2F6FED"
TRACE = "#138A88"
TEXT = "#172033"
MUTED = "#667085"
GREEN = "#36A269"
ORANGE = "#D97706"
RED = "#D94A45"


def _format_raw(value: float) -> str:
    if not math.isfinite(value):
        return "--"
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


class MetricCard(QFrame):
    """Compact factual metric card; raw wire does not define a physical unit."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("metricTitle")
        self.value_label = QLabel("--")
        self.value_label.setObjectName("metricValue")
        unit_label = QLabel("raw units")
        unit_label.setObjectName("metricUnit")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(unit_label)

    def set_value(self, value: float) -> None:
        self.value_label.setText(_format_raw(float(value)))


class HealthCell(QFrame):
    def __init__(self, title: str, value: str = "--", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("healthCell")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 7, 13, 7)
        layout.setSpacing(1)
        title_label = QLabel(title)
        title_label.setObjectName("healthTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("healthValue")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str, state: str = "neutral") -> None:
        colors = {"healthy": GREEN, "warning": ORANGE, "error": RED, "neutral": TEXT}
        self.value_label.setText(value)
        self.value_label.setStyleSheet(f"color:{colors[state]};")


class HeaderValue(QWidget):
    def __init__(self, title: str, value: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 3, 12, 3)
        layout.setSpacing(0)
        title_label = QLabel(title)
        title_label.setObjectName("headerMetricTitle")
        self.value_label = QLabel(value)
        self.value_label.setObjectName("headerMetricValue")
        layout.addWidget(title_label)
        layout.addWidget(self.value_label)


class WaveformCanvas(QWidget):
    """Fast QPainter view for stacked or selected signed-raw traces."""

    def __init__(self, parent: QWidget | None = None, *, detail: bool = False) -> None:
        super().__init__(parent)
        self.labels: tuple[str, ...] = ()
        self.timestamps = np.empty(0, dtype=np.float64)
        self.values = np.empty((0, 0), dtype=np.float32)
        self.gain = 1.0
        self.selected = 0
        self.view_seconds = 5.0
        self.detail = detail
        # Keep all eight default lanes visible in a 720 px-tall window. The
        # splitter still grants the overview most of the available height.
        self.setMinimumHeight(110 if detail else 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_data(
        self, labels: Sequence[str], timestamps: np.ndarray, values: np.ndarray,
        *, gain: float, selected: int, view_seconds: float = 5.0,
    ) -> None:
        self.labels = tuple(labels)
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float32)
        self.gain = max(float(gain), 1e-12)
        self.selected = int(selected)
        self.view_seconds = max(float(view_seconds), 0.1)
        self.update()

    def _draw_grid(self, painter: QPainter, plot: QRectF) -> None:
        font = painter.font()
        font.setPointSizeF(max(8.0, font.pointSizeF() - 1.0))
        painter.setFont(font)
        for tick in range(6):
            x = plot.left() + tick / 5 * plot.width()
            painter.setPen(QPen(QColor("#E6EAF0"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(QColor(MUTED))
            painter.drawText(
                QRectF(x - 25, plot.bottom() + 5, 50, 18),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                f"{tick / 5 * self.view_seconds:g} s",
            )

    def _draw_scale(self, painter: QPainter, plot: QRectF) -> None:
        x = plot.right() + 14
        center = plot.top() + min(32.0, plot.height() / 2)
        half = min(18.0, max(9.0, plot.height() * (0.20 if self.detail else 0.035)))
        painter.setPen(QPen(QColor("#98A2B3"), 1))
        painter.drawLine(QPointF(x, center - half), QPointF(x, center + half))
        painter.drawLine(QPointF(x - 3, center - half), QPointF(x + 3, center - half))
        painter.drawLine(QPointF(x - 3, center + half), QPointF(x + 3, center + half))
        painter.setPen(QColor(MUTED))
        painter.drawText(
            QRectF(x + 5, center - 12, 72, 24),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"±{_format_raw(self.gain)}",
        )

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        count = len(self.labels)
        label_width = 72
        if count:
            metrics = QFontMetrics(self.font())
            label_width = max(64, min(128, max(metrics.horizontalAdvance(x) for x in self.labels) + 18))
        plot = QRectF(label_width, 8, max(1, self.width() - label_width - 88), max(1, self.height() - 39))
        self._draw_grid(painter, plot)
        self._draw_scale(painter, plot)
        if count == 0:
            painter.setPen(QColor(MUTED))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "等待 raw EMG 数据")
            return
        right = float(self.timestamps[-1]) if self.timestamps.size else 0.0
        left = right - self.view_seconds
        lane_height = plot.height() / count
        # One display point per horizontal pixel is already above the visible
        # resolution. The backend min/max reducer still preserves extrema.
        max_points = max(100, int(plot.width()))
        for lane, label in enumerate(self.labels):
            top = plot.top() + lane * lane_height
            center = top + lane_height / 2
            if lane == self.selected:
                painter.fillRect(QRectF(0, top, plot.right(), lane_height), QColor("#F2F6FF"))
            painter.setPen(QPen(QColor("#E5E9F0"), 1))
            painter.drawLine(QPointF(plot.left(), center), QPointF(plot.right(), center))
            painter.setPen(QColor(PRIMARY if lane == self.selected else "#36506B"))
            painter.drawText(QRectF(8, top, label_width - 14, lane_height), Qt.AlignmentFlag.AlignVCenter, label)
            if lane >= self.values.shape[0] or not self.timestamps.size:
                continue
            painter.setPen(QPen(QColor(PRIMARY if lane == self.selected else TRACE), 1.0))
            amplitude = max(2.0, lane_height * (0.42 if not self.detail else 0.40))
            for segment in decimate_minmax(self.timestamps, self.values[lane], max_points):
                visible = segment[(segment[:, 0] >= left) & (segment[:, 0] <= right)]
                points = [
                    QPointF(
                        plot.left() + (stamp - left) / self.view_seconds * plot.width(),
                        center - float(np.clip(value / self.gain, -1.0, 1.0)) * amplitude,
                    ) for stamp, value in visible
                ]
                if len(points) > 1:
                    painter.drawPolyline(QPolygonF(points))
                elif points:
                    painter.drawPoint(points[0])


class RawStreamMonitorWindow(QMainWindow):
    """Read-only raw-stream monitor with an explicitly local channel config."""

    def __init__(
        self, config: MonitorConfig | None = None, *,
        worker_factory: Callable[[MonitorConfig], RawStreamWorker] = RawStreamWorker,
        auto_connect: bool = False,
    ) -> None:
        super().__init__()
        self._initial_config = (config or MonitorConfig()).validated()
        self._worker_factory = worker_factory
        self.worker: RawStreamWorker | None = None
        self._connection_token = None
        self.history = RawEmgHistory(self._initial_config.channel_count, self._initial_config.history_seconds)
        self.filtered_history = RawEmgHistory(
            self._initial_config.channel_count,
            self._initial_config.history_seconds,
        )
        self._bandpass_enabled = True
        self._bandpass_config = DisplayBandpassConfig().validated()
        self._display_bandpass = OnlineDisplayBandpass(
            self._initial_config.channel_count, self._bandpass_config
        )
        self.selected_channel = 0
        self.channel_buttons: list[QPushButton] = []
        self._button_labels: tuple[str, ...] = ()
        self._last_health: dict[str, Any] = {}
        self._raw_labels = self._initial_config.channel_labels
        try:
            self._labels, _values = transform_display_values(
                np.empty((len(self._raw_labels), 0), dtype=np.float32),
                self._raw_labels,
                DISPLAY_MODE_BIPOLAR32,
            )
        except ValueError:
            self._display_mode = DISPLAY_MODE_RAW
            self._labels = self._raw_labels
        else:
            self._display_mode = DISPLAY_MODE_BIPOLAR32
        self._page = 0
        self._stats_tick = 0
        self._cached_stats = tuple(np.full(len(self._labels), np.nan) for _ in range(3))
        self._time_correction_s: float | None = None
        self._disconnect_total = 0
        self._worker_failed = False
        self._build_ui()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(50)
        self.refresh_timer.timeout.connect(self._refresh_display)
        self.refresh_timer.start()
        self.setWindowTitle("EMG 实时监控台")
        self.resize(1440, 900)
        self.setMinimumSize(980, 680)
        if auto_connect:
            QTimer.singleShot(0, self.connect_stream)

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("central")
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        root.addWidget(self._build_header())
        root.addWidget(self._build_toolbar())
        body = QSplitter(Qt.Orientation.Horizontal)
        body.setObjectName("bodySplitter")
        body.setChildrenCollapsible(False)
        body.addWidget(self._build_channel_browser())
        right = QSplitter(Qt.Orientation.Vertical)
        right.setObjectName("rightSplitter")
        right.setChildrenCollapsible(False)
        right.addWidget(self._build_waveform_card())
        right.addWidget(self._build_inspector_card())
        right.setStretchFactor(0, 7)
        right.setStretchFactor(1, 3)
        body.addWidget(right)
        body.setStretchFactor(0, 0)
        body.setStretchFactor(1, 1)
        body.setSizes([310, 1090])
        root.addWidget(body, 1)
        root.addWidget(self._build_health_bar())
        self.setCentralWidget(central)
        self._build_settings_dialog()
        self._set_display_mode_availability()
        self._ensure_channel_buttons()
        self._update_channel_header()
        self._update_page_controls()
        self._apply_stylesheet()

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("headerCard")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 8, 10, 8)
        layout.setSpacing(0)
        logo = QLabel("⌁")
        logo.setObjectName("appLogo")
        title = QLabel("EMG 实时监控台")
        title.setObjectName("appTitle")
        layout.addWidget(logo)
        layout.addSpacing(10)
        layout.addWidget(title)
        layout.addSpacing(22)
        self.status_dot = QLabel("●")
        self.status_dot.setObjectName("statusDot")
        self.connection_label = QLabel("未连接")
        self.connection_label.setObjectName("connectionStatus")
        self.connection_label.setMinimumWidth(76)
        layout.addWidget(self.status_dot)
        layout.addSpacing(5)
        layout.addWidget(self.connection_label)
        layout.addStretch(1)
        self.stream_header = HeaderValue("流名称", self._initial_config.stream_name)
        self.channel_header = HeaderValue("通道", f"{self._initial_config.channel_count} ch")
        self.sample_rate_header = HeaderValue("采样率", f"{PHYSICAL_SAMPLE_RATE_HZ:g} Hz")
        self.packet_rate_header = HeaderValue("包率", "-- pkt/s")
        self.age_header = HeaderValue("数据年龄", "-- ms")
        for widget in (self.stream_header, self.channel_header, self.sample_rate_header, self.packet_rate_header, self.age_header):
            layout.addWidget(widget)
        self.age_header.setToolTip("校时后的最近有效样本年龄，不等同于单向网络延迟。")
        self.settings_button = QPushButton("⚙  设置")
        self.settings_button.setObjectName("settingsButton")
        self.settings_button.clicked.connect(self._show_settings)
        layout.addSpacing(8)
        layout.addWidget(self.settings_button)
        self._set_connection_state("disconnected", "未连接")
        return header

    def _build_toolbar(self) -> QWidget:
        toolbar = QFrame()
        toolbar.setObjectName("toolbarCard")
        rows = QVBoxLayout(toolbar)
        rows.setContentsMargins(14, 7, 14, 7)
        rows.setSpacing(8)
        layout = QHBoxLayout()
        rows.addLayout(layout)
        layout.setSpacing(8)
        layout.addWidget(QLabel("显示模式"))
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems([DISPLAY_MODE_RAW, DISPLAY_MODE_BIPOLAR32])
        self.display_mode_combo.setCurrentText(self._display_mode)
        self.display_mode_combo.setMinimumWidth(205)
        layout.addWidget(self.display_mode_combo)
        layout.addSpacing(8)
        layout.addWidget(QLabel("每页通道"))
        self.page_size_combo = QComboBox()
        self.page_size_combo.addItems(["8", "16", "32"])
        self.page_size_combo.setCurrentText("32")
        self.page_size_combo.setFixedWidth(72)
        layout.addWidget(self.page_size_combo)
        self.prev_button = QPushButton("‹  上一页")
        self.next_button = QPushButton("下一页  ›")
        self.page_label = QLabel()
        self.page_label.setObjectName("pageLabel")
        for widget in (self.prev_button, self.next_button, self.page_label):
            layout.addWidget(widget)
        layout.addStretch(1)
        layout = QHBoxLayout()
        layout.setSpacing(8)
        rows.addLayout(layout)
        layout.addWidget(QLabel("时间窗口"))
        self.view_combo = QComboBox()
        self.view_combo.addItems(["2", "5", "10"])
        self.view_combo.setCurrentText("5")
        self.view_combo.setFixedWidth(74)
        layout.addWidget(self.view_combo)
        layout.addWidget(QLabel("s"))
        layout.addSpacing(10)
        layout.addWidget(QLabel("幅度刻度"))
        self.gain_mode_combo = QComboBox()
        self.gain_mode_combo.addItems(["Auto", "±50 raw", "±100 raw", "±200 raw", "±500 raw", "Custom"])
        self.gain_mode_combo.setFixedWidth(112)
        layout.addWidget(self.gain_mode_combo)
        self.fixed_gain_spin = QDoubleSpinBox()
        self.fixed_gain_spin.setDecimals(3)
        self.fixed_gain_spin.setRange(0.001, 1e9)
        self.fixed_gain_spin.setValue(100.0)
        self.fixed_gain_spin.setSuffix(" raw")
        self.fixed_gain_spin.setFixedWidth(118)
        self.fixed_gain_spin.setVisible(False)
        layout.addWidget(self.fixed_gain_spin)
        self.pause_check = QPushButton("Ⅱ  暂停")
        self.pause_check.setObjectName("pauseButton")
        self.pause_check.setCheckable(True)
        layout.addStretch(1)
        layout.addWidget(self.pause_check)
        self.prev_button.clicked.connect(lambda: self._change_page(-1))
        self.next_button.clicked.connect(lambda: self._change_page(1))
        self.page_size_combo.currentTextChanged.connect(self._on_page_size_changed)
        self.display_mode_combo.currentTextChanged.connect(self._on_display_mode_changed)
        self.gain_mode_combo.currentTextChanged.connect(self._on_gain_mode_changed)
        self.pause_check.toggled.connect(self._on_pause_toggled)
        return toolbar

    def _build_channel_browser(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("card")
        panel.setMinimumWidth(280)
        panel.setMaximumWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(9)
        title_row = QHBoxLayout()
        title = QLabel("通道浏览器")
        title.setObjectName("cardTitle")
        self.channel_count_label = QLabel(f"{len(self._labels)} 个")
        self.channel_count_label.setObjectName("mutedLabel")
        title_row.addWidget(title)
        title_row.addStretch(1)
        title_row.addWidget(self.channel_count_label)
        layout.addLayout(title_row)
        self.channel_search = QLineEdit()
        self.channel_search.setPlaceholderText("搜索通道")
        self.channel_search.setClearButtonEnabled(True)
        self.channel_search.textChanged.connect(self._filter_channels)
        layout.addWidget(self.channel_search)
        self.channel_matrix_widget = QWidget()
        self.channel_matrix_layout = QGridLayout(self.channel_matrix_widget)
        self.channel_matrix_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_matrix_layout.setHorizontalSpacing(6)
        self.channel_matrix_layout.setVerticalSpacing(6)
        self.channel_matrix_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        channel_scroll = QScrollArea()
        channel_scroll.setObjectName("channelScroll")
        channel_scroll.setWidgetResizable(True)
        channel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        channel_scroll.setWidget(self.channel_matrix_widget)
        layout.addWidget(channel_scroll, 1)
        legend = QLabel("● 选中     ● 未评估")
        legend.setObjectName("channelLegend")
        legend.setToolTip("当前 raw wire 没有通道 QC；幅值不用于判定正常或异常。")
        layout.addWidget(legend)
        return panel

    def _build_waveform_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QFrame()
        header.setObjectName("cardHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 10, 16, 8)
        self.waveform_title = QLabel("signed raw EMG 波形")
        self.waveform_title.setObjectName("cardTitle")
        self.waveform_meta = QLabel()
        self.waveform_meta.setObjectName("mutedLabel")
        header_layout.addWidget(self.waveform_title)
        header_layout.addStretch(1)
        header_layout.addWidget(self.waveform_meta)
        layout.addWidget(header)
        self.canvas = WaveformCanvas()
        layout.addWidget(self.canvas, 1)
        return card

    def _build_inspector_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QFrame()
        header.setObjectName("cardHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 9, 16, 7)
        self.detail_label = QLabel(f"选中通道详情 — {self._labels[0]}")
        self.detail_label.setObjectName("cardTitle")
        self.detail_meta = QLabel("signed raw · 未进行 QC")
        self.detail_meta.setObjectName("mutedLabel")
        header_layout.addWidget(self.detail_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.detail_meta)
        layout.addWidget(header)
        content = QHBoxLayout()
        content.setContentsMargins(10, 0, 10, 10)
        content.setSpacing(10)
        self.detail_canvas = WaveformCanvas(detail=True)
        content.addWidget(self.detail_canvas, 1)
        metrics = QGridLayout()
        metrics.setSpacing(8)
        self.metric_cards = {"rms": MetricCard("RMS"), "p2p": MetricCard("P-P"), "mean": MetricCard("Mean")}
        metrics.addWidget(self.metric_cards["rms"], 0, 0)
        metrics.addWidget(self.metric_cards["p2p"], 0, 1)
        metrics.addWidget(self.metric_cards["mean"], 1, 0, 1, 2)
        metric_host = QWidget()
        metric_host.setLayout(metrics)
        metric_host.setFixedWidth(270)
        content.addWidget(metric_host)
        layout.addLayout(content, 1)
        return card

    def _build_health_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("healthBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(0)
        state_host = QWidget()
        state_layout = QHBoxLayout(state_host)
        state_layout.setContentsMargins(10, 0, 16, 0)
        self.health_state_dot = QLabel("●")
        self.health_state_dot.setStyleSheet(f"color:{MUTED};")
        self.health_state_label = QLabel("等待连接")
        self.health_state_label.setObjectName("healthState")
        state_layout.addWidget(self.health_state_dot)
        state_layout.addSpacing(6)
        state_layout.addWidget(self.health_state_label)
        layout.addWidget(state_host)
        self.health_cells = {
            "gap": HealthCell("Gap", "0"), "out_of_order": HealthCell("Out-of-order", "0"),
            "invalid": HealthCell("Invalid", "0"), "disconnect": HealthCell("Disconnect", "0"),
            "packet_rate": HealthCell("包率", "-- pkt/s"), "age": HealthCell("数据年龄", "-- ms"),
            "correction": HealthCell("时间校正", "-- s"),
        }
        self.health_cells["age"].setToolTip("校时后的最近有效样本年龄，不等同于单向网络延迟。")
        for cell in self.health_cells.values():
            layout.addWidget(cell, 1)
        self.health_label = QLabel()
        self.health_label.hide()
        layout.addWidget(self.health_label)
        return bar

    def _build_settings_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("EMG 流设置")
        dialog.setModal(False)
        dialog.setMinimumWidth(560)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 18, 20, 18)
        title = QLabel("LSL 连接设置")
        title.setObjectName("dialogTitle")
        subtitle = QLabel("raw wire 不携带 shape、通道标签或物理单位；以下均为本地契约。")
        subtitle.setObjectName("mutedLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        form = QFormLayout()
        form.setContentsMargins(0, 12, 0, 6)
        form.setSpacing(10)
        self.name_edit = QLineEdit(self._initial_config.stream_name)
        self.type_edit = QLineEdit(self._initial_config.stream_type)
        self.source_edit = QLineEdit(self._initial_config.source_id or "")
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(1, MAX_CHANNEL_COUNT)
        self.channel_spin.setValue(self._initial_config.channel_count)
        generated_labels = default_channel_labels(self._initial_config.channel_count)
        self.labels_edit = QLineEdit("" if self._initial_config.channel_labels == generated_labels else ",".join(self._initial_config.channel_labels))
        self.labels_edit.setPlaceholderText("留空：64 通道使用 SAGA-64；其他 C 生成 ch001, ch002, ...")
        self.history_spin = QDoubleSpinBox()
        self.history_spin.setRange(2.0, 120.0)
        self.history_spin.setValue(self._initial_config.history_seconds)
        self.history_spin.setSuffix(" s")
        for label, widget in (("流名称", self.name_edit), ("流类型", self.type_edit), ("source_id（可选）", self.source_edit), ("逻辑通道 C", self.channel_spin), ("通道标签", self.labels_edit), ("内存 history", self.history_spin)):
            form.addRow(label, widget)
        layout.addLayout(form)
        filter_title = QLabel("在线显示滤波")
        filter_title.setObjectName("dialogTitle")
        layout.addWidget(filter_title)
        filter_note = QLabel(
            "只影响 Viewer 显示与统计；raw history 始终保留。参数应用后仅处理新到数据。"
        )
        filter_note.setObjectName("mutedLabel")
        filter_note.setWordWrap(True)
        layout.addWidget(filter_note)
        filter_form = QFormLayout()
        filter_form.setContentsMargins(0, 6, 0, 2)
        filter_form.setSpacing(8)
        self.bandpass_enable_check = QCheckBox("启用在线带通")
        self.bandpass_enable_check.setChecked(self._bandpass_enabled)
        self.bandpass_low_spin = QDoubleSpinBox()
        self.bandpass_low_spin.setDecimals(1)
        self.bandpass_low_spin.setRange(0.1, PHYSICAL_SAMPLE_RATE_HZ / 2.0 - 0.1)
        self.bandpass_low_spin.setValue(DEFAULT_BANDPASS_LOW_HZ)
        self.bandpass_low_spin.setSuffix(" Hz")
        self.bandpass_high_spin = QDoubleSpinBox()
        self.bandpass_high_spin.setDecimals(1)
        self.bandpass_high_spin.setRange(0.1, PHYSICAL_SAMPLE_RATE_HZ / 2.0 - 0.1)
        self.bandpass_high_spin.setValue(DEFAULT_BANDPASS_HIGH_HZ)
        self.bandpass_high_spin.setSuffix(" Hz")
        fixed_order = QLabel(f"{DISPLAY_BANDPASS_ORDER} 阶单向因果 Butterworth（SOS）")
        fixed_order.setObjectName("mutedLabel")
        filter_form.addRow("状态", self.bandpass_enable_check)
        filter_form.addRow("低截止", self.bandpass_low_spin)
        filter_form.addRow("高截止", self.bandpass_high_spin)
        filter_form.addRow("固定实现", fixed_order)
        layout.addLayout(filter_form)
        filter_action = QHBoxLayout()
        self.bandpass_status = QLabel(self._bandpass_description())
        self.bandpass_status.setObjectName("settingsStatus")
        self.bandpass_status.setWordWrap(True)
        self.bandpass_status.setStyleSheet(f"color:{GREEN};")
        self.bandpass_apply_button = QPushButton("应用显示滤波")
        self.bandpass_apply_button.setObjectName("primaryButton")
        filter_action.addWidget(self.bandpass_status, 1)
        filter_action.addWidget(self.bandpass_apply_button)
        layout.addLayout(filter_action)
        self.settings_status = QLabel("未连接")
        self.settings_status.setObjectName("settingsStatus")
        self.settings_status.setWordWrap(True)
        layout.addWidget(self.settings_status)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.disconnect_button = QPushButton("断开")
        self.connect_button = QPushButton("连接")
        self.connect_button.setObjectName("primaryButton")
        self.disconnect_button.setEnabled(False)
        buttons.addWidget(self.disconnect_button)
        buttons.addWidget(self.connect_button)
        layout.addLayout(buttons)
        self._config_widgets = [self.name_edit, self.type_edit, self.source_edit, self.channel_spin, self.labels_edit, self.history_spin]
        self.connect_button.clicked.connect(self.connect_stream)
        self.disconnect_button.clicked.connect(self.disconnect_stream)
        self.bandpass_apply_button.clicked.connect(self._apply_bandpass_settings)
        self.settings_dialog = dialog

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ color:{TEXT}; font-size:13px; }} QWidget#central {{ background:{BACKGROUND}; }}
            QFrame#headerCard, QFrame#toolbarCard, QFrame#card, QFrame#healthBar {{ background:#FFFFFF; border:1px solid {CARD_BORDER}; border-radius:10px; }}
            QFrame#cardHeader {{ background:transparent; border:none; border-bottom:1px solid #E7EAF0; }}
            QLabel#appLogo {{ background:{PRIMARY}; color:white; border-radius:8px; font-size:22px; font-weight:700; padding:5px 9px; }}
            QLabel#appTitle {{ font-size:20px; font-weight:700; color:#101828; }}
            QLabel#statusDot {{ color:{MUTED}; font-size:14px; }} QLabel#connectionStatus {{ font-weight:600; }}
            QLabel#headerMetricTitle, QLabel#healthTitle, QLabel#metricTitle, QLabel#metricUnit, QLabel#mutedLabel {{ color:{MUTED}; font-size:11px; }}
            QLabel#headerMetricValue {{ font-weight:600; color:#344054; }}
            QLabel#cardTitle, QLabel#dialogTitle {{ font-size:14px; font-weight:700; color:#1D2939; }} QLabel#dialogTitle {{ font-size:18px; }}
            QLabel#pageLabel {{ color:#344054; font-weight:600; }} QLabel#channelLegend {{ color:#667085; font-size:11px; }}
            QLabel#metricValue {{ font-size:21px; font-weight:600; color:#1D2939; }} QLabel#healthValue {{ font-size:13px; font-weight:600; }} QLabel#healthState {{ font-size:12px; font-weight:600; }}
            QFrame#metricCard {{ background:#F8FAFC; border:1px solid #E3E8EF; border-radius:8px; }}
            QFrame#healthCell {{ border:none; border-left:1px solid #E7EAF0; border-radius:0; }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background:#FFFFFF; border:1px solid #D0D5DD; border-radius:7px; padding:6px 9px; min-height:20px; }}
            QPushButton {{ background:#FFFFFF; border:1px solid #D0D5DD; border-radius:7px; padding:6px 12px; }}
            QPushButton:hover {{ background:#F8FAFC; border-color:#98A2B3; }} QPushButton:disabled {{ color:#98A2B3; background:#F2F4F7; }}
            QPushButton#primaryButton {{ background:{PRIMARY}; color:white; border-color:{PRIMARY}; font-weight:600; }}
            QPushButton#settingsButton {{ padding:7px 13px; }} QPushButton#pauseButton:checked {{ background:#EDF3FF; color:{PRIMARY}; border-color:#A9C2FA; }}
            QPushButton[channelButton="true"] {{ padding:6px 4px; background:#F7F8FA; border-color:#E2E6EC; }}
            QPushButton[channelButton="true"]:checked {{ background:{PRIMARY}; color:#FFFFFF; border-color:{PRIMARY}; font-weight:600; }}
            QScrollArea#channelScroll {{ border:none; background:transparent; }} QSplitter::handle {{ background:{BACKGROUND}; }}
            QSplitter::handle:horizontal {{ width:8px; }} QSplitter::handle:vertical {{ height:8px; }}
        """)

    def _show_settings(self) -> None:
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def _config_from_ui(self) -> MonitorConfig:
        labels = parse_channel_labels(self.labels_edit.text(), self.channel_spin.value())
        return MonitorConfig(
            stream_name=self.name_edit.text(), stream_type=self.type_edit.text(),
            source_id=self.source_edit.text() or None, channel_count=self.channel_spin.value(),
            channel_labels=labels, history_seconds=self.history_spin.value(),
            resolve_timeout_s=self._initial_config.resolve_timeout_s,
            info_timeout_s=self._initial_config.info_timeout_s,
            pull_timeout_s=self._initial_config.pull_timeout_s,
        ).validated()

    def _set_config_locked(self, locked: bool) -> None:
        for widget in self._config_widgets:
            widget.setEnabled(not locked)
        self.connect_button.setEnabled(not locked)
        self.disconnect_button.setEnabled(locked)

    def _bandpass_description(self) -> str:
        if not self._bandpass_enabled:
            return "未滤波"
        config = self._bandpass_config
        return (
            f"在线带通 {config.low_hz:g}–{config.high_hz:g} Hz · "
            f"{config.order} 阶因果 Butterworth"
        )

    def _set_bandpass_status(self, *, newly_applied: bool = False) -> None:
        description = self._bandpass_description()
        if self._bandpass_enabled and newly_applied:
            description += "；新参数仅作用于应用后的新数据"
        self.bandpass_status.setText(description)
        self.bandpass_status.setStyleSheet(f"color:{GREEN if self._bandpass_enabled else MUTED};")

    def _apply_bandpass_settings(self) -> bool:
        """Atomically apply valid display-filter settings, preserving raw history."""

        if not self.bandpass_enable_check.isChecked():
            self._bandpass_enabled = False
            self._display_bandpass.reset()
            self.filtered_history.clear()
            self._cached_stats = tuple(
                np.full(len(self._labels), np.nan) for _ in range(3)
            )
            self._set_bandpass_status()
            self._update_channel_header()
            self._refresh_display(force=True)
            return True

        try:
            config = DisplayBandpassConfig(
                low_hz=self.bandpass_low_spin.value(),
                high_hz=self.bandpass_high_spin.value(),
            ).validated()
            replacement = OnlineDisplayBandpass(self.history.channel_count, config)
        except (TypeError, ValueError) as error:
            self.bandpass_status.setText(f"参数错误：{error}；当前滤波配置未改变")
            self.bandpass_status.setStyleSheet(f"color:{RED};")
            return False

        self._bandpass_config = config
        self._display_bandpass = replacement
        self.filtered_history.clear()
        self._bandpass_enabled = True
        self._cached_stats = tuple(np.full(len(self._labels), np.nan) for _ in range(3))
        self._set_bandpass_status(newly_applied=True)
        self._update_channel_header()
        self._refresh_display(force=True)
        return True

    def _set_connection_state(self, state: str, text: str, detail: str | None = None) -> None:
        colors = {"disconnected": MUTED, "discovering": ORANGE, "connected": GREEN, "receiving": GREEN, "interrupted": RED, "error": RED}
        self.status_dot.setStyleSheet(f"color:{colors[state]}; font-size:14px;")
        self.connection_label.setText(text)
        self.connection_label.setToolTip(detail or text)
        if hasattr(self, "settings_status"):
            self.settings_status.setText(detail or text)

    def connect_stream(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            config = self._config_from_ui()
        except (TypeError, ValueError) as error:
            self._set_connection_state("error", "配置错误", f"配置错误：{error}")
            return
        self.history = RawEmgHistory(config.channel_count, config.history_seconds)
        self.filtered_history = RawEmgHistory(
            config.channel_count, config.history_seconds
        )
        self._display_bandpass = OnlineDisplayBandpass(
            config.channel_count, self._bandpass_config
        )
        self._raw_labels = config.channel_labels
        self._set_display_mode_availability()
        self._apply_display_mode(self.display_mode_combo.currentText())
        self._cached_stats = tuple(np.full(len(self._labels), np.nan) for _ in range(3))
        self.selected_channel = 0
        self._page = 0
        self._last_health = {}
        self._time_correction_s = None
        self._worker_failed = False
        self._ensure_channel_buttons()
        self._update_page_controls()
        self.stream_header.value_label.setText(config.stream_name)
        self._update_channel_header()
        self._set_config_locked(True)
        self._set_connection_state("discovering", "正在连接", "正在发现并校验 LSL header…")
        self.worker = self._worker_factory(config)
        self._bind_worker(self.worker)
        self.worker.start()

    def _bind_worker(self, worker: RawStreamWorker) -> None:
        # Tokens remain valid even if Qt has destroyed an old signal sender.
        token = object()
        self._connection_token = token
        for signal, handler in (
            (worker.connected, self._on_connected),
            (worker.batch_received, self._on_batch),
            (worker.health_updated, self._on_health),
            (worker.status_changed, self._on_status),
            (worker.failed, self._on_failed),
            (worker.finished, self._on_worker_finished),
        ):
            def receive(*args, callback=handler, generation=token):
                if self._connection_token is generation:
                    callback(*args)
            signal.connect(receive, Qt.ConnectionType.QueuedConnection)

    def disconnect_stream(self) -> bool:
        worker = self.worker
        if worker is None:
            self._display_bandpass.reset()
            self._set_config_locked(False)
            return True
        if worker.isRunning():
            worker.request_stop()
            wait_ms = round(max(0.2, worker.config.pull_timeout_s, worker.config.resolve_timeout_s, worker.config.info_timeout_s) * 1000) + 1500
            if not worker.wait(wait_ms):
                self._set_connection_state("error", "停止超时", "后台线程仍在停止；为保护 QThread 生命周期，暂不关闭")
                return False
        self._connection_token = None
        self.worker = None
        self._display_bandpass.reset()
        self._set_config_locked(False)
        self._set_connection_state("disconnected", "已断开")
        self.health_state_dot.setStyleSheet(f"color:{MUTED};")
        self.health_state_label.setText("等待连接")
        return True

    def _on_connected(self, identity: dict[str, Any]) -> None:
        self._time_correction_s = float(identity.get("time_correction_s", 0.0))
        detail = (
            f"已连接：{identity['name']} · source_id={identity['source_id'] or '<empty>'} · "
            f"C={len(self._raw_labels)} / LSL width={identity['channel_count']} · "
            f"packet float32 @ {identity['nominal_hz']:g} pkt/s · time correction={self._time_correction_s:+.6f} s"
        )
        self._set_connection_state("connected", "已连接", detail)
        self.health_cells["correction"].set_value(f"{self._time_correction_s:+.6f} s")

    def _on_batch(self, packets: Sequence[DecodedPacket]) -> None:
        try:
            for packet in packets:
                self.history.append(packet)
                if self._bandpass_enabled:
                    filtered = self._display_bandpass.process_packet(packet)
                    self.filtered_history.append(filtered)
        except ValueError as error:
            self._set_connection_state("error", "数据错误", f"history contract error：{error}")

    def _on_health(self, health: dict[str, Any]) -> None:
        self._last_health = dict(health)
        age = health.get("last_sample_age_ms")
        rate = float(health.get("actual_rate_hz", 0.0))
        rx_total = int(health.get("rx_total", 0))
        if rx_total == 0 or age is None:
            state, state_color, connection_state = "等待首包", MUTED, "connected"
        elif float(age) > 3 * PACKET_DURATION_SECONDS * 1000:
            state, state_color, connection_state = "数据中断", RED, "interrupted"
        else:
            state, state_color, connection_state = "接收中", GREEN, "receiving"
        self.health_state_label.setText(state)
        self.health_state_dot.setStyleSheet(f"color:{state_color};")
        self._set_connection_state(connection_state, state)
        age_text = "-- ms" if age is None else f"{float(age):.0f} ms"
        self.packet_rate_header.value_label.setText(f"{rate:.1f} pkt/s" if rx_total else "-- pkt/s")
        self.age_header.value_label.setText(age_text)
        counters = {
            "gap": int(health.get("gap_total", 0)),
            "out_of_order": int(health.get("out_of_order_total", 0)),
            "invalid": int(health.get("invalid_total", 0)),
            "disconnect": max(self._disconnect_total, int(health.get("lost_total", 0))),
        }
        for key, value in counters.items():
            self.health_cells[key].set_value(str(value), "error" if value else "healthy")
        self.health_cells["packet_rate"].set_value(f"{rate:.1f} pkt/s" if rx_total else "-- pkt/s", "healthy" if rx_total and state == "接收中" else "neutral")
        self.health_cells["age"].set_value(age_text, "healthy" if state == "接收中" else ("error" if state == "数据中断" else "neutral"))
        self.health_label.setText(
            f"{state} · packet rate={rate:g} pkt/s · data age={age_text} · gap={counters['gap']} · "
            f"out-of-order={counters['out_of_order']} · invalid={counters['invalid']} · disconnect={counters['disconnect']}"
        )

    def _on_status(self, status: str) -> None:
        if status.startswith(("contract error", "header rejected")):
            self._set_connection_state("error", "流契约错误", status)

    def _on_failed(self, message: str) -> None:
        self._display_bandpass.reset()
        self._worker_failed = True
        self._disconnect_total += 1
        self.health_cells["disconnect"].set_value(str(self._disconnect_total), "error")
        self.health_state_dot.setStyleSheet(f"color:{RED};")
        self.health_state_label.setText("接收失败")
        self._set_connection_state("error", "连接/接收错误", message)

    def _on_worker_finished(self) -> None:
        self._display_bandpass.reset()
        self._set_config_locked(False)
        self.disconnect_button.setEnabled(False)
        if not self._worker_failed and self.connection_label.text() not in {"已断开", "未连接"}:
            self._set_connection_state("disconnected", "连接已结束")

    def _page_size(self) -> int:
        return int(self.page_size_combo.currentText())

    def _set_display_mode_availability(self) -> None:
        """Enable Bipolar-32 only when labels satisfy the explicit montage contract."""

        empty = np.empty((len(self._raw_labels), 0), dtype=np.float32)
        try:
            transform_display_values(empty, self._raw_labels, DISPLAY_MODE_BIPOLAR32)
        except ValueError as error:
            bipolar_item = self.display_mode_combo.model().item(1)
            if bipolar_item is not None:
                bipolar_item.setEnabled(False)
                bipolar_item.setToolTip(str(error))
            if self.display_mode_combo.currentText() == DISPLAY_MODE_BIPOLAR32:
                self.display_mode_combo.setCurrentText(DISPLAY_MODE_RAW)
            self.display_mode_combo.setToolTip(f"Bipolar-32 不可用：{error}")
        else:
            bipolar_item = self.display_mode_combo.model().item(1)
            if bipolar_item is not None:
                bipolar_item.setEnabled(True)
                bipolar_item.setToolTip("固定 SAGA 标签配对；仅做正端减负端。")
            self.display_mode_combo.setToolTip("Raw-64 或固定 signed Bipolar-32 显示；不改变输入 history。")

    def _apply_display_mode(self, mode: str) -> None:
        empty = np.empty((len(self._raw_labels), 0), dtype=np.float32)
        labels, _values = transform_display_values(empty, self._raw_labels, mode)
        self._display_mode = mode
        self._labels = labels

    def _on_display_mode_changed(self, mode: str) -> None:
        try:
            self._apply_display_mode(mode)
        except ValueError as error:
            self.display_mode_combo.blockSignals(True)
            self.display_mode_combo.setCurrentText(self._display_mode)
            self.display_mode_combo.blockSignals(False)
            self.display_mode_combo.setToolTip(f"Bipolar-32 不可用：{error}")
            return
        self.selected_channel = 0
        self._page = 0
        self._cached_stats = tuple(np.full(len(self._labels), np.nan) for _ in range(3))
        self._ensure_channel_buttons()
        self._update_channel_header()
        self._refresh_display(force=True)

    def _update_channel_header(self) -> None:
        filtering = self._bandpass_description()
        if self._display_mode == DISPLAY_MODE_BIPOLAR32:
            self.channel_header.value_label.setText("32 pairs")
            self.waveform_title.setText(f"signed Bipolar-32 EMG 波形 · {filtering}")
            self.detail_meta.setText(f"正端 − 负端 · {filtering} · 未进行 CAR/notch/QC")
        else:
            self.channel_header.value_label.setText(f"{len(self._raw_labels)} ch")
            self.waveform_title.setText(f"signed raw EMG 波形 · {filtering}")
            self.detail_meta.setText(f"signed raw · {filtering} · 未进行 QC")

    def _change_page(self, delta: int) -> None:
        pages = max(1, math.ceil(len(self._labels) / self._page_size()))
        self._page = min(max(0, self._page + delta), pages - 1)
        self._update_page_controls()
        self._refresh_display(force=True)

    def _on_page_size_changed(self, _text: str) -> None:
        self._page = min(self.selected_channel // self._page_size(), max(0, math.ceil(len(self._labels) / self._page_size()) - 1))
        self._update_page_controls()
        self._refresh_display(force=True)

    def _on_gain_mode_changed(self, text: str) -> None:
        self.fixed_gain_spin.setVisible(text == "Custom")
        self._refresh_display(force=True)

    def _on_pause_toggled(self, paused: bool) -> None:
        self.pause_check.setText("▶  继续" if paused else "Ⅱ  暂停")

    @staticmethod
    def _clear_layout(layout: QGridLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

    def _ensure_channel_buttons(self) -> None:
        if self._button_labels == self._labels and len(self.channel_buttons) == len(self._labels):
            self._update_channel_selection()
            return
        self._clear_layout(self.channel_matrix_layout)
        self.channel_buttons = []
        if self._display_mode == DISPLAY_MODE_BIPOLAR32:
            self.channel_matrix_layout.setHorizontalSpacing(2)
            for column in range(8):
                header = QLabel(f"C{column + 1}")
                header.setObjectName("matrixHeader")
                header.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.channel_matrix_layout.addWidget(header, 0, column + 1)
            for row, pair_rows in enumerate(("R1–R2", "R3–R4", "R5–R6", "R7–R8")):
                header = QLabel(pair_rows)
                header.setObjectName("matrixHeader")
                self.channel_matrix_layout.addWidget(header, row + 1, 0)
        for index, label in enumerate(self._labels):
            button_text = label
            if self._display_mode == DISPLAY_MODE_BIPOLAR32:
                _row, column = bipolar32_grid_position(index)
                button_text = f"C{column + 1}"
            button = QPushButton(button_text)
            button.setAccessibleName(label)
            button.setToolTip(label)
            if self._display_mode == DISPLAY_MODE_BIPOLAR32:
                button.setStyleSheet("padding:4px 1px;")
            button.setProperty("channelButton", True)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, channel=index: self._select_channel(channel))
            if self._display_mode == DISPLAY_MODE_BIPOLAR32:
                row, column = bipolar32_grid_position(index)
                row += 1
                column += 1
            else:
                row, column = index // 4, index % 4
            self.channel_matrix_layout.addWidget(button, row, column)
            self.channel_buttons.append(button)
        self._button_labels = self._labels
        if self._display_mode != DISPLAY_MODE_BIPOLAR32:
            self.channel_matrix_layout.setHorizontalSpacing(6)
        self.channel_count_label.setText(f"{len(self._labels)} 个")
        self._update_channel_selection()
        self._filter_channels(self.channel_search.text())

    def _rebuild_channel_matrix(self, *, reset_page: bool = False) -> None:
        """Compatibility hook; no widget rebuild when the channel set is unchanged."""
        if reset_page:
            self._page = 0
        self._ensure_channel_buttons()
        self._update_page_controls()

    def _update_channel_selection(self) -> None:
        for index, button in enumerate(self.channel_buttons):
            button.setChecked(index == self.selected_channel)

    def _filter_channels(self, text: str) -> None:
        query = text.strip().casefold()
        for index, button in enumerate(self.channel_buttons):
            button.setVisible(not query or query in self._labels[index].casefold())

    def _update_page_controls(self) -> None:
        page_size = self._page_size()
        pages = max(1, math.ceil(len(self._labels) / page_size))
        self._page = min(self._page, pages - 1)
        first = self._page * page_size
        last = min(first + page_size, len(self._labels))
        self.page_label.setText(f"通道 {first + 1}–{last} / {len(self._labels)} · 第 {self._page + 1}/{pages} 页")
        self.prev_button.setEnabled(self._page > 0)
        self.next_button.setEnabled(self._page + 1 < pages)
        self.waveform_meta.setText(
            f"显示 {last - first} / {len(self._labels)} 通道 · 时间窗口 {self.view_combo.currentText()} s · "
            f"幅度 {self.gain_mode_combo.currentText()} · {self._bandpass_description()}"
        )

    def _select_channel(self, index: int) -> None:
        self.selected_channel = int(index)
        self._page = self.selected_channel // self._page_size()
        self._update_channel_selection()
        self._update_page_controls()
        self._refresh_display(force=True)

    def _gain_for(self, values: np.ndarray) -> float:
        mode = self.gain_mode_combo.currentText()
        if mode == "Auto":
            finite_abs = np.abs(values[np.isfinite(values)]) if values.size else np.empty(0)
            value = float(np.percentile(finite_abs, 99)) if finite_abs.size else 1.0
            return max(value, 1e-6)
        if mode == "Custom":
            return self.fixed_gain_spin.value()
        return float(mode.removeprefix("±").split()[0])

    def _refresh_display(self, *, force: bool = False) -> None:
        if self.pause_check.isChecked() and not force:
            return
        seconds = float(self.view_combo.currentText())
        source_history = self.filtered_history if self._bandpass_enabled else self.history
        timestamps, raw_values = source_history.window(seconds)
        _labels, values = transform_display_values(
            raw_values, self._raw_labels, self._display_mode
        )
        self._stats_tick += 1
        if force or self._stats_tick % 4 == 0:
            self._cached_stats = raw_channel_statistics(values)
            rms, p2p, mean = self._cached_stats
            for index, button in enumerate(self.channel_buttons):
                button.setToolTip(
                    ("signed Bipolar-32（正端 − 负端；未进行 CAR/notch/QC）\n"
                     if self._display_mode == DISPLAY_MODE_BIPOLAR32
                     else "signed raw（未进行 QC）\n")
                    + self._bandpass_description() + "\n"
                    + f"RMS={_format_raw(rms[index])}\nP-P={_format_raw(p2p[index])}\nMean={_format_raw(mean[index])}"
                )
        rms, p2p, mean = self._cached_stats
        if self.selected_channel < len(self._labels):
            index = self.selected_channel
            self.detail_label.setText(f"选中通道详情 — {self._labels[index]}")
            self.metric_cards["rms"].set_value(rms[index])
            self.metric_cards["p2p"].set_value(p2p[index])
            self.metric_cards["mean"].set_value(mean[index])
        first = self._page * self._page_size()
        indices = list(range(first, min(first + self._page_size(), len(self._labels))))
        visible = values[indices] if indices else np.empty((0, values.shape[1]), dtype=np.float32)
        gain = self._gain_for(visible)
        selected_local = indices.index(self.selected_channel) if self.selected_channel in indices else -1
        self.canvas.set_data([self._labels[index] for index in indices], timestamps, visible, gain=gain, selected=selected_local, view_seconds=seconds)
        if self.selected_channel < values.shape[0]:
            detail_values = values[self.selected_channel:self.selected_channel + 1]
            self.detail_canvas.set_data([self._labels[self.selected_channel]], timestamps, detail_values, gain=self._gain_for(detail_values), selected=0, view_seconds=seconds)
        self._update_page_controls()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.refresh_timer.stop()
        self.settings_dialog.close()
        if self.disconnect_stream():
            event.accept()
        else:
            self.refresh_timer.start()
            event.ignore()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 Qt 只读监控 emg_raw_signed_2khz packetized raw EMG。")
    parser.add_argument("--stream-name", default=DEFAULT_STREAM_NAME)
    parser.add_argument("--stream-type", default=DEFAULT_STREAM_TYPE)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--channels", type=int, default=64, metavar="C")
    parser.add_argument(
        "--channel-labels",
        default="",
        help=(
            "本地通道标签，逗号分隔；64 通道缺省使用固定 SAGA-64 preset，"
            "其他 C 缺省 ch001...。raw wire 不携带标签。"
        ),
    )
    parser.add_argument("--history-seconds", type=float, default=DEFAULT_HISTORY_SECONDS)
    parser.add_argument("--resolve-timeout-s", type=float, default=0.5)
    parser.add_argument("--info-timeout-s", type=float, default=2.0)
    parser.add_argument("--pull-timeout-s", type=float, default=0.2)
    args = parser.parse_args(argv)
    try:
        labels = parse_channel_labels(args.channel_labels, args.channels)
        args.config = MonitorConfig(
            stream_name=args.stream_name, stream_type=args.stream_type,
            source_id=args.source_id, channel_count=args.channels,
            channel_labels=labels, history_seconds=args.history_seconds,
            resolve_timeout_s=args.resolve_timeout_s, info_timeout_s=args.info_timeout_s,
            pull_timeout_s=args.pull_timeout_s,
        ).validated()
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setApplicationName("Signed Raw EMG Monitor")
    font = QFont("SF Pro Text")
    font.setStyleHint(QFont.StyleHint.System)
    app.setFont(font)
    window = RawStreamMonitorWindow(args.config, auto_connect=True)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
