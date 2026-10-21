"""Qt mouse-controlled, packetized signed EMG source. No task/decoder required.

Run from this tool directory:
    uv run --locked python app.py
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace
import math
import signal
import sys
import threading
import time

import numpy as np
from PySide6.QtCore import QEvent, QPointF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton, QScrollArea,
    QToolButton, QVBoxLayout, QWidget,
)

from mouse_emg import AmplitudeMouseEmg, MouseSourceConfig, SpringBall
from packet import flatten_packet

STREAM_NAME = "emg_raw_signed_2khz"
PERIOD = 0.020


def create_outlet(config):
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(STREAM_NAME, "EMG", config.channels * 40, 50.0, "float32", config.source_id)
    return StreamOutlet(info, chunk_size=1, max_buffered=30)


class MousePublisher(QThread):
    ready = Signal()
    progress = Signal(int, float, int)
    failed = Signal(str)

    def __init__(self, config, ball, *, outlet_factory=create_outlet, lsl_clock=None):
        super().__init__()
        self.config = config.validate()
        self.ball = ball
        self.outlet_factory = outlet_factory
        self.lsl_clock = lsl_clock
        self.stop_event = threading.Event()

    def request_stop(self):
        self.stop_event.set()

    def run(self):
        outlet = None
        try:
            generator = AmplitudeMouseEmg(self.config)
            outlet = self.outlet_factory(self.config)
            if self.lsl_clock is None:
                from pylsl import local_clock
                clock = local_clock
            else:
                clock = self.lsl_clock
            # Schedule and timestamp in the same clock domain. On macOS,
            # Python monotonic and LSL may advance differently across sleep.
            lsl_epoch = clock()
            slot, count, late = 1, 0, 0
            pushes = deque()
            last_report = time.monotonic()
            self.ready.emit()
            while not self.stop_event.is_set():
                deadline = lsl_epoch + slot * PERIOD
                if self.stop_event.wait(max(0.0, deadline - clock())):
                    break
                now = clock()
                # Skip expired slots rather than backfill with current mouse input.
                missed = max(0, math.floor((now - deadline) / PERIOD))
                slot += missed
                late += missed
                packet = generator.packet(self.ball.sample(), sample_index=(slot - 1) * 40)
                outlet.push_sample(flatten_packet(packet).tolist(), timestamp=lsl_epoch + slot * PERIOD)
                count += 1
                pushed = time.monotonic()
                pushes.append(pushed)
                while len(pushes) > 1 and pushes[0] < pushed - 2.0:
                    pushes.popleft()
                rate = (len(pushes) - 1) / (pushes[-1] - pushes[0]) if len(pushes) > 1 else 0.0
                if pushed - last_report >= 0.2 or count == 1:
                    self.progress.emit(count, rate, late)
                    last_report = pushed
                slot += 1
            self.progress.emit(count, 0.0, late)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if outlet is not None:
                try:
                    if hasattr(outlet, "close"):
                        outlet.close()
                finally:
                    del outlet


class BallCanvas(QWidget):
    """Resolution-independent Qt canvas; display and publishing share one model."""

    def __init__(self, ball, parent=None):
        super().__init__(parent)
        self.ball = ball
        self.compact = False
        self.setMinimumSize(160, 160)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._dragging = False
        self._offset = QPointF()

    def geometry_values(self):
        radius = min(self.width(), self.height()) / 2 - 20 if self.compact else min(self.width() * 0.36, self.height() * 0.40)
        return QPointF(self.width() / 2, self.height() / 2), radius

    def ball_point(self):
        center, radius = self.geometry_values()
        x, y = self.ball.sample()
        return center + QPointF(x * radius, -y * radius)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        center, radius = self.geometry_values()
        p.setPen(QPen(QColor("#d5d9df"), 1.1))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(center, radius, radius)
        reach = radius if self.compact else radius * 1.12
        p.drawLine(center + QPointF(-reach, 0), center + QPointF(reach, 0))
        p.drawLine(center + QPointF(0, reach), center + QPointF(0, -reach))
        if not self.compact:
            p.setBrush(QColor("#969eaa"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawPolygon(QPolygonF([center + QPointF(reach + 4, 0), center + QPointF(reach - 3, -3), center + QPointF(reach - 3, 3)]))
            p.drawPolygon(QPolygonF([center + QPointF(0, -reach - 4), center + QPointF(-3, -reach + 3), center + QPointF(3, -reach + 3)]))
            p.setPen(QColor("#606b7a"))
            p.drawText(center + QPointF(reach + 14, 5), "+x")
            p.drawText(center + QPointF(14, -reach + 3), "+y")
        point = self.ball_point()
        if np.hypot(point.x() - center.x(), point.y() - center.y()) > 19:
            p.setPen(QPen(QColor("#007aff"), 1.8))
            p.drawLine(center, point)
            p.drawText((center + point) * 0.5 + QPointF(0, -12), "v")
        p.setPen(QPen(QColor("#444d58"), 3.0))
        p.drawLine(center + QPointF(-10, 0), center + QPointF(10, 0))
        p.drawLine(center + QPointF(0, -10), center + QPointF(0, 10))
        p.setPen(QPen(QColor("#ffffff"), 1.5))
        p.setBrush(QColor("#007aff"))
        p.drawEllipse(point, 12, 12)

    def mousePressEvent(self, event):
        delta = event.position() - self.ball_point()
        if event.button() == Qt.MouseButton.LeftButton and math.hypot(delta.x(), delta.y()) <= 20:
            self._dragging = True
            self._offset = delta
            self.ball.drag(self.ball.sample())
            self.setFocus()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event):
        if self._dragging:
            center, radius = self.geometry_values()
            point = event.position() - self._offset - center
            self.ball.drag((point.x() / radius, -point.y() / radius))
            self.update()
        else:
            delta = event.position() - self.ball_point()
            self.setCursor(Qt.CursorShape.OpenHandCursor if math.hypot(delta.x(), delta.y()) <= 20 else Qt.CursorShape.ArrowCursor)

    def end_drag(self):
        self._dragging = False
        self.ball.release()
        self.unsetCursor()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.end_drag()

    def focusOutEvent(self, event):
        self.end_drag()
        super().focusOutEvent(event)


STYLE = """
QMainWindow, QWidget#root { background: #ffffff; color: #191b20; }
QWidget { font-size: 15px; }
QLabel { color: #20242b; background: transparent; }
QFrame#header, QFrame#footer { background: #f4f5f7; }
QFrame#header { border-bottom: 1px solid #d5d9df; }
QFrame#footer { border-top: 1px solid #d5d9df; }
QFrame#sidebar { background: #f1f2f4; border-left: 1px solid #d5d9df; }
QFrame#controls { background: #f1f2f4; border-top: 1px solid #d5d9df; border-left: 1px solid #d5d9df; }
QLabel[muted="true"] { color: #67717f; }
QLabel[heading="true"] { font-size: 18px; font-weight: 600; }
QFrame#rule { background: #daddE2; max-height: 1px; }
QFrame#readout { background: #f8f8fa; border: 1px solid #d8dce2; border-radius: 8px; }
QLabel[value="true"] { font-size: 22px; font-weight: 600; }
QLabel#streamName { background: #edeff2; border: 1px solid #d9dde3; border-radius: 6px; padding: 7px; font-size: 12px; }
QPushButton { background: #ffffff; border: 1px solid #c9cdd4; border-radius: 7px; padding: 9px 14px; color: #17191d; }
QPushButton:hover { background: #e8eef8; }
QPushButton:pressed { background: #dce8fa; }
QPushButton:disabled { color: #969ca5; background: #eceef0; }
QPushButton#stop { color: #d70015; border-color: #ef6470; }
QPushButton#apply { background: #007aff; border-color: #007aff; color: #ffffff; }
QPushButton#apply:hover { background: #006be0; }
QPushButton#apply:disabled { background: #a0bfe5; border-color: #a0bfe5; }
QLineEdit { background: #ffffff; color: #20242b; border: 1px solid #c9cdd4; border-radius: 5px; padding: 6px; }
QLineEdit:focus { border-color: #007aff; }
QLineEdit:disabled { color: #969ca5; background: #eceef0; }
QToolButton#sidebarToggle { border: none; border-radius: 5px; background: transparent; }
QToolButton#sidebarToggle:hover { background: #e2e5ea; }
QToolButton#sidebarToggle:checked { background: #e8eaee; }
"""


def sidebar_icon():
    """macOS-style split-panel glyph, drawn at 2x for a crisp toolbar icon."""
    pixmap = QPixmap(48, 40)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor("#626b77"), 1.4))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRoundedRect(2, 3, 20, 14, 2, 2)
    painter.drawLine(15, 3, 15, 17)
    painter.fillRect(17, 6, 3, 8, QColor("#a1a8b1"))
    painter.end()
    return QIcon(pixmap)


def label(text, *, muted=False, heading=False):
    widget = QLabel(text)
    widget.setProperty("muted", muted)
    widget.setProperty("heading", heading)
    return widget


def rule(layout):
    line = QFrame()
    line.setObjectName("rule")
    line.setFixedHeight(1)
    layout.addWidget(line)


class MockMouseEmgWindow(QMainWindow):
    def __init__(self, config, *, auto_start=True, publisher_factory=MousePublisher):
        super().__init__()
        self.config = config.validate()
        self.ball = SpringBall(config.return_tau)
        self.publisher_factory = publisher_factory
        self.publisher = None
        # Keep the worker active until its queued finished signal is handled.
        self._stream_active = False
        self._pending_config = None
        self._closing = False
        self._error = False
        self.sidebar_visible = True
        self._expanded_size = QSize(1280, 880)
        self._compact_size = QSize(400, 400)
        self._status_name = "Starting"
        self.setWindowTitle("Mock Mouse EMG")
        self.resize(1280, 880)
        self.setMinimumSize(1040, 760)
        self.setStyleSheet(STYLE)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = QFrame()
        header.setObjectName("header")
        top = QHBoxLayout(header)
        self.header_layout = top
        top.setContentsMargins(24, 14, 24, 14)
        self.header_left = QWidget()
        self.header_left.setFixedWidth(150)
        toggle_layout = QHBoxLayout(self.header_left)
        toggle_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_toggle = QToolButton()
        self.sidebar_toggle.setObjectName("sidebarToggle")
        self.sidebar_toggle.setIcon(sidebar_icon())
        self.sidebar_toggle.setIconSize(QSize(24, 20))
        self.sidebar_toggle.setFixedSize(32, 28)
        self.sidebar_toggle.setCheckable(True)
        self.sidebar_toggle.setChecked(True)
        self.sidebar_toggle.setToolTip("Hide Sidebar")
        self.sidebar_toggle.setAccessibleName("Toggle Sidebar")
        self.sidebar_toggle.toggled.connect(self.set_sidebar_visible)
        toggle_layout.addWidget(self.sidebar_toggle)
        toggle_layout.addStretch()
        top.addWidget(self.header_left)
        top.addStretch(1)
        title = label("Mock Mouse EMG", heading=True)
        self.header_title = title
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top.addWidget(title)
        top.addStretch(1)
        self.header_status = label("●  Starting", muted=True)
        self.header_status.setFixedWidth(150)
        self.header_status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top.addWidget(self.header_status)
        outer.addWidget(header)
        body = QHBoxLayout()
        body.setSpacing(0)
        outer.addLayout(body, 1)
        left = QVBoxLayout()
        self.canvas_layout = left
        left.setContentsMargins(24, 12, 24, 30)
        left.setSpacing(18)
        self.canvas = BallCanvas(self.ball)
        left.addWidget(self.canvas, 1)
        hint = label("Drag to move. Release to return.", muted=True)
        self.hint = hint
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left.addWidget(hint)
        readout = QFrame()
        self.readout = readout
        readout.setObjectName("readout")
        readout.setFixedWidth(460)
        values_layout = QHBoxLayout(readout)
        values_layout.setContentsMargins(20, 12, 20, 12)
        self.displacement_labels = []
        for index, title in enumerate(("x", "y", "|v|")):
            if index:
                line = QFrame()
                line.setFixedWidth(1)
                line.setStyleSheet("background: #d8dce2")
                values_layout.addWidget(line)
            column = QVBoxLayout()
            name = label(title, muted=True)
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value = label("0.00")
            value.setProperty("value", True)
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(name)
            column.addWidget(value)
            values_layout.addLayout(column, 1)
            self.displacement_labels.append(value)
        left.addWidget(readout, 0, Qt.AlignmentFlag.AlignHCenter)
        body.addLayout(left, 1)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(24, 20, 24, 20)
        side.setSpacing(6)
        side.addWidget(label("Source Parameters", heading=True))
        side.addWidget(label("Edit values, then apply.", muted=True))
        rule(side)

        def row(name, value):
            layout = QHBoxLayout()
            layout.addWidget(label(name), 1)
            result = label(value)
            result.setMinimumWidth(142)
            layout.addWidget(result)
            side.addLayout(layout)
            rule(side)
            return result

        self.parameter_inputs = {}
        for key, title, hint in (
            ("mapping_norm", "Amplitude norm", "Directional amplitude map norm; ≥ 0. Zero keeps rest carriers."),
            ("noise_level", "Noise std", "Independent Gaussian noise standard deviation; ≥ 0."),
            ("return_tau", "Return time (s)", "Exponential return time constant in seconds; > 0."),
            ("seed", "Seed", "Non-negative integer controlling carrier phases and noise."),
        ):
            layout = QHBoxLayout()
            name = label(title)
            field = QLineEdit(str(getattr(config, key)))
            field.setFixedWidth(142)
            field.setAccessibleName(title)
            field.setToolTip(hint)
            field.returnPressed.connect(self.apply_parameters)
            field.textEdited.connect(self._parameters_edited)
            name.setBuddy(field)
            layout.addWidget(name, 1)
            layout.addWidget(field)
            side.addLayout(layout)
            self.parameter_inputs[key] = field
        source_id = QLineEdit(config.source_id)
        source_id.setAccessibleName("Source ID")
        source_id.setToolTip("Use a unique LSL Source ID for each concurrent instance.")
        source_id.returnPressed.connect(self.apply_parameters)
        source_id.textEdited.connect(self._parameters_edited)
        source_id.setCursorPosition(0)
        source_label = label("Source ID", muted=True)
        source_label.setBuddy(source_id)
        side.addWidget(source_label)
        side.addWidget(source_id)
        self.parameter_inputs["source_id"] = source_id
        self.apply_button = QPushButton("Apply Parameters")
        self.apply_button.setObjectName("apply")
        self.apply_button.clicked.connect(self.apply_parameters)
        side.addWidget(self.apply_button)
        self.parameters_note = label("Applying restarts a running stream.\nChanges last for this session.", muted=True)
        self.parameters_note.setWordWrap(True)
        self.parameters_note.setMinimumHeight(40)
        side.addWidget(self.parameters_note)
        side.addSpacing(12)
        side.addWidget(label("Output", heading=True))
        rule(side)
        side.addWidget(label("64 channels · 2000 Hz"))
        side.addWidget(label("40 samples / packet · 50 packets/s"))
        side.addWidget(label("80 Hz carrier · 4 directions", muted=True))
        stream_name = label(STREAM_NAME)
        stream_name.setObjectName("streamName")
        stream_name.setMinimumHeight(32)
        stream_name.setFont(QFont("Menlo" if sys.platform == "darwin" else "monospace"))
        stream_name.setToolTip("LSL stream name")
        side.addWidget(stream_name)
        rule(side)
        side.addSpacing(12)
        side.addWidget(label("Live Status", heading=True))
        self.sending_status = label("●  Starting", muted=True)
        side.addWidget(self.sending_status)
        rule(side)
        self.rate_label = row("Actual rate", "0.0 packets/s")
        self.count_label = row("Packets sent", "0")
        self.late_label = row("Late packets", "0")
        self.late_label.setToolTip("Expired 20 ms slots skipped due to sender scheduling delays.")
        side.addStretch()
        buttons = QHBoxLayout()
        buttons.setContentsMargins(24, 12, 24, 12)
        self.center_button = QPushButton("Center Ball")
        self.center_button.setMinimumHeight(40)
        self.center_button.clicked.connect(self.center_ball)
        self.stop_button = QPushButton("Stop Stream")
        self.stop_button.setMinimumHeight(40)
        self.stop_button.setObjectName("stop")
        self.stop_button.clicked.connect(self.toggle_stream)
        buttons.addWidget(self.center_button)
        buttons.addWidget(self.stop_button)
        inspector = QWidget()
        self.inspector = inspector
        inspector.setFixedWidth(368)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(sidebar)
        inspector_layout.addWidget(scroll, 1)
        controls = QFrame()
        controls.setObjectName("controls")
        controls.setLayout(buttons)
        inspector_layout.addWidget(controls)
        body.addWidget(inspector)
        footer = QFrame()
        self.footer = footer
        footer.setObjectName("footer")
        bottom = QHBoxLayout(footer)
        bottom.setContentsMargins(20, 12, 20, 12)
        self.footer_status = label("Starting LSL outlet…", muted=True)
        self.footer_status.setWordWrap(True)
        bottom.addWidget(self.footer_status, 1)
        bottom.addWidget(label("Synthetic source", muted=True))
        outer.addWidget(footer)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_motion)
        self.timer.start(16)
        if auto_start:
            QTimer.singleShot(0, self.start_stream)
        else:
            self._finished()

    def set_sidebar_visible(self, visible):
        if visible == self.sidebar_visible:
            return
        if visible:
            self._compact_size = self.size()
        else:
            self._expanded_size = self.size()
        self.sidebar_visible = visible
        self.sidebar_toggle.setChecked(visible)
        self.sidebar_toggle.setToolTip("Hide Sidebar" if visible else "Show Sidebar")
        for widget in (self.inspector, self.hint, self.readout, self.footer, self.header_title):
            widget.setVisible(visible)
        self.canvas.compact = not visible
        self.header_left.setFixedWidth(150 if visible else 32)
        self.header_status.setFixedWidth(150 if visible else 14)
        self.header_status.setText("●  " + self._status_name if visible else "●")
        self.header_layout.setContentsMargins(*( (24, 14, 24, 14) if visible else (8, 4, 12, 4) ))
        self.canvas_layout.setContentsMargins(*( (24, 12, 24, 30) if visible else (0, 0, 0, 0) ))
        self.canvas_layout.setSpacing(18 if visible else 0)
        self.setMinimumSize(1040, 760) if visible else self.setMinimumSize(300, 340)
        self.centralWidget().layout().activate()
        self.resize(self._expanded_size if visible else self._compact_size)
        self.canvas.update()

    def refresh_motion(self):
        v = self.ball.sample()
        for widget, value in zip(self.displacement_labels, (v[0], v[1], np.linalg.norm(v))):
            widget.setText(f"{0.0 if abs(value) < 0.0005 else value:+.2f}" if widget is not self.displacement_labels[-1] else f"{value:.2f}")
        self.canvas.update()

    def center_ball(self):
        self.canvas.end_drag()
        self.ball.center()
        self.refresh_motion()

    def _enable_parameters(self, enabled):
        for field in self.parameter_inputs.values():
            field.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)

    def _parameters_message(self, text, *, error=False):
        self.parameters_note.setText(text)
        self.parameters_note.setStyleSheet("color: #cf2639" if error else "color: #67717f")

    def _parameters_edited(self):
        self._parameters_message("Unapplied changes. Click Apply Parameters.")

    def apply_parameters(self):
        if self._closing or not self.apply_button.isEnabled():
            return
        values = {}
        try:
            for key, convert in (("mapping_norm", float), ("noise_level", float),
                                 ("return_tau", float), ("seed", int)):
                field = self.parameter_inputs[key]
                try:
                    values[key] = convert(field.text().strip())
                except ValueError:
                    kind = "an integer" if key == "seed" else "a number"
                    raise ValueError(f"{field.accessibleName()} must be {kind}.") from None
            values["source_id"] = self.parameter_inputs["source_id"].text().strip()
            config = replace(self.config, **values).validate()
        except ValueError as exc:
            self._parameters_message(str(exc), error=True)
            return
        if config == self.config:
            self._parameters_message("Parameters are already applied.")
            return
        if self._stream_active:
            self._pending_config = config
            self._enable_parameters(False)
            self.stop_button.setEnabled(False)
            self._parameters_message("Applying parameters…")
            self._set_status("Applying", "Applying", "Restarting stream with new parameters…", "#67717f")
            self.publisher.request_stop()
        else:
            self._commit_parameters(config)

    def _commit_parameters(self, config):
        self.ball.set_tau(config.return_tau)
        self.config = config
        self.parameter_inputs["source_id"].setText(config.source_id)
        self.parameter_inputs["source_id"].setCursorPosition(0)
        self._enable_parameters(True)
        self._parameters_message("Parameters applied for this session.")

    def start_stream(self):
        if self._closing or self._stream_active:
            return
        if self.publisher is not None:
            self.publisher.deleteLater()
        self._error = False
        self.publisher = self.publisher_factory(self.config, self.ball)
        self.publisher.ready.connect(self._ready)
        self.publisher.progress.connect(self._progress)
        self.publisher.failed.connect(self._failed)
        self.publisher.finished.connect(self._finished)
        self._stream_active = True
        self._set_status("Starting", "Starting", "Starting LSL outlet…", "#67717f")
        self.stop_button.setText("Stop Stream")
        self.stop_button.setEnabled(True)
        self._progress(0, 0.0, 0)
        self.publisher.start()

    def _set_status(self, header, sending, footer, color):
        self._status_name = header
        self.header_status.setText("●  " + header if self.sidebar_visible else "●")
        self.header_status.setToolTip(header + ": " + footer)
        self.sending_status.setText("●  " + sending)
        self.header_status.setStyleSheet(f"color: {color}")
        self.sending_status.setStyleSheet(f"color: {color}")
        self.footer_status.setText(footer)

    def _ready(self):
        if self.publisher is not None and not self.publisher.stop_event.is_set():
            self._set_status("Streaming", "Sending", "LSL outlet active", "#359b32")

    def _progress(self, count, rate, late):
        self.count_label.setText(f"{count:,}")
        self.rate_label.setText(f"{rate:.1f} packets/s")
        self.late_label.setText(f"{late:,}")

    def _failed(self, error):
        self._error = True
        self._set_status("Error", "Error", "LSL error: " + error, "#cf2639")
        print(f"Mock Mouse EMG: {error}", file=sys.stderr)

    def _finished(self):
        self._stream_active = False
        pending, self._pending_config = self._pending_config, None
        if pending is not None and not self._closing:
            # The old outlet has closed before a replacement can publish.
            self._commit_parameters(pending)
            self.start_stream()
            return
        if not self._error:
            self._set_status("Stopped", "Stopped", "LSL outlet stopped", "#67717f")
        self.rate_label.setText("0.0 packets/s")
        self.stop_button.setText("Start Stream")
        self.stop_button.setEnabled(not self._closing)
        self._enable_parameters(not self._closing)

    def toggle_stream(self):
        if self._closing or not self.stop_button.isEnabled():
            return
        if self._stream_active:
            self.publisher.request_stop()
            self.stop_button.setEnabled(False)
            self._enable_parameters(False)
            self._set_status("Stopping", "Stopping", "Stopping LSL outlet…", "#67717f")
        else:
            self.start_stream()

    def changeEvent(self, event):
        if event.type() == QEvent.Type.ActivationChange and not self.isActiveWindow() and hasattr(self, "canvas"):
            self.canvas.end_drag()
        super().changeEvent(event)

    def closeEvent(self, event):
        self._closing = True
        self._pending_config = None
        self._enable_parameters(False)
        self.canvas.end_drag()
        if self.publisher is not None and self.publisher.isRunning():
            self.publisher.request_stop()
            if not self.publisher.wait(1000):
                event.ignore()
                QTimer.singleShot(100, self.close)
                return
        self.timer.stop()
        event.accept()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Mouse spring-ball source: signed EMG [C,40] packets at 50 Hz.")
    parser.add_argument("--mapping-norm", type=float, default=100.0, help="Frobenius norm of the nonnegative 32-by-4 directional amplitude map (default: 100; 0 keeps rest carriers)")
    parser.add_argument("--noise-level", type=float, default=5.0, help="Independent per-channel/per-sample Gaussian noise standard deviation (default: 5)")
    parser.add_argument("--return-tau", type=float, default=1.0, help="Exponential return time constant in seconds (default: 1)")
    parser.add_argument("--channels", type=int, default=64, help="Fixed SAGA Bipolar-32 source: must be 64")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-id", default="mouse_mock_emg_raw")
    args = parser.parse_args(argv)
    try:
        return MouseSourceConfig(**vars(args)).validate()
    except ValueError as exc:
        parser.error(str(exc))


def main(argv=None):
    config = parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Mock Mouse EMG")
    window = MockMouseEmgWindow(config)
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
