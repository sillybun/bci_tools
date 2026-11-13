from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import platform
import queue
import signal
import socket
import subprocess
import threading
import time
import tkinter as tk
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from tkinter import font as tkfont
from typing import Any, Literal

from .branding import stacked_brand
from .version import VERSION_STRING

logger = logging.getLogger(__name__)

Language = Literal["zh", "en"]

UI_TEXT: dict[Language, dict[str, str]] = {
    "zh": {
        "window_title": "LSL 信号观测器",
        "title": "LSL 信号观测器",
        "subtitle": "发现局域网内可用的 Lab Streaming Layer 信号，并查看实时状态与发布进程",
        "discovering": "正在发现",
        "refresh": "手动刷新",
        "refresh_tooltip": "手动刷新信号流",
        "language": "EN",
        "streams_title": "可用信号流",
        "details_title": "信号详情",
        "publisher_refresh": "刷新进程识别",
        "terminate": "终止发布进程",
        "sample_title": "实时样本预览",
        "confirm_terminate_title": "确认终止发布进程",
        "confirm_terminate_body": "PID {pid}\n{command}\n\n确认发送终止信号？",
        "terminate_failed": "终止失败",
        "refreshing": "正在刷新",
        "offline": "离线 · {error}",
        "stream_count": "{count:02d} 个信号流 · 在线",
        "state_live": "● 实时",
        "state_lost": "● 中断",
        "state_wait": "● 等待",
        "state_data": "● 数据",
        "state_live_detail": "周期信号 · 正在传输",
        "state_lost_detail": "周期信号 · 已断开",
        "state_wait_detail": "周期信号 · 等待首个样本",
        "state_data_detail": "非周期 / 不规律",
        "irregular": "不规律",
        "no_timestamp": "未收到时间戳",
        "invalid_timestamp": "时间戳无效",
        "waiting_samples": "[等待样本]",
        "select_stream": "请选择一个信号流以查看元数据。",
        "no_stream": "[未选择信号流]",
        "origin_registry": "本机 registry",
        "origin_process": "本机进程候选",
        "origin_remote": "来自其他电脑节点",
        "origin_unmatched": "未识别到本机发布进程",
        "heading_state": "状态",
        "heading_name": "信号流",
        "heading_type": "类别",
        "heading_channels": "通道",
        "heading_rate": "名义 / 观测频率",
        "heading_format": "格式",
        "heading_source": "发布者 / 来源",
        "detail_state": "状态",
        "detail_name": "名称",
        "detail_class": "类别",
        "detail_channels": "通道",
        "detail_format": "格式",
        "detail_nominal": "名义频率",
        "detail_observed": "观测频率",
        "detail_source": "来源",
        "detail_host": "主机",
        "detail_uid": "UID",
        "detail_publisher": "发布进程",
        "detail_metadata": "元数据",
        "preview_timestamp": "时间戳",
        "preview_raw_timestamp": "原始时间戳",
        "preview_samples": "样本数",
    },
    "en": {
        "window_title": "LSL Signal Viewer",
        "title": "LSL Signal Viewer",
        "subtitle": "Discover Lab Streaming Layer signals on the local network and inspect their live status and publishers",
        "discovering": "DISCOVERING",
        "refresh": "Refresh",
        "refresh_tooltip": "Refresh streams",
        "language": "ZH",
        "streams_title": "Available streams",
        "details_title": "Stream details",
        "publisher_refresh": "Refresh publishers",
        "terminate": "Terminate publisher",
        "sample_title": "Live sample preview",
        "confirm_terminate_title": "Confirm publisher termination",
        "confirm_terminate_body": "PID {pid}\n{command}\n\nSend a termination signal?",
        "terminate_failed": "Termination failed",
        "refreshing": "REFRESHING",
        "offline": "OFFLINE · {error}",
        "stream_count": "{count:02d} STREAMS · LIVE",
        "state_live": "● LIVE",
        "state_lost": "● LOST",
        "state_wait": "● WAIT",
        "state_data": "● DATA",
        "state_live_detail": "Periodic signal · live",
        "state_lost_detail": "Periodic signal · disconnected",
        "state_wait_detail": "Periodic signal · waiting for first sample",
        "state_data_detail": "Aperiodic / irregular",
        "irregular": "irregular",
        "no_timestamp": "No timestamp received",
        "invalid_timestamp": "Invalid timestamp",
        "waiting_samples": "[waiting for samples]",
        "select_stream": "Select a stream to inspect its metadata.",
        "no_stream": "[no stream selected]",
        "origin_registry": "Local registry",
        "origin_process": "Local process candidate",
        "origin_remote": "Remote computer",
        "origin_unmatched": "No local publisher identified",
        "heading_state": "STATE",
        "heading_name": "STREAM",
        "heading_type": "CLASS",
        "heading_channels": "CH",
        "heading_rate": "NOMINAL / OBSERVED",
        "heading_format": "FORMAT",
        "heading_source": "BROADCASTER / SOURCE",
        "detail_state": "state",
        "detail_name": "name",
        "detail_class": "class",
        "detail_channels": "channels",
        "detail_format": "format",
        "detail_nominal": "nominal",
        "detail_observed": "observed",
        "detail_source": "source",
        "detail_host": "host",
        "detail_uid": "uid",
        "detail_publisher": "publisher",
        "detail_metadata": "metadata",
        "preview_timestamp": "timestamp",
        "preview_raw_timestamp": "raw timestamp",
        "preview_samples": "samples",
    },
}


def ui_text(language: Language, key: str, **values: Any) -> str:
    return UI_TEXT[language][key].format(**values)


def publisher_origin_label(origin: str, language: Language) -> str:
    return ui_text(language, f"origin_{origin}")


def rounded_rectangle_points(
    x1: float, y1: float, x2: float, y2: float, radius: float
) -> list[float]:
    """Return smooth-polygon control points for a rounded rectangle."""
    radius = max(0.0, min(float(radius), abs(x2 - x1) / 2, abs(y2 - y1) / 2))
    return [
        x1 + radius,
        y1,
        x2 - radius,
        y1,
        x2,
        y1,
        x2,
        y1 + radius,
        x2,
        y2 - radius,
        x2,
        y2,
        x2 - radius,
        y2,
        x1 + radius,
        y2,
        x1,
        y2,
        x1,
        y2 - radius,
        x1,
        y1 + radius,
        x1,
        y1,
    ]


def scrollbar_thumb_bounds(
    length: float,
    first: float,
    last: float,
    *,
    inset: float = 2.0,
    min_thumb: float = 24.0,
) -> tuple[float, float]:
    """Map scrollbar fractions to a bounded, usable thumb interval."""
    length = max(0.0, float(length))
    inset = max(0.0, min(float(inset), length / 2))
    track_length = max(0.0, length - 2 * inset)
    first = max(0.0, min(1.0, float(first)))
    last = max(first, min(1.0, float(last)))
    span = last - first
    thumb_length = min(track_length, max(min_thumb, track_length * span))
    travel = max(0.0, track_length - thumb_length)
    scrollable_span = max(0.0, 1.0 - span)
    progress = first / scrollable_span if scrollable_span else 0.0
    start = inset + travel * max(0.0, min(1.0, progress))
    return start, start + thumb_length


class RoundedButton(tk.Canvas):
    """Borderless rounded button with hover, disabled, keyboard, and tooltip states."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        text: str = "",
        command: Callable[[], None] | None = None,
        width: int | None = None,
        min_width: int = 40,
        height: int = 34,
        radius: int = 8,
        background: str = "#f4f7fb",
        fill: str = "#e8edf3",
        hover_fill: str = "#dce4ed",
        pressed_fill: str = "#cfd9e5",
        disabled_fill: str = "#f1f5f9",
        foreground: str = "#1f2937",
        disabled_foreground: str = "#94a3b8",
        font: tuple[str, int, str] | tuple[str, int] = ("Helvetica", 11),
        tooltip: str = "",
    ):
        self._font = tkfont.Font(master=master, font=font)
        self._fixed_width = width
        self._min_width = min_width
        self._height = height
        self._radius = radius
        self._fills = {
            "normal": fill,
            "hover": hover_fill,
            "pressed": pressed_fill,
            "disabled": disabled_fill,
        }
        self._foreground = foreground
        self._disabled_foreground = disabled_foreground
        self._text = text
        self._state = "normal"
        self._visual_state = "normal"
        self._command = command
        self._tooltip_text = tooltip
        self._tooltip_after: str | None = None
        self._tooltip_window: tk.Toplevel | None = None
        initial_width = self._button_width()
        super().__init__(
            master,
            width=initial_width,
            height=height,
            background=background,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            cursor="hand2",
            takefocus=1,
        )
        self._shape = self.create_polygon(
            rounded_rectangle_points(0, 0, initial_width, height, radius),
            smooth=True,
            splinesteps=24,
            outline="",
        )
        self._label = self.create_text(
            initial_width / 2,
            height / 2,
            text=text,
            fill=foreground,
            font=self._font,
        )
        self.bind("<Configure>", self._redraw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<KeyPress-space>", self._on_key_press)
        self.bind("<KeyRelease-space>", self._on_key_release)
        self.bind("<KeyPress-Return>", self._on_key_press)
        self.bind("<KeyRelease-Return>", self._on_key_release)
        self.bind("<FocusIn>", lambda _event: self._set_visual("hover"))
        self.bind("<FocusOut>", lambda _event: self._set_visual("normal"))
        self._render()

    def _button_width(self) -> int:
        if self._fixed_width is not None:
            return self._fixed_width
        return max(self._min_width, self._font.measure(self._text) + 24)

    def _redraw(self, _event: tk.Event[Any] | None = None) -> None:
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        self.coords(
            self._shape,
            *rounded_rectangle_points(0, 0, width, height, min(self._radius, height / 2)),
        )
        self.coords(self._label, width / 2, height / 2)

    def _render(self) -> None:
        visual = "disabled" if self._state == "disabled" else self._visual_state
        self.itemconfigure(self._shape, fill=self._fills[visual])
        foreground = self._disabled_foreground if self._state == "disabled" else self._foreground
        self.itemconfigure(self._label, fill=foreground, text=self._text)
        self.configure(cursor="" if self._state == "disabled" else "hand2")

    def _set_visual(self, visual: str) -> None:
        if self._state != "disabled":
            self._visual_state = visual
            self._render()

    def _on_enter(self, _event: tk.Event[Any]) -> None:
        self._set_visual("hover")
        if self._tooltip_text:
            self._tooltip_after = self.after(450, self._show_tooltip)

    def _on_leave(self, _event: tk.Event[Any]) -> None:
        self._set_visual("normal")
        self._hide_tooltip()

    def _on_press(self, _event: tk.Event[Any]) -> None:
        if self._state != "disabled":
            self.focus_set()
            self._set_visual("pressed")

    def _on_release(self, event: tk.Event[Any]) -> None:
        if self._state == "disabled":
            return
        inside = 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height()
        self._set_visual("hover" if inside else "normal")
        if inside and self._command is not None:
            self._command()

    def _on_key_press(self, _event: tk.Event[Any]) -> str:
        self._set_visual("pressed")
        return "break"

    def _on_key_release(self, _event: tk.Event[Any]) -> str:
        if self._state != "disabled":
            self._set_visual("hover")
            if self._command is not None:
                self._command()
        return "break"

    def _show_tooltip(self) -> None:
        self._tooltip_after = None
        if not self._tooltip_text or self._state == "disabled":
            return
        popup = tk.Toplevel(self)
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)
        popup.configure(background="#0f172a")
        tk.Label(
            popup,
            text=self._tooltip_text,
            background="#0f172a",
            foreground="#ffffff",
            padx=8,
            pady=5,
            font=("Helvetica", 10),
        ).pack()
        popup.update_idletasks()
        x = self.winfo_rootx() + (self.winfo_width() - popup.winfo_reqwidth()) // 2
        y = self.winfo_rooty() + self.winfo_height() + 6
        popup.geometry(f"+{max(0, x)}+{max(0, y)}")
        self._tooltip_window = popup

    def _hide_tooltip(self) -> None:
        if self._tooltip_after is not None:
            self.after_cancel(self._tooltip_after)
            self._tooltip_after = None
        if self._tooltip_window is not None:
            self._tooltip_window.destroy()
            self._tooltip_window = None

    def set_text(self, text: str) -> None:
        self._text = text
        width = self._button_width()
        self.configure(width=width)
        self._render()
        self._redraw()

    def set_tooltip(self, text: str) -> None:
        self._tooltip_text = text
        self._hide_tooltip()

    def set_state(self, state: str) -> None:
        if state not in {"normal", "disabled"}:
            raise ValueError(f"unsupported button state: {state}")
        self._state = state
        self._visual_state = "normal"
        self._render()

    def set_command(self, command: Callable[[], None] | None) -> None:
        self._command = command


class RoundedPanel(tk.Canvas):
    """A borderless rounded background with a normal Frame content surface."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        fill: str = "#ffffff",
        background: str = "#f4f7fb",
        radius: int = 8,
        padding: int = 16,
    ):
        super().__init__(
            master,
            width=320,
            height=240,
            background=background,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
        )
        self._fill = fill
        self._radius = radius
        self._padding = padding
        self._shape = self.create_polygon(
            rounded_rectangle_points(0, 0, 320, 240, radius),
            smooth=True,
            splinesteps=24,
            fill=fill,
            outline="",
        )
        self.content = tk.Frame(self, background=fill, borderwidth=0, highlightthickness=0)
        self._window = self.create_window(
            padding, padding, anchor="nw", window=self.content, width=288, height=208
        )
        self.bind("<Configure>", self._resize)

    def _resize(self, _event: tk.Event[Any] | None = None) -> None:
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        self.coords(
            self._shape,
            *rounded_rectangle_points(0, 0, width, height, self._radius),
        )
        self.coords(self._window, self._padding, self._padding)
        self.itemconfigure(
            self._window,
            width=max(1, width - 2 * self._padding),
            height=max(1, height - 2 * self._padding),
        )


class RoundedScrollbar(tk.Canvas):
    """Borderless light scrollbar with a rounded track and rounded thumb."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        orient: str,
        command: Callable[..., Any],
        background: str = "#ffffff",
        thickness: int = 10,
    ):
        if orient not in {"vertical", "horizontal"}:
            raise ValueError(f"unsupported scrollbar orientation: {orient}")
        dimensions = {"width": thickness, "height": 80}
        if orient == "horizontal":
            dimensions = {"width": 80, "height": thickness}
        super().__init__(
            master,
            **dimensions,
            background=background,
            borderwidth=0,
            highlightthickness=0,
            relief="flat",
            cursor="hand2",
        )
        self.orient = orient
        self.command = command
        self.first = 0.0
        self.last = 1.0
        self._drag_offset: float | None = None
        self._hover = False
        initial_points = rounded_rectangle_points(0, 0, thickness, thickness, thickness / 2)
        self._track = self.create_polygon(initial_points, smooth=True, splinesteps=24, outline="")
        self._thumb = self.create_polygon(initial_points, smooth=True, splinesteps=24, outline="")
        self.bind("<Configure>", self._draw)
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", lambda _event: setattr(self, "_drag_offset", None))
        self._draw()

    def set(self, first: str | float, last: str | float) -> None:
        self.first = max(0.0, min(1.0, float(first)))
        self.last = max(self.first, min(1.0, float(last)))
        self._draw()

    def _length(self) -> float:
        return float(self.winfo_height() if self.orient == "vertical" else self.winfo_width())

    def _position(self, event: tk.Event[Any]) -> float:
        return float(event.y if self.orient == "vertical" else event.x)

    def _draw(self, _event: tk.Event[Any] | None = None) -> None:
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        inset = 2.0
        if self.orient == "vertical":
            self.coords(
                self._track,
                *rounded_rectangle_points(inset, 0, width - inset, height, width / 2),
            )
            start, end = scrollbar_thumb_bounds(height, self.first, self.last, inset=inset)
            self.coords(
                self._thumb,
                *rounded_rectangle_points(inset, start, width - inset, end, width / 2),
            )
        else:
            self.coords(
                self._track,
                *rounded_rectangle_points(0, inset, width, height - inset, height / 2),
            )
            start, end = scrollbar_thumb_bounds(width, self.first, self.last, inset=inset)
            self.coords(
                self._thumb,
                *rounded_rectangle_points(start, inset, end, height - inset, height / 2),
            )
        self.itemconfigure(self._track, fill="#f1f5f9")
        self.itemconfigure(self._thumb, fill="#aebdcd" if self._hover else "#cbd5e1")

    def _on_enter(self, _event: tk.Event[Any]) -> None:
        self._hover = True
        self._draw()

    def _on_leave(self, _event: tk.Event[Any]) -> None:
        self._hover = False
        self._drag_offset = None
        self._draw()

    def _on_press(self, event: tk.Event[Any]) -> None:
        position = self._position(event)
        start, end = scrollbar_thumb_bounds(self._length(), self.first, self.last)
        if start <= position <= end:
            self._drag_offset = position - start
            return
        self.command("scroll", -1 if position < start else 1, "pages")

    def _on_drag(self, event: tk.Event[Any]) -> None:
        if self._drag_offset is None:
            return
        length = self._length()
        start, end = scrollbar_thumb_bounds(length, self.first, self.last)
        thumb_length = end - start
        travel = max(0.0, length - 4.0 - thumb_length)
        scrollable_span = max(0.0, 1.0 - (self.last - self.first))
        if travel <= 0 or scrollable_span <= 0:
            return
        target = max(
            2.0, min(length - 2.0 - thumb_length, self._position(event) - self._drag_offset)
        )
        fraction = ((target - 2.0) / travel) * scrollable_span
        self.command("moveto", fraction)


@dataclass
class LslStreamRecord:
    name: str
    stream_type: str
    channel_count: int
    nominal_srate: float
    channel_format: str
    source_id: str
    hostname: str
    uid: str
    created_at: float
    description: dict[str, Any] = field(default_factory=dict)
    latest_sample: list[Any] = field(default_factory=list)
    latest_timestamp: float | None = None
    sample_count: int = 0
    sample_rate_hz: float | None = None
    last_sample_monotonic: float | None = None
    publisher_pid: int | None = None
    publisher_command: str = ""
    publisher_candidates: list[tuple[int, str]] = field(default_factory=list)
    publisher_origin: str = "remote"
    last_seen_monotonic: float | None = None
    sample_history: list[tuple[float, list[Any]]] = field(default_factory=list)

    def record_sample(self, sample: Any, timestamp: float, history_seconds: float) -> None:
        """Keep received samples ordered by LSL time, including delayed arrivals."""
        timestamp = float(timestamp)
        payload = list(sample)
        self.sample_count += 1
        self.last_sample_monotonic = time.monotonic()
        self.sample_history = [
            (existing_timestamp, existing_sample)
            for existing_timestamp, existing_sample in self.sample_history
            if existing_timestamp != timestamp
        ]
        self.sample_history.append((timestamp, payload))
        self.sample_history.sort(key=lambda item: item[0])
        newest_timestamp = self.sample_history[-1][0]
        cutoff = newest_timestamp - max(0.0, history_seconds)
        self.sample_history = [item for item in self.sample_history if item[0] >= cutoff]
        latest_timestamp, latest_sample = self.sample_history[-1]
        self.latest_timestamp = latest_timestamp
        self.latest_sample = list(latest_sample)

    def signal_state(self, now: float | None = None) -> str:
        now = time.monotonic() if now is None else now
        if self.nominal_srate <= 0:
            return "irregular"
        if self.last_sample_monotonic is None:
            return "periodic-waiting"
        if self.last_sample_monotonic is not None and now - self.last_sample_monotonic <= max(
            1.5, 3.0 / self.nominal_srate
        ):
            return "periodic-live"
        if self.sample_count:
            return "periodic-stale"
        return "periodic-live"


@dataclass(frozen=True)
class CatalogSnapshot:
    records: dict[str, LslStreamRecord]
    error: str | None = None


def _safe(value: Any, default: Any = "") -> Any:
    try:
        return value() if callable(value) else value
    except Exception:  # noqa: BLE001 - malformed metadata should remain visible
        return default


def _metadata(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    result: dict[str, Any] = {}
    name = _safe(getattr(node, "name", None))
    value = _safe(getattr(node, "value", None))
    if name:
        result["name"] = name
    if value:
        result["value"] = value
    child = _safe(getattr(node, "child", None), None)
    children = []
    while child is not None:
        children.append(_metadata(child))
        child = _safe(getattr(child, "next_sibling", None), None)
    if children:
        result["children"] = children
    return result


def stream_record(info: Any) -> LslStreamRecord:
    return LslStreamRecord(
        name=str(_safe(getattr(info, "name", None), "Unnamed")),
        stream_type=str(_safe(getattr(info, "type", None))),
        channel_count=int(_safe(getattr(info, "channel_count", None), 0) or 0),
        nominal_srate=float(_safe(getattr(info, "nominal_srate", None), 0.0) or 0.0),
        channel_format=str(_safe(getattr(info, "channel_format", None))),
        source_id=str(_safe(getattr(info, "source_id", None))),
        hostname=str(_safe(getattr(info, "hostname", None))),
        uid=str(_safe(getattr(info, "uid", None))),
        created_at=float(_safe(getattr(info, "created_at", None), 0.0) or 0.0),
        description=_metadata(_safe(getattr(info, "desc", None), None)),
    )


CHANNEL_FORMATS = {
    "0": "undefined",
    "1": "int8",
    "2": "int16",
    "3": "int32",
    "4": "int64",
    "5": "float32",
    "6": "double64",
    "7": "string",
    "8": "string (serialized)",
}


def format_label(value: Any) -> str:
    """Translate pylsl enum values or names to a human-readable format."""
    text = str(value or "").strip()
    return CHANNEL_FORMATS.get(text.lower(), text or "unknown")


def human_timestamp(timestamp: float | None, language: Language = "zh") -> str:
    if timestamp is None:
        return ui_text(language, "no_timestamp")
    try:
        local_time = datetime.fromtimestamp(float(timestamp)).astimezone()
        return (
            local_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            + f" {local_time.tzname() or ''}".rstrip()
        )
    except (TypeError, ValueError, OverflowError, OSError):
        return ui_text(language, "invalid_timestamp")


def format_sample_preview(sample: list[Any], language: Language = "en") -> str:
    """Render an LSL sample as readable JSON, decoding JSON string channels."""
    if not sample:
        return ui_text(language, "waiting_samples")

    def decode_json_string(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    decoded = [decode_json_string(value) for value in sample]
    value = decoded[0] if len(decoded) == 1 else decoded
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return repr(sample)


def local_processes() -> list[tuple[int, str]]:
    """Return local PID and command-line pairs without requiring psutil."""
    try:
        if platform.system() == "Windows":
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            raw = json.loads(result.stdout or "[]")
            rows = raw if isinstance(raw, list) else [raw]
            return [
                (int(row["ProcessId"]), str(row.get("CommandLine") or ""))
                for row in rows
                if row.get("ProcessId")
            ]
        result = subprocess.run(
            ["ps", "-axo", "pid=,command="], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.SubprocessError):
        return []
    processes = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) == 2 and fields[0].isdigit():
            processes.append((int(fields[0]), fields[1]))
    return processes


def active_registry_path() -> str:
    if platform.system() == "Windows":
        base = os.environ.get("APPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Roaming"
        )
    elif platform.system() == "Darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "BCIControl", "active_lsl_processes.json")


def read_active_registry(path: str | None = None) -> list[dict[str, Any]]:
    try:
        with open(path or active_registry_path(), encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return []
    return (
        raw if isinstance(raw, list) else raw.get("processes", []) if isinstance(raw, dict) else []
    )


def match_registry_process(
    record: LslStreamRecord, entries: list[dict[str, Any]]
) -> list[tuple[int, str]]:
    """Match a stream against integrated-term's static stream registry."""
    scored: list[tuple[int, float, int, str]] = []
    for entry in entries:
        try:
            pid = int(entry["pid"])
        except (KeyError, TypeError, ValueError):
            continue
        score = 0
        for stream in entry.get("streams", []):
            if not isinstance(stream, dict):
                continue
            candidate_score = 0
            if stream.get("type") == record.stream_type:
                candidate_score += 4
            if stream.get("name") == record.name:
                candidate_score += 3
            if stream.get("source_id") and stream.get("source_id") == record.source_id:
                candidate_score += 10
            score = max(score, candidate_score)
        if score:
            scored.append(
                (
                    score,
                    float(entry.get("started_at", 0.0)),
                    pid,
                    str(entry.get("name") or entry.get("command") or ""),
                )
            )
    scored.sort(reverse=True)
    return [(pid, label) for _score, _started_at, pid, label in scored]


def match_publisher_process(
    record: LslStreamRecord, processes: list[tuple[int, str]] | None = None
) -> list[tuple[int, str]]:
    """Match likely local publishers using explicit LSL identity tokens."""
    tokens = [
        token.lower() for token in (record.name, record.stream_type, record.source_id) if token
    ]
    if not tokens:
        return []
    matches = []
    for pid, command in processes if processes is not None else local_processes():
        if pid == os.getpid() or "lsl-viewer" in command.lower():
            continue
        command_lower = command.lower()
        if any(token in command_lower for token in tokens):
            matches.append((pid, command))
    return matches


def terminate_publisher(pid: int) -> None:
    """Gracefully terminate one resolved local publisher PID."""
    if pid <= 1 or pid == os.getpid():
        raise ValueError("refusing to terminate an unsafe PID")
    os.kill(pid, signal.SIGTERM)


class LslStreamCatalog:
    def __init__(
        self,
        resolve: Callable[[], list[Any]] | None = None,
        inlet_factory: Any = None,
        resolve_wait_time: float = 0.1,
        missing_grace_s: float = 2.0,
        history_seconds: float = 5.0,
        max_samples_per_stream: int = 512,
    ):
        self.resolve = resolve
        self.inlet_factory = inlet_factory
        self.resolve_wait_time = max(0.0, float(resolve_wait_time))
        self.missing_grace_s = max(0.0, float(missing_grace_s))
        self.history_seconds = max(0.0, float(history_seconds))
        self.max_samples_per_stream = max(1, int(max_samples_per_stream))
        self.records: dict[str, LslStreamRecord] = {}
        self.inlets: dict[str, Any] = {}
        self.last_tick = time.monotonic()
        self.last_counts: dict[str, int] = {}
        self.missing_since: dict[str, float] = {}
        self.registry_reader = read_active_registry

    def refresh_publishers(
        self,
        record: LslStreamRecord,
        processes: list[tuple[int, str]] | None = None,
    ) -> list[tuple[int, str]]:
        processes = local_processes() if processes is None else processes
        live_pids = {pid for pid, _command in processes}
        registry_entries = [
            entry for entry in self.registry_reader() if entry.get("pid") in live_pids
        ]
        local_names = {platform.node().lower(), socket.gethostname().lower(), "localhost"}
        remote_host = bool(record.hostname and record.hostname.lower() not in local_names)
        process_matches = [] if remote_host else match_publisher_process(record, processes)
        registry_matches = [] if remote_host else match_registry_process(record, registry_entries)
        if registry_matches:
            record.publisher_candidates = registry_matches
            record.publisher_origin = "registry"
        elif process_matches:
            record.publisher_candidates = process_matches
            record.publisher_origin = "process"
        else:
            record.publisher_candidates = []
            record.publisher_origin = "remote" if remote_host else "unmatched"
        if record.publisher_candidates:
            record.publisher_pid, record.publisher_command = record.publisher_candidates[0]
        else:
            record.publisher_pid, record.publisher_command = None, ""
        return record.publisher_candidates

    def refresh(self) -> list[LslStreamRecord]:
        if self.resolve is None:
            from pylsl import resolve_streams

            infos = resolve_streams(wait_time=self.resolve_wait_time)
        else:
            infos = self.resolve()
        infos = list(infos or [])
        seen_keys: set[str] = set()
        processes = local_processes()
        for info in infos:
            record = stream_record(info)
            key = record.uid or f"{record.name}|{record.source_id}|{record.hostname}"
            seen_keys.add(key)
            self.records.setdefault(key, record)
            current = self.records[key]
            # Discovery metadata can change while an outlet keeps the same UID.
            current.name = record.name
            current.stream_type = record.stream_type
            current.channel_count = record.channel_count
            current.nominal_srate = record.nominal_srate
            current.channel_format = record.channel_format
            current.source_id = record.source_id
            current.hostname = record.hostname
            current.created_at = record.created_at
            current.description = record.description
            current.last_seen_monotonic = time.monotonic()
            self.refresh_publishers(current, processes)
            if key not in self.inlets:
                try:
                    factory = self.inlet_factory
                    if factory is None:
                        from pylsl import StreamInlet

                        factory = StreamInlet
                    self.inlets[key] = factory(info, max_buflen=2, max_chunklen=1)
                except Exception:  # noqa: BLE001, S112 - isolate unavailable streams
                    continue
            try:
                for _ in range(self.max_samples_per_stream):
                    sample, timestamp = self.inlets[key].pull_sample(timeout=0.0)
                    if sample is None:
                        break
                    current = self.records[key]
                    current.record_sample(sample, timestamp, self.history_seconds)
            except Exception:  # noqa: BLE001, S112 - isolate malformed stream samples
                continue
        # liblsl discovery can briefly return incomplete snapshots during
        # network/startup convergence, so retain short-lived misses.
        now = time.monotonic()
        for key in seen_keys:
            self.missing_since.pop(key, None)
        for key in set(self.records) - seen_keys:
            self.missing_since.setdefault(key, now)
            if now - self.missing_since[key] < self.missing_grace_s:
                continue
            inlet = self.inlets.pop(key, None)
            close_stream = getattr(inlet, "close_stream", None)
            if callable(close_stream):
                try:
                    close_stream()
                except Exception:
                    logger.debug("failed to close removed LSL inlet %s", key, exc_info=True)
            self.records.pop(key, None)
            self.last_counts.pop(key, None)
            self.missing_since.pop(key, None)
        for key, record in self.records.items():
            if len(record.sample_history) >= 2:
                first_timestamp = record.sample_history[0][0]
                last_timestamp = record.sample_history[-1][0]
                timestamp_span = last_timestamp - first_timestamp
                record.sample_rate_hz = (
                    (len(record.sample_history) - 1) / timestamp_span
                    if timestamp_span > 0
                    else None
                )
            else:
                record.sample_rate_hz = None
        self.last_counts = {key: record.sample_count for key, record in self.records.items()}
        self.last_tick = now
        return list(self.records.values())

    def close(self) -> None:
        for key, inlet in list(self.inlets.items()):
            close_stream = getattr(inlet, "close_stream", None)
            if callable(close_stream):
                try:
                    close_stream()
                except Exception:
                    logger.debug("failed to close LSL inlet %s", key, exc_info=True)
        self.inlets.clear()


class AsyncLslStreamCatalog:
    """Own pylsl discovery and inlets off the Tk event thread."""

    def __init__(self, catalog: LslStreamCatalog, refresh_interval_s: float = 0.2):
        self.catalog = catalog
        self.refresh_interval_s = max(0.01, float(refresh_interval_s))
        self._stop = threading.Event()
        self._refresh_requested = threading.Event()
        self._snapshots: queue.Queue[CatalogSnapshot] = queue.Queue(maxsize=1)
        self._worker = threading.Thread(
            target=self._run,
            name="lsl-viewer-catalog",
            daemon=True,
        )

    def start(self) -> None:
        self._worker.start()

    def request_refresh(self) -> None:
        self._refresh_requested.set()

    def latest(self) -> CatalogSnapshot | None:
        snapshot = None
        while True:
            try:
                snapshot = self._snapshots.get_nowait()
            except queue.Empty:
                return snapshot

    def close(self, timeout: float = 0.5) -> None:
        self._stop.set()
        self._refresh_requested.set()
        self._worker.join(timeout=max(0.0, timeout))

    def _publish(self, snapshot: CatalogSnapshot) -> None:
        try:
            self._snapshots.put_nowait(snapshot)
        except queue.Full:
            try:
                self._snapshots.get_nowait()
            except queue.Empty:
                pass
            self._snapshots.put_nowait(snapshot)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                started_at = time.monotonic()
                try:
                    self.catalog.refresh()
                    snapshot = CatalogSnapshot(copy.deepcopy(self.catalog.records))
                except Exception as exc:  # noqa: BLE001 - surface worker failures in UI
                    snapshot = CatalogSnapshot(
                        copy.deepcopy(self.catalog.records),
                        error=type(exc).__name__,
                    )
                self._publish(snapshot)
                remaining = self.refresh_interval_s - (time.monotonic() - started_at)
                self._refresh_requested.wait(timeout=max(0.0, remaining))
                self._refresh_requested.clear()
        finally:
            self.catalog.close()


class TkLslViewer:
    def __init__(self, title: str = "LSL Signal Observatory", refresh_hz: float = 5.0, language: Language = "zh"):
        self.title = title
        self.refresh_ms = max(100, round(1000 / refresh_hz))
        self.language = language

    def run(self) -> None:
        from tkinter import ttk

        root = tk.Tk()
        root.geometry("1360x820")
        root.minsize(980, 620)
        root.configure(bg="#f4f7fb")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background="#ffffff",
            foreground="#1f2937",
            fieldbackground="#ffffff",
            rowheight=30,
        )
        style.configure(
            "Treeview.Heading", background="#e7edf5", foreground="#334155", relief="flat"
        )
        style.configure("TFrame", background="#f4f7fb")
        style.configure("TLabel", background="#f4f7fb", foreground="#1f2937")
        style.configure("Header.TLabel", font=("Helvetica", 20, "bold"), foreground="#111827")
        style.configure("Muted.TLabel", foreground="#64748b")
        style.configure(
            "Clean.TCombobox",
            fieldbackground="#ffffff",
            background="#ffffff",
            foreground="#1f2937",
            bordercolor="#cbd5e1",
            lightcolor="#ffffff",
            darkcolor="#ffffff",
            arrowcolor="#64748b",
            relief="flat",
            borderwidth=1,
            padding=(9, 6),
        )
        style.map(
            "Clean.TCombobox",
            fieldbackground=[("readonly", "#ffffff"), ("disabled", "#f8fafc")],
            foreground=[("disabled", "#64748b")],
            bordercolor=[("focus", "#94a3b8"), ("readonly", "#cbd5e1")],
        )
        root.option_add("*TCombobox*Listbox.background", "#ffffff")
        root.option_add("*TCombobox*Listbox.foreground", "#1f2937")
        root.option_add("*TCombobox*Listbox.selectBackground", "#e2e8f0")
        root.option_add("*TCombobox*Listbox.selectForeground", "#0f172a")
        root.option_add("*TCombobox*Listbox.font", "Helvetica 11")
        catalog = LslStreamCatalog()
        async_catalog = AsyncLslStreamCatalog(catalog, self.refresh_ms / 1000.0)
        records_by_key: dict[str, LslStreamRecord] = {}
        selected: dict[str, str | None] = {"key": None}
        language: dict[str, Language] = {"value": self.language}
        status_state: dict[str, Any] = {"kind": "discovering", "error": None}
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        header = ttk.Frame(root, padding=(24, 20, 24, 12))
        header.grid(row=0, column=0, sticky="ew")
        title_box = ttk.Frame(header)
        title_box.pack(side="left")
        stacked_brand(title_box, background="#f4f7fb").pack(side="left", padx=(0, 14))
        title_text = ttk.Frame(title_box)
        title_text.pack(side="left")
        title_label = ttk.Label(title_text, style="Header.TLabel")
        title_label.pack(anchor="w")
        version_label = ttk.Label(title_text, text=f"v{VERSION_STRING}", style="Muted.TLabel")
        version_label.pack(anchor="w", pady=(2, 0))
        subtitle_label = ttk.Label(title_text, style="Muted.TLabel")
        subtitle_label.pack(anchor="w", pady=(3, 0))
        status = ttk.Label(header, style="Muted.TLabel")
        status.pack(side="right", pady=8)
        language_button = RoundedButton(header, width=48, background="#f4f7fb")
        language_button.pack(side="right", padx=(0, 8), pady=5)
        refresh_button = RoundedButton(
            header,
            text="↻",
            width=38,
            height=34,
            radius=8,
            background="#f4f7fb",
            font=("Helvetica", 18),
        )
        refresh_button.pack(side="right", padx=(0, 14), pady=5)
        body = ttk.Frame(root, padding=(24, 0, 24, 24))
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        left_panel = RoundedPanel(body)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left = left_panel.content
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        streams_label = ttk.Label(
            left, font=("Helvetica", 13, "bold"), foreground="#172b4d", background="#ffffff"
        )
        streams_label.grid(row=0, column=0, sticky="w", pady=(0, 12))
        columns = ("state", "name", "type", "channels", "rate", "format", "source")
        tree = ttk.Treeview(left, columns=columns, show="headings", selectmode="browse")
        heading_keys = {
            "state": "heading_state",
            "name": "heading_name",
            "type": "heading_type",
            "channels": "heading_channels",
            "rate": "heading_rate",
            "format": "heading_format",
            "source": "heading_source",
        }
        widths = {
            "state": 65,
            "name": 170,
            "type": 120,
            "channels": 45,
            "rate": 125,
            "format": 85,
            "source": 210,
        }
        for col in columns:
            tree.column(col, width=widths[col], minwidth=40, anchor="w", stretch=False)
        tree.grid(row=1, column=0, sticky="nsew")
        for tag, color in (
            ("periodic-live", "#16a34a"),
            ("periodic-stale", "#ca8a04"),
            ("periodic-waiting", "#64748b"),
            ("irregular", "#2563eb"),
        ):
            tree.tag_configure(tag, foreground=color)
        scroll = RoundedScrollbar(left, orient="vertical", command=tree.yview)
        scroll.grid(row=1, column=1, sticky="ns", padx=(4, 0))
        tree.configure(yscrollcommand=scroll.set)
        hscroll = RoundedScrollbar(left, orient="horizontal", command=tree.xview)
        hscroll.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        tree.configure(xscrollcommand=hscroll.set)
        right_panel = RoundedPanel(body)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right = right_panel.content
        right.rowconfigure(4, weight=1)
        right.columnconfigure(0, weight=1)
        details_label = ttk.Label(
            right, font=("Helvetica", 13, "bold"), foreground="#172b4d", background="#ffffff"
        )
        details_label.grid(row=0, column=0, sticky="w", pady=(0, 12))
        details = tk.Text(
            right,
            height=12,
            bg="#ffffff",
            fg="#1f2937",
            insertbackground="#1f2937",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=14,
            pady=12,
            font=("Menlo", 10),
            wrap="word",
        )
        details.grid(row=1, column=0, sticky="ew")
        publisher_frame = tk.Frame(
            right,
            background="#ffffff",
            borderwidth=0,
            highlightthickness=0,
            padx=8,
            pady=6,
        )
        publisher_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        publisher_frame.columnconfigure(1, weight=1)
        publisher_refresh = RoundedButton(publisher_frame, background="#ffffff")
        publisher_refresh.grid(row=0, column=0, sticky="w")
        publisher_var = tk.StringVar(value="")
        publisher_choice = ttk.Combobox(
            publisher_frame, textvariable=publisher_var, state="disabled", style="Clean.TCombobox"
        )
        publisher_choice.grid(row=0, column=1, sticky="ew", padx=(8, 8))
        terminate = RoundedButton(publisher_frame, background="#ffffff")
        terminate.set_state("disabled")
        terminate.grid(row=0, column=2, sticky="e")
        sample_label = ttk.Label(right, style="Muted.TLabel")
        sample_label.grid(row=3, column=0, sticky="nw", pady=(16, 8))
        sample = tk.Text(
            right,
            bg="#ffffff",
            fg="#166534",
            insertbackground="#1f2937",
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            padx=14,
            pady=12,
            font=("Menlo", 10),
            wrap="none",
        )
        sample.grid(row=4, column=0, sticky="nsew")
        for widget in (details, sample):
            widget.configure(state="disabled")

        def select(_event: Any = None) -> None:
            item = tree.selection()
            selected["key"] = item[0] if item else None

        def terminate_selected() -> None:
            from tkinter import messagebox

            record = records_by_key.get(selected["key"] or "")
            if record is None:
                return
            try:
                pid = int(publisher_var.get().split(" ", 1)[0])
            except (ValueError, IndexError):
                return
            command = next(
                (cmd for candidate_pid, cmd in record.publisher_candidates if candidate_pid == pid),
                "",
            )
            current_language = language["value"]
            if not messagebox.askyesno(
                ui_text(current_language, "confirm_terminate_title"),
                ui_text(current_language, "confirm_terminate_body", pid=pid, command=command),
                parent=root,
            ):
                return
            try:
                terminate_publisher(pid)
                terminate.set_state("disabled")
            except (OSError, ValueError) as exc:
                messagebox.showerror(
                    ui_text(current_language, "terminate_failed"),
                    str(exc),
                    parent=root,
                )

        terminate.set_command(terminate_selected)

        publisher_choice.bind("<<ComboboxSelected>>", lambda _event: terminate.set_state("normal"))

        tree.bind("<<TreeviewSelect>>", select)

        def update_publisher_controls(record: LslStreamRecord | None) -> None:
            previous_pid = publisher_var.get().split(" ", 1)[0]
            candidates = [
                f"{pid} · {command}"
                for pid, command in (record.publisher_candidates if record else [])
            ]
            display_values = candidates or (
                [publisher_origin_label(record.publisher_origin, language["value"])]
                if record
                else []
            )
            publisher_choice.configure(
                values=display_values, state="readonly" if candidates else "disabled"
            )
            retained = next(
                (value for value in candidates if value.split(" ", 1)[0] == previous_pid), None
            )
            publisher_var.set(retained or (display_values[0] if display_values else ""))
            terminate.set_state("normal" if candidates and publisher_var.get() else "disabled")

        def refresh_selected_publishers() -> None:
            record = records_by_key.get(selected["key"] or "")
            if record is None:
                update_publisher_controls(None)
                return
            status_state["kind"] = "refreshing"
            status.configure(text=ui_text(language["value"], "refreshing"))
            async_catalog.request_refresh()

        publisher_refresh.set_command(refresh_selected_publishers)

        def tick(schedule: bool = True) -> None:
            snapshot = async_catalog.latest()
            if snapshot is not None:
                records_by_key.clear()
                records_by_key.update(snapshot.records)
                status_state["kind"] = "offline" if snapshot.error else "stream_count"
                status_state["error"] = snapshot.error
            current_language = language["value"]
            if status_state["kind"] == "offline":
                status.configure(
                    text=ui_text(current_language, "offline", error=status_state["error"])
                )
            elif status_state["kind"] == "stream_count":
                status.configure(
                    text=ui_text(current_language, "stream_count", count=len(records_by_key))
                )
            else:
                status.configure(text=ui_text(current_language, status_state["kind"]))
            for item in tree.get_children():
                tree.delete(item)
            label_keys = {
                "periodic-live": "state_live",
                "periodic-stale": "state_lost",
                "periodic-waiting": "state_wait",
                "irregular": "state_data",
            }
            for key, record in records_by_key.items():
                state = record.signal_state()
                nominal = (
                    f"{record.nominal_srate:g} Hz"
                    if record.nominal_srate
                    else ui_text(current_language, "irregular")
                )
                observed = (
                    "—" if record.sample_rate_hz is None else f"{record.sample_rate_hz:.1f} Hz"
                )
                tree.insert(
                    "",
                    "end",
                    iid=key,
                    values=(
                        ui_text(current_language, label_keys[state]),
                        record.name,
                        record.stream_type or "—",
                        record.channel_count,
                        f"{nominal} / {observed}",
                        format_label(record.channel_format),
                        record.source_id or record.hostname or "—",
                    ),
                    tags=(state,),
                )
            selected_key = selected["key"]
            if selected_key in records_by_key and tree.exists(selected_key):
                tree.selection_set(selected_key)
                tree.focus(selected_key)
            record = records_by_key.get(selected["key"] or "")
            if record:
                state = record.signal_state()
                state_detail_keys = {
                    "periodic-live": "state_live_detail",
                    "periodic-stale": "state_lost_detail",
                    "periodic-waiting": "state_wait_detail",
                    "irregular": "state_data_detail",
                }
                state_text = ui_text(current_language, state_detail_keys[state])
                detail_rows = (
                    ("detail_state", state_text),
                    ("detail_name", record.name),
                    ("detail_class", record.stream_type or "—"),
                    ("detail_channels", record.channel_count),
                    ("detail_format", format_label(record.channel_format)),
                    ("detail_nominal", f"{record.nominal_srate:g} Hz"),
                    ("detail_observed", f"{record.sample_rate_hz or 0:.1f} Hz"),
                    ("detail_source", record.source_id or "—"),
                    ("detail_host", record.hostname or "—"),
                    ("detail_uid", record.uid or "—"),
                    (
                        "detail_publisher",
                        publisher_origin_label(record.publisher_origin, current_language),
                    ),
                )
                detail = "\n".join(
                    f"{ui_text(current_language, label):<12}{value}" for label, value in detail_rows
                )
                detail += (
                    f"\n\n{ui_text(current_language, 'detail_metadata')}\n{record.description}"
                )
                preview = (
                    f"{ui_text(current_language, 'preview_timestamp')}: "
                    f"{human_timestamp(record.latest_timestamp, current_language)}\n"
                    f"{ui_text(current_language, 'preview_raw_timestamp')}: "
                    f"{record.latest_timestamp}\n"
                    f"{ui_text(current_language, 'preview_samples')}:   "
                    f"{record.sample_count}\n\n"
                    f"{format_sample_preview(record.latest_sample, current_language)}"
                )
            else:
                detail = ui_text(current_language, "select_stream")
                preview = ui_text(current_language, "no_stream")
            for widget, value in ((details, detail), (sample, preview)):
                widget.configure(state="normal")
                widget.delete("1.0", "end")
                widget.insert("1.0", value)
                widget.configure(state="disabled")
            update_publisher_controls(record)
            if schedule:
                root.after(self.refresh_ms, tick)

        def request_refresh() -> None:
            status_state["kind"] = "refreshing"
            status.configure(text=ui_text(language["value"], "refreshing"))
            async_catalog.request_refresh()

        def apply_language() -> None:
            current_language = language["value"]
            root.title(ui_text(current_language, "window_title"))
            title_label.configure(text=ui_text(current_language, "title"))
            subtitle_label.configure(text=ui_text(current_language, "subtitle"))
            refresh_button.set_tooltip(ui_text(current_language, "refresh_tooltip"))
            language_button.set_text(ui_text(current_language, "language"))
            streams_label.configure(text=ui_text(current_language, "streams_title"))
            details_label.configure(text=ui_text(current_language, "details_title"))
            publisher_refresh.set_text(ui_text(current_language, "publisher_refresh"))
            terminate.set_text(ui_text(current_language, "terminate"))
            sample_label.configure(text=ui_text(current_language, "sample_title"))
            for column, heading_key in heading_keys.items():
                tree.heading(column, text=ui_text(current_language, heading_key))
            tick(schedule=False)

        def toggle_language() -> None:
            language["value"] = "en" if language["value"] == "zh" else "zh"
            apply_language()

        def close_viewer() -> None:
            async_catalog.close()
            root.destroy()

        refresh_button.set_command(request_refresh)
        language_button.set_command(toggle_language)
        root.protocol("WM_DELETE_WINDOW", close_viewer)
        apply_language()
        async_catalog.start()
        tick()
        root.mainloop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Browse Lab Streaming Layer signals")
    parser.add_argument(
        "--lang",
        choices=("ch", "en", "zh"),
        default="zh",
        help="UI language (ch/zh for Chinese, en for English)",
    )
    args = parser.parse_args()
    language: Language = "en" if args.lang == "en" else "zh"
    TkLslViewer(language=language).run()
    return 0
