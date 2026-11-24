"""Section contract.

Each analysis lives in its own versioned folder `data_analysis_gui/sections/<id>_<name>_v<N>/`
containing `section.py` that defines a module-level `SECTION` instance. The registry
discovers folders automatically; when several versions of one id exist the highest
version is shown by default and older ones stay selectable.

A section receives an `AnalysisContext` and returns a `SectionResult`. Sections
build matplotlib `Figure` objects directly (never pyplot) so they can run in a
worker thread; the GUI embeds and exports them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from matplotlib.figure import Figure

from ..core.filters import Cohort, TrialFilter
from ..core.session import Session


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    kind: str = 'float'            # int | float | bool | choice | recording | text | file
    default: Any = None
    choices: tuple = ()
    help: str = ''
    minimum: float | None = None
    maximum: float | None = None

    def parse(self, raw):
        if self.kind == 'bool':
            return bool(raw) if not isinstance(raw, str) else raw.lower() in ('1', 'true', 'yes', 'on')
        if self.kind == 'recording':
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f'{self.label}: 请选择参考记录')
            return raw
        if self.kind in ('text', 'file'):
            return str(raw or '').strip()
        if self.kind == 'choice':
            if raw not in self.choices:
                raise ValueError(f'{self.label}: 可选值为 {", ".join(map(str, self.choices))}')
            return raw
        if raw in (None, ''):
            raise ValueError(f'{self.label}: 不能为空')
        value = int(float(raw)) if self.kind == 'int' else float(raw)
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f'{self.label}: 不能小于 {self.minimum}')
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f'{self.label}: 不能大于 {self.maximum}')
        return value


@dataclass
class AnalysisContext:
    """Everything a section may read. Single-day: one item; cross-day: several."""
    sessions: list[Session]
    cohorts: list[Cohort]
    criteria: TrialFilter
    params: dict[str, Any]
    layout: dict | None = None               # physical workbook layout (may be None)
    channel_config_path: str = ''           # <recording day folder>/configs.ini, '' if absent
    progress: Callable[[str], None] = lambda message: None
    cache: dict = field(default_factory=dict)   # shared across sections for the same selection
    cancelled: Callable[[], bool] = lambda: False
    artifacts: Any = None                    # optional session cache, shared by single and cross jobs

    @property
    def session(self) -> Session:
        return self.sessions[0]

    @property
    def cohort(self) -> Cohort:
        return self.cohorts[0]

    def cached(self, key, compute):
        """Memoize expensive intermediate results across sections (same data + filter)."""
        if key not in self.cache:
            self.cache[key] = self.artifacts.remember(key, compute) if self.artifacts is not None else compute()
        return self.cache[key]

    def check_cancel(self):
        if self.cancelled():
            raise CancelledError()


class CancelledError(Exception):
    pass


@dataclass
class SectionResult:
    figures: list[tuple[str, Figure]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    tables: dict[str, list[dict]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # Per figure (same order as `figures`): calculation method text and the exact
    # data behind each panel. Exported next to the PNG/SVG/PDF as CSV.
    methods: list[str] = field(default_factory=list)
    figure_data: list[dict[str, Any]] = field(default_factory=list)
    # Standalone interactive results, stored and hashed alongside static figures.
    attachments: dict[str, str | bytes] = field(default_factory=dict)

    def add_figure(self, title: str, fig: Figure, method: str = '', data: dict | None = None):
        """Register a figure.

        method: how every panel was computed (inputs, windows, statistics, folds,
                normalisation) in plain sentences; written into the CSV header and
                figure.json so the figure can be reproduced.
        data:   {name: table} with the values actually plotted. A table is a list of
                row dicts or a dict of equal-length 1-D arrays/lists. Each becomes
                `<figure>__<name>.csv`. If omitted, plotted values are extracted from
                the matplotlib artists automatically (lines, scatter, bars, images).
        """
        self.figures.append((title, fig))
        self.methods.append(method)
        self.figure_data.append(data or {})
        return fig


class Section:
    id: str = ''                 # e.g. 's01' (single-day) or 'x01' (cross-day)
    title: str = ''
    mode: str = 'single'         # single | cross
    version: str = 'v1'
    description: str = ''
    params: tuple[Param, ...] = ()
    requires: tuple[str, ...] = ()     # e.g. ('online',) or ('channel_config',)
    min_recordings: int = 2           # minimum for cross-session sections
    source_sections: tuple[str, ...] = ()  # source settings inherited by combined analyses

    def defaults(self) -> dict:
        return {p.name: p.default for p in self.params}

    def resolve_params(self, raw: dict | None = None) -> dict:
        raw = raw or {}
        values = self.defaults()
        for param in self.params:
            if param.name in raw:
                values[param.name] = param.parse(raw[param.name])
        if self.source_sections:
            from . import versions
            sources = raw.get('_source_sections', {})
            values['_source_sections'] = {}
            for sid in self.source_sections:
                source = sources.get(sid, {})
                available = versions(sid)
                selected = next((s for s in available if s.version == source.get('version')), None)
                if selected is None:
                    if source.get('version'):
                        raise ValueError(f'未找到来源分析 {sid} {source["version"]}')
                    selected = available[-1]
                values['_source_sections'][sid] = dict(
                    version=selected.version, params=selected.resolve_params(source.get('params')))
        return values

    def available(self, ctx_sessions: list[Session], channel_config_path: str = '') -> str:
        """Empty string if runnable, otherwise a human-readable reason."""
        if self.mode == 'cross' and len(ctx_sessions) < self.min_recordings:
            return f'至少选择 {self.min_recordings} 份记录'
        if 'online' in self.requires and not any(s.decoder_velocity is not None for s in ctx_sessions):
            return '需要包含 cursortasks_decoder_velocity 的在线记录'
        if 'channel_config' in self.requires and not channel_config_path:
            return '未找到该记录的通道配置 INI 或 Neuralinsight 日志'
        return ''

    def run(self, ctx: AnalysisContext) -> SectionResult:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def display(self) -> str:
        return f'{self.id.upper()} · {self.title}'


def new_figure(width_in=10.0, height_in=6.0, dpi=100) -> Figure:
    fig = Figure(figsize=(width_in, height_in), dpi=dpi, layout='constrained')
    return fig
