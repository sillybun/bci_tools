"""Headless section execution shared by the GUI, tests and the CLI."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .core.channels import read_layout
from .core.style import DEFAULT_FONT_PERCENT, finish_figure
from .core.filters import Cohort, TrialFilter, apply_filter
from .core.session import Session, get_session
from .sections.base import AnalysisContext, Section, SectionResult  # noqa: F401


def build_context(sessions: list[Session], criteria: TrialFilter, params: dict | None = None,
                  section: Section | None = None, layout_path=None, channel_config_path=None,
                  cache=None, progress=None, cancelled=None, artifacts=None) -> AnalysisContext:
    values = section.resolve_params(params) if section else {}
    layout = read_layout(layout_path)
    if layout_path and layout is None:
        raise ValueError(f'找不到通道映射表：{layout_path}')
    cache = {} if cache is None else cache
    cohorts = []
    for session in sessions:
        key = ('cohort', id(session), criteria)
        if key not in cache:
            if artifacts is None:
                cache[key] = apply_filter(session, criteria)
            else:
                def filtered():
                    cohort = apply_filter(session, criteria)
                    return cohort.table, cohort.include
                table, include = artifacts.remember(('cohort', id(session), criteria.to_dict()), filtered)
                cache[key] = Cohort(session, criteria, table, include, artifacts=artifacts)
        if artifacts is not None:
            artifacts.bind_cohort(cache[key])
        cohorts.append(cache[key])
    ctx = AnalysisContext(sessions=sessions, cohorts=cohorts, criteria=criteria, params=values, layout=layout,
                          channel_config_path=_config_for(sessions, section, channel_config_path), cache=cache,
                          artifacts=artifacts)
    if progress:
        ctx.progress = progress
    if cancelled:
        ctx.cancelled = cancelled
    return ctx


def _config_for(sessions, section, explicit):
    """Explicit path (tests) or the single recording's own day configs.ini; never another day's."""
    if explicit is not None:
        return explicit or ''
    if not sessions or (section is not None and section.mode != 'single'):
        return ''
    from .core.catalog import day_config_path
    try:
        return day_config_path(sessions[0].recording.path)
    except (OSError, ValueError):
        return ''


def run_section(section: Section, ctx: AnalysisContext, *, font_percent=DEFAULT_FONT_PERCENT) -> SectionResult:
    reason = section.available(ctx.sessions, ctx.channel_config_path)
    if reason:
        raise ValueError(reason)
    result = section.run(ctx)
    if ctx.channel_config_path:
        from .core.channels import read_channel_config
        from .storage import uses_channel_config
        if uses_channel_config(section):
            try:
                source = read_channel_config(ctx.channel_config_path).get('_source')
            except (OSError, ValueError):
                source = None  # Optional config failures are reported by the section.
            if source:
                result.summary['channel_config_source'] = source
                result.notes.append(f'通道分类取自 Neuralinsight 日志最后一次保存的筛选结果：'
                                    f'{source["timestamp"]}，第 {source["line"]} 行；'
                                    '实际传输通道和列顺序使用本记录的 XDF 元数据。')
    for session in ctx.sessions:
        note = getattr(session, 'header', {}).get('analysis_window_note')
        if note:
            result.notes.append(f'{session.label}：{note}')
    if section.mode == 'cross' and section.id not in ('x00', 'x08', 'x09', 'x10'):
        from .sections.session_overview import METHOD, TITLE, session_overview
        excluded = set(result.summary.get('excluded_dates', ()))
        figure, rows = session_overview(s.recording for s in ctx.sessions if s.recording.date not in excluded)
        result.add_figure(TITLE, figure, method=METHOD, data={'sessions': rows})
        for entries in (result.figures, result.methods, result.figure_data):
            entries.insert(0, entries.pop())
    for _, figure in result.figures:
        if section.mode == 'single' and ctx.sessions:
            run_label = ctx.session.recording.run_name
            if run_label and not any(run_label in label.get_text() for label in figure.texts):
                figure.text(0.99, 0.005, run_label, ha='right', va='bottom', fontsize=8)
        finish_figure(figure, font_percent)
    from .sections.result_summaries import result_sentence
    result.summary['one_sentence_summary'] = result_sentence(section.id, result.summary)
    return result


def load_sessions(recordings) -> list[Session]:
    return [get_session(r) for r in recordings]
