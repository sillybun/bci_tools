"""Block-level performance for one online session."""
from collections import defaultdict

import numpy as np

from ...core import behavior
from ...core.crossday import recording_ticks
from ..base import Section, SectionResult, new_figure
from . import co8_bps, rolling_bps


def behavior_figure(rows, mode, title, *, session_labels=False):
    width = max(10, min(22, 5 + .5 * len(rows))) if session_labels else 10
    height = 8 if session_labels else 7
    if mode == 'centerout' and not session_labels:
        height *= 0.8
    fig = new_figure(width, height)
    panels = [('success_pct', 'Success rate (%)'), ('rt_median_s', 'Time to target (s)')]
    axes = fig.subplots(len(panels), 1, sharex=True)
    x = np.arange(len(rows))
    for ax, (key, ylabel) in zip(axes, panels):
        values = np.array([r[key] for r in rows])
        if np.isfinite(values).any():
            if key == 'rt_median_s':
                q1 = np.array([r['rt_q1_s'] for r in rows])
                q3 = np.array([r['rt_q3_s'] for r in rows])
                ax.errorbar(x, values, yerr=[values - q1, q3 - values], fmt='o-', capsize=3,
                            color='#a65b24', label='Median and IQR')
                ax.legend(frameon=False)
            else:
                ax.plot(x, values, 'o-', color='#286e9e')
        else:
            ax.text(.5, .5, 'No verified values', transform=ax.transAxes, ha='center')
        if key == 'success_pct':
            ax.set_ylim(0, 105)
            if mode != 'centerout' or session_labels:
                for i, row in enumerate(rows):
                    if row['attempts']:
                        offset, align = (5, 'bottom') if values[i] < 15 else (-5, 'top')
                        ax.annotate(f'{row["successes"]}/{row["attempts"]}', (i, values[i]), (0, offset),
                                    textcoords='offset points', ha='center', va=align, fontsize=6.5)
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.23)
    axes[-1].set_xlim(-.5, len(rows) - .5)
    if session_labels:
        recording_ticks(axes[-1], x, [r['session'] for r in rows], fontsize=8)
    else:
        step = max(1, len(rows) // 12)
        ticks = sorted(set(range(0, len(rows), step)) | {len(rows) - 1})
        axes[-1].set_xticks(ticks, [str(rows[i]['block'] or 'Unknown') for i in ticks])
        axes[-1].set_xlabel('Task block')
    fig.suptitle(title)
    return fig


class OnlineBehavior(Section):
    id = 's13'
    title = '在线行为表现'
    description = '保留按 block 的在线行为统计；新增累计60秒有效尝试时间的滚动BPS与峰值，排除ITI和反馈等待。'
    requires = ('online',)

    def available(self, ctx_sessions, channel_config_path=''):
        if not ctx_sessions[0].recording.online or ctx_sessions[0].task_mode not in behavior.TASK_LABELS:
            return '需要一份在线 CO8 或 Grid 记录'
        return ''

    def run(self, ctx):
        session = ctx.session
        trials, source, note = behavior.collect(session)
        if not trials:
            raise ValueError('该记录没有可分析的行为 trial')
        is_co8 = session.task_mode == 'centerout'
        if is_co8:
            trials = co8_bps.add_xdf_attempt_times(session, trials, source)
        grouped = defaultdict(list)
        for trial in trials:
            grouped[trial['block']].append(trial)
        def summarize(group):
            row = behavior.metrics(group, session.task_mode)
            if is_co8:
                row.update(co8_bps.co8_metrics(group, row['bps']))
            else:
                row.update(co8_bps.grid_metrics(group))
            return row

        rows = [dict(block=block, **summarize(grouped[block]))
                for block in sorted(grouped, key=lambda b: (b is None, b or 0))]
        title = 'Online Behavior by Block'
        fig = behavior_figure(rows, session.task_mode,
                              f'{session.label} · {behavior.TASK_LABELS[session.task_mode]} Behavior')
        result = SectionResult()
        result.add_figure(title, fig, method=behavior.METHOD, data={'blocks': rows})
        rolling = rolling_bps.windows(trials, session.task_mode, session=session.label,
                                      date=session.recording.date)
        daily = rolling_bps.daily_summary(rolling, session.task_mode, dates=[session.recording.date])
        rolling_title = f'Online {behavior.TASK_LABELS[session.task_mode]} Rolling BPS'
        result.add_figure(rolling_title,
                          rolling_bps.rolling_figure(rolling, session.task_mode, f'{session.label} · 60-active-second rolling BPS'),
                          method=rolling_bps.METHOD, data={'windows': rolling, 'peaks': daily})
        result.tables.update(rolling_bps=rolling, rolling_bps_peaks=daily)
        for row in rows:
            for key in ('bps', 'neuralink_bps', 'wolpaw_bps', 'fitts_bps'):
                row.pop(key, None)
        result.tables.update(block_metrics=rows, trials=trials)
        result.summary.update(recording=session.label, behavior_source=source,
                              **summarize(trials))
        result.summary.update(bps_window_seconds=60, bps_step_seconds=1, bps_short_run_policy='ratio_actual_duration_only_if_entire_run_under_60s', bps_summary_statistic='median_of_60s_step1s_windows_or_short_run_ratio')
        for key, _ in rolling_bps.methods(session.task_mode):
            result.summary[key] = daily[0][f'{key}_median']
            result.summary[f'{key}_peak'] = daily[0][f'{key}_peak']
        if is_co8:
            result.summary['bps'] = result.summary['fitts_bps']
        if note:
            result.notes.append(note)
        return result


SECTION = OnlineBehavior()
