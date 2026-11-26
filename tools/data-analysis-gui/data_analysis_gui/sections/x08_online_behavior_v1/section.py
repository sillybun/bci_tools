"""Compare online behavior separately for CO8 and Grid."""
from collections import defaultdict
from ...core import behavior
from ...core.crossday import recording_labels
from ..base import Section, SectionResult
from ..s13_online_behavior_v1.section import behavior_figure
from ..s13_online_behavior_v1 import co8_bps, rolling_bps, run_behavior


class CrossOnlineBehavior(Section):
    id = 'x08'
    mode = 'cross'
    min_recordings = 1
    title = '在线行为表现对比'
    description = '按任务比较在线行为，并按run汇总60秒有效时间滚动BPS的median/IQR和peak；窗口不跨session。'
    cache_code_dependencies = ('sections/s13_online_behavior_v1',)

    def extra_signature_inputs(self, recordings, params):
        from ...storage import optional_digest

        return {str(path): optional_digest(path) for rec in recordings if rec.online and rec.task_mode == 'grid'
                if (path := rolling_bps.grid_run_config_path(rolling_bps.run_key(rec)))}

    def run(self, ctx):
        sessions = sorted((s for s in ctx.sessions
                           if s.recording.online and s.task_mode in behavior.TASK_LABELS),
                          key=lambda s: (s.recording.date, s.recording.key))
        if not sessions:
            raise ValueError('请选择在线 CO8 或 Grid 记录')
        result = SectionResult()
        labels = recording_labels(s.recording for s in sessions)
        grouped = {mode: [] for mode in behavior.TASK_LABELS}
        rolling_groups = {mode: [] for mode in behavior.TASK_LABELS}
        trial_groups = {mode: defaultdict(list) for mode in behavior.TASK_LABELS}
        co8_trials, co8_runs, used_csv = defaultdict(list), {}, set()
        co8_recordings = {}
        # Within one run, a complete CSV supersedes an overlapping short XDF.
        collected = {id(s): behavior.collect(s) for s in sessions}
        full_csv_runs = {rolling_bps.run_key(s.recording) for s in sessions
                         if collected[id(s)][1] != 'XDF event markers'}
        for session, label in zip(sessions, labels):
            ctx.check_cancel()
            ctx.progress(f'Behavior · {label}')
            trials, source, note = collected[id(session)]
            if session.task_mode == 'centerout':
                trials = co8_bps.add_xdf_attempt_times(session, trials, source)
            run_id = rolling_bps.run_key(session.recording)
            if source == 'XDF event markers' and run_id in full_csv_runs:
                note += ' Run滚动BPS使用本run的完整CSV，跳过重叠XDF片段。'
            else:
                trial_groups[session.task_mode][run_id].extend(trials)
                rolling_groups[session.task_mode].extend(rolling_bps.windows(
                    trials, session.task_mode, session=label, date=session.recording.date, run=run_id))
                if session.task_mode == 'centerout':
                    csv = behavior.source_files(session.recording).get('trials')
                    csv_key = str(csv.resolve()) if csv else None
                    if csv_key is None or csv_key not in used_csv:
                        co8_trials[run_id].extend(trials)
                        co8_runs[run_id] = dict(session=label, date=session.recording.date, run_key=run_id)
                        co8_recordings[run_id] = session.recording
                        if csv_key:
                            used_csv.add(csv_key)
            row = behavior.metrics(trials, session.task_mode)
            if session.task_mode == 'centerout':
                row.update(co8_bps.co8_metrics(trials, row['bps']))
            else:
                row.update(co8_bps.grid_metrics(trials))
            grouped[session.task_mode].append(dict(session=label, date=session.recording.date, run_key=run_id,
                                                   source=source, note=note, **row))
        for mode, rows in grouped.items():
            task = behavior.TASK_LABELS[mode]
            result.summary[f'{task.lower()}_sessions'] = len(rows)
            if rows:
                title = f'Online {task} Behavior'
                if mode == 'centerout':
                    run_rows = [dict(co8_runs[key], **behavior.metrics(trials, mode))
                                for key, trials in co8_trials.items()]
                    for row in run_rows:row.pop('bps', None)
                    for row, label in zip(run_rows, recording_labels(co8_recordings.values())):
                        row['session'] = label
                    result.add_figure(run_behavior.TITLE, run_behavior.run_figure(run_rows),
                                      method=run_behavior.METHOD, data={'runs': run_rows})
                    result.tables['co8_run_behavior'] = run_rows
                else:
                    result.add_figure(title, behavior_figure(rows, mode, title, session_labels=True),
                                      method=behavior.METHOD, data={'sessions': rows})
                rolling = rolling_groups[mode]
                runs = rolling_bps.run_summary(rolling, mode, runs=rows, trials_by_run=trial_groups[mode])
                for peak in ((None,) if mode == 'grid' else (False, True)):
                    title, fig = rolling_bps.run_figure(runs, mode, peak=peak)
                    result.add_figure(title, fig, method=rolling_bps.METHOD,
                                      data={'runs': runs, 'windows': rolling})
                result.tables[f'{task.lower()}_rolling_bps'] = rolling
                result.tables[f'{task.lower()}_run_rolling_bps'] = runs
                for row in rows:
                    for key in ('bps', 'neuralink_bps', 'wolpaw_bps', 'fitts_bps'):
                        row.pop(key, None)
                result.tables[f'{task.lower()}_sessions'] = rows
        return result


SECTION = CrossOnlineBehavior()
