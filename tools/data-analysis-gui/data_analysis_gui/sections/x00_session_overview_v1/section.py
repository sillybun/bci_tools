"""X0 exports the live cross-session selection diagrams."""
from ..base import Section, SectionResult
from ..session_overview import METHOD, PAGES, session_overview


class RecordingOverview(Section):
    id = 'x00'
    mode = 'cross'
    min_recordings = 1
    title = '记录示意图'
    description = '按日期展示所选 session。首图汇总全部记录，后四页分别显示四种任务和在线状态组合。'

    @property
    def display(self):
        return f'X0 · {self.title}'

    def run(self, ctx):
        result = SectionResult(summary={'session_count': len(ctx.sessions)})
        recordings = [session.recording for session in ctx.sessions]
        for kind, title in PAGES:
            ctx.check_cancel()
            figure, rows = session_overview(recordings, kind)
            method = METHOD if kind is None else METHOD + f' This page includes only {title} sessions.'
            result.add_figure(title, figure, method=method, data={'sessions': rows})
            if kind is None:
                result.tables['sessions'] = rows
            else:
                result.summary[kind] = len(rows)
        return result


SECTION = RecordingOverview()
