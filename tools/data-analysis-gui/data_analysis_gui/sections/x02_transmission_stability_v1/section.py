"""X02 v1 — whole-session transmission quality across recordings."""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from ...core.crossday import OTHER_SESSION_STYLE, SESSION_STYLES, recording_labels, recording_ticks
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure
from ..shared_frame_loss import fpga_analysis, transmission_totals


class TransmissionStability(Section):
    id = 'x02'
    mode = 'cross'
    title = '跨 session 传输质量'
    description = '按 session 对比整场 FPGA 推断缺帧率与长接收间隔时间占比，沿用 S03 统计口径。'
    cache_code_dependencies = ('sections/shared_frame_loss.py',)
    params = (
        Param('gap_threshold', '接收缺口阈值 / s', 'float', 0.15, minimum=0.02, maximum=10),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        threshold = ctx.params['gap_threshold']
        sessions = sorted(ctx.sessions, key=lambda session: (session.recording.date, session.recording.relative))
        rows = []
        labels = recording_labels(session.recording for session in sessions)
        for session, label in zip(sessions, labels):
            ctx.check_cancel()
            ctx.progress(f'transmission totals · {session.label}')
            def compute():
                fpga = ctx.cached(('s03_fpga', id(session)), lambda: fpga_analysis(session))
                return transmission_totals(session.neural_times, threshold, fpga)

            totals = ctx.cached(('s03_transmission_totals', id(session), threshold), compute)
            rows.append(dict(session=label, recording=session.recording.key,
                             date=session.recording.date, session_type=session.recording.data_type,
                             **totals))

        fig = new_figure(max(10, min(22, 4 + len(rows) * .55)), 7.8)
        axes = fig.subplots(2, 1, sharex=True)
        positions = np.arange(len(rows))
        dates = [row['date'] for row in rows]
        day_boundaries = [i - .5 for i in range(1, len(dates)) if dates[i] != dates[i - 1]]
        types = np.array([row['session_type'] for row in rows])
        styles = dict(SESSION_STYLES)
        styles.update({kind: OTHER_SESSION_STYLE
                       for kind in dict.fromkeys(types) if kind not in styles})
        summary = dict(session_count=len(rows))
        for ax, field, title in (
            (axes[0], 'fpga_frame_lost_rate', 'A  Total FPGA frame loss rate'),
            (axes[1], 'long_receive_gap_time_fraction', 'B  Total long receive gap time fraction'),
        ):
            values = np.asarray([row[field] for row in rows], dtype=float)
            valid = np.isfinite(values)
            for boundary in day_boundaries:
                ax.axvline(boundary, color='#c8c8c8', linestyle='--', linewidth=.9, zorder=1)
            ax.plot(positions, 100 * values, color='#687078', linewidth=1.4, zorder=2)
            for kind, (_, color, marker) in styles.items():
                selected = (types == kind) & valid
                if selected.any():
                    ax.plot(positions[selected], 100 * values[selected], linestyle='none',
                            marker=marker, color=color, markersize=6.5, clip_on=False, zorder=3)
            ax.set(title=title, ylabel='Percent')
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
            ax.grid(axis='y', alpha=.25)
            for index in positions[~valid]:
                ax.text(index, .93, 'N/A', transform=ax.get_xaxis_transform(), ha='center', va='top',
                        fontsize=8, color='#666666')
            summary[f'max_{field}'] = float(values[valid].max()) if valid.any() else None
        recording_ticks(axes[1], positions, [row['session'] for row in rows], fontsize=8)
        axes[1].set_xlabel('Session')
        axes[1].set_xlim(-.45, len(rows) - .55)
        handles = [Line2D([], [], linestyle='none', color=color, marker=marker, label=label)
                   for label, color, marker in styles.values()]
        axes[0].legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, 1.12),
                       ncol=4, frameon=False, fontsize=9)
        fig.suptitle(f'Transmission quality across sessions · receive gap threshold > {1000 * threshold:g} ms')
        fig.supxlabel(
            'A  FPGA loss (%) = 100 × inferred missing frames / (FPGA-matched received frames + inferred missing frames).\n'
            '     Missing frames per valid step = max(round(FPGA step / 19.2 ms) − 1, 0). N/A = no usable FPGA estimate.\n'
            f'B  Long-gap time (%) = 100 × sum of full receive intervals > {1000 * threshold:g} ms / first-to-last receive time.\n'
            'Reading guide: Lower is better for each metric. A measures inferred missing frames; B measures time in long '
            'receive gaps. Compare sessions at the same threshold; no fixed pass/fail cutoff.',
            fontsize=8, multialignment='left',
        )

        result = SectionResult(summary=summary)
        result.tables['session_transmission'] = rows
        result.add_figure('Transmission quality across sessions', fig, method=(
            'Each point is a whole recording, ordered by recording date and file. The analysis uses all '
            'neural frames regardless of the trial filter. FPGA frame loss rate = inferred missing frames / '
            '(FPGA-matched received frames + inferred missing frames), using the same 19.2 ms clock-step '
            'inference as S03. Long receive gap time fraction = sum of the full adjacent receive intervals '
            f'greater than {threshold:g} s / the first-to-last receive span. Percent units on both axes. '
            'Sessions are equally spaced; light grey dashed lines separate recording dates. Colour and '
            'marker shape identify the recording type. Missing values are labelled N/A and break the line.'),
            data=dict(session_transmission=rows))
        return result


SECTION = TransmissionStability()
