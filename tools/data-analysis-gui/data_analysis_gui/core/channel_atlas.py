"""64-channel atlas pages with explicitly independent, fully labelled y axes."""
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from . import style


def channel_pages(ids, significant, q, pvalues=None):
    ids = np.asarray(ids)
    orders = [('all channels', np.argsort(ids, kind='stable'))]
    selected = np.flatnonzero(significant)
    selected = selected[np.lexsort((ids[selected],
                                   np.asarray(pvalues if pvalues is not None else q)[selected],
                                   q[selected]))]
    orders.append(('significant channels', selected))
    for name, order in orders:
        pages = max(1, (len(order) + 63) // 64)
        for page in range(pages):
            yield name, page + 1, pages, order[page * 64:(page + 1) * 64]


def atlas_figure(heading, page, pages, banner, xlabel, footer):
    # A fixed layout avoids expensive constrained-layout passes on 64 panels.
    fig = Figure(figsize=(17, 14.5), dpi=100)
    axes = fig.subplots(8, 8, sharex=False, sharey=False).ravel()
    fig.subplots_adjust(left=.045, right=.988, bottom=.07, top=.91, hspace=.66, wspace=.58)
    fig.suptitle(f'{heading} · page {page}/{pages}\n{banner}', fontsize=9, y=.985)
    fig.supxlabel(xlabel, fontsize=9, y=.035)
    fig.supylabel('Rate / Hz · independent y scale in every panel', fontsize=9, x=.008)
    fig.text(.5, .009, footer, ha='center', fontsize=7)
    for ax in axes:
        ax.tick_params(axis='both', labelsize=5, length=2, pad=1,
                       labelleft=True, labelbottom=True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    return fig, axes


def append_anova_atlas(result, info, ids, banner):
    st, q, sig, active = info['st'], info['q'], info['sig'], info['active']
    for group, page, pages, order in channel_pages(ids, sig, q, st['p']):
        heading = f'ANOVA tuning curves - {group}'
        fig, axes = atlas_figure(heading, page, pages, banner, 'Target direction / deg',
                                'Mean ± SEM; dashed = trial-weighted grand mean. Red * = S05 direction significant; '
                                'blue = not significant; grey = inactive. Each panel has its own y ticks.')
        rows = []
        for ax, j in zip(axes, order):
            color = style.GREY if not active[j] else ('#c0392b' if sig[j] else '#2176ae')
            ax.errorbar(np.arange(8)*45, st['mean'][:, j], yerr=st['sem'][:, j],
                        fmt='o-', ms=2, lw=.75, elinewidth=.5, capsize=1.3, color=color)
            ax.axhline(info['mean_hz'][j], color=style.GREY, ls='--', lw=.5)
            ax.set_xticks([0, 90, 180, 270, 315])
            ax.set_xlim(-15, 330)
            ax.set_title(f'E{int(ids[j])}{" *" if sig[j] else ""} · q={q[j]:.1g}', fontsize=6, color=color, pad=3)
            for d in range(8):
                rows.append(dict(electrode=int(ids[j]), stored_column=int(j+1), direction_deg=d*45,
                                 mean_hz=float(st['mean'][d,j]), sem_hz=float(st['sem'][d,j]),
                                 grand_mean_hz=float(info['mean_hz'][j]),
                                 q=float(q[j]), significant=bool(sig[j]), active=bool(active[j])))
        for ax in axes[len(order):]:
            ax.axis('off')
        if not len(order):
            axes[0].text(.5, .5, 'No significant channels', ha='center', transform=axes[0].transAxes)
        fig._series_pdf_name = heading
        result.add_figure(f'{heading} {page}', fig, method=(
            'S05 ordinary direction ANOVA; significant = its active-family BH q < source alpha. '
            'All transmitted channels sorted by physical electrode ID, including inactive channels. '
            'Significant subset sorted by q, p, electrode ID. Pages have 8x8 slots, up to 64 channels; '
            'empty final slots hidden. Each subplot independently autoscales y and displays its own '
            'y tick labels; no shared y axis. Rate = trial-window Hz; direction mean ± SEM '
            '(sample SD/sqrt(direction trials)); dashed = trial-weighted grand mean. '
            'No gain-controlled substitution. ' + banner), data={'channel_tuning': rows})


def append_psth_atlas(result, info, ids, p, banner, smooth):
    b, dirs, times = info['binned'], info['dirs'], info['centres']
    raw = np.full((8, len(times), len(ids)), np.nan)
    counts = np.zeros(8, int)
    for d in range(8):
        selected = dirs == d
        counts[d] = selected.sum()
        if counts[d]:
            raw[d] = np.nanmean(b[selected], axis=0)
    plotted = smooth(raw, info['sigma'], axis=1)
    for group, page, pages, order in channel_pages(ids, info['sig'], info['q'], info['p_ch']):
        heading = f'PSTH curves - {group}'
        fig, axes = atlas_figure(heading, page, pages, banner, f'Time from {p["align"]} / s',
                                '* = S06 pooled temporal response, not direction selectivity. '
                                'Shaded = baseline; dotted = event. Each panel has its own y ticks.')
        handles = [Line2D([], [], color=style.direction_color(d), label=f'{d*45}°', lw=1) for d in range(8)]
        fig.legend(handles=handles, ncol=8, loc='upper center', bbox_to_anchor=(.5, .95),
                   frameon=False, fontsize=7)
        rows = []
        for ax, j in zip(axes, order):
            for d in range(8):
                if not counts[d]:
                    continue
                ax.plot(times, plotted[d,:,j], lw=.7, color=style.direction_color(d))
                row = dict(electrode=int(ids[j]), stored_column=int(j+1), direction_deg=d*45,
                           direction_trials=int(counts[d]), temporal_q=float(info['q'][j]),
                           temporal_significant=bool(info['sig'][j]))
                row.update({f'raw_bin_{k}_t_{t:+.4f}s':float(v) for k,(t,v) in enumerate(zip(times,raw[d,:,j]))})
                row.update({f'plotted_bin_{k}_t_{t:+.4f}s':float(v) for k,(t,v) in enumerate(zip(times,plotted[d,:,j]))})
                rows.append(row)
            ax.axvspan(p['baseline_start'], p['baseline_end'], color='#e8e6df', zorder=-2, lw=0)
            ax.axvline(0, color=style.INK, lw=.5, ls=':')
            ax.set_xlim(p['tmin'], p['tmax'])
            color = style.GREY if not info['active'][j] else ('#c0392b' if info['sig'][j] else style.INK)
            ax.set_title(f'E{int(ids[j])}{" *" if info["sig"][j] else ""} · q={info["q"][j]:.1g}',
                         fontsize=6, color=color, pad=3)
        for ax in axes[len(order):]:
            ax.axis('off')
        if not len(order):
            axes[0].text(.5,.5,'No significant channels',ha='center',transform=axes[0].transAxes)
        fig._series_pdf_name = heading
        result.add_figure(f'{heading} {page}', fig, method=(
            'S06 exact pooled-direction temporal response test: unsmoothed post-event bins vs '
            'same-trial baseline paired t tests; min-p bin Bonferroni, then BH over active channels. '
            'Significant subset uses the existing S06 q<alpha; it is not an ANOVA subset or '
            'direction-specific temporal test. Each panel shows eight direction-mean PSTHs in Hz; '
            'NaN-aware Gaussian smoothing affects display only. All channels sorted by electrode ID; '
            'significant subset by q, channel p, electrode ID. Up to 64 panels in an 8x8 grid. '
            'Independent y limits AND visible y tick labels for every subplot; empty slots hidden. '
            'Raw and displayed bin means exported separately. ' + banner), data={'direction_psth': rows})
