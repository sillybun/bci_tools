"""Direction-specific temporal response tests and descriptive example PSTHs."""
from dataclasses import replace
import numpy as np

from ...core import analysis as A, style
from ..base import new_figure
from ..s05_anova_v1.section import channel_anova
from ..s06_psth_v1.section import temporal_statistics, paired_t


METHOD = (
    'Preferred direction: among S05 ordinary-ANOVA significant channels, the unique maximum '
    'of direction-mean window Hz; ties (rtol=1e-7, atol=1e-10) are unassigned. This is a '
    'descriptive preference, not a significant comparison against every other direction. '
    'Direction-specific time response: reuse S06 alignment, baseline, raw bins and alpha, '
    'but split trials by direction; require at least 8 baseline-eligible trials per direction '
    'and at least 8 finite paired observations per tested bin/channel. Two-sided paired t '
    'tests vs same-trial baseline; min-p times ALL post-event bins (Bonferroni); BH jointly '
    'over all eligible positive-mean channel-by-direction pairs. Untestable pairs remain N/A. '
    'Direction-and-time(d) = S05 globally significant AND temporal response in d; '
    'preferred-and-time(d) additionally requires preferred direction d. Neither is a '
    'pairwise direction-selectivity test. Both differ from pooled S05/S06 Both. Counts across directions for '
    'temporal responses can repeat electrodes. Excitation and suppression both count. '
    'These tests do not test direction-by-time interaction or held-out direction decoding. '
)


def direction_statistics(ctx, groups):
    p5 = groups['sources']['s05']['params']
    p6 = groups['sources']['s06']['params']
    st, _, _ = channel_anova(replace(ctx, params=p5))
    means = st['mean']
    safe = np.where(np.isfinite(means), means, -np.inf)
    maximum = safe.max(axis=0)
    ties = np.isclose(safe, maximum[None], rtol=1e-7, atol=1e-10).sum(axis=0) != 1
    preferred = np.argmax(safe, axis=0)
    preferred[~groups['direction'] | ties | ~np.isfinite(maximum)] = -1
    info = temporal_statistics(replace(ctx, params=p6))
    post = info['post']
    n_ch = len(preferred)
    p = np.ones((8, n_ch))
    eligible = np.zeros((8, n_ch), bool)
    peak_time = np.full((8, n_ch), np.nan)
    n_trials = np.zeros(8, int)
    curves = np.full((8, len(post), n_ch), np.nan)
    sem = np.full_like(curves, np.nan)
    for d in range(8):
        selected = info['valid'] & (info['dirs'] == d)
        n_trials[d] = selected.sum()
        if not selected.any():
            continue
        delta = info['binned'][selected][:, post] - info['base'][selected, None, :]
        mean, pbin, n = paired_t(delta)
        curves[d] = mean
        sem[d] = np.nanstd(delta, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
        if n_trials[d] < 8:
            continue
        finite_bin = n >= 8
        active = np.nanmean(info['binned'][selected], axis=(0, 1)) > 0
        eligible[d] = active & finite_bin.any(axis=0)
        corrected = np.where(finite_bin, pbin, 1).min(axis=0) * len(post)
        p[d] = np.minimum(corrected, 1)
        peak = np.argmax(np.where(finite_bin, np.abs(mean), -np.inf), axis=0)
        peak_time[d, eligible[d]] = info['centres'][post[peak[eligible[d]]]]
    q = np.full((8, n_ch), np.nan)
    family = np.flatnonzero(eligible.ravel())
    if len(family):
        q[:] = A.bh_fdr(p.ravel(), family).reshape(p.shape)
        q[~eligible] = np.nan
    sig = eligible & (q < p6['alpha'])
    rows, channel_rows = [], []
    for d in range(8):
        pd = preferred == d
        rows.append(dict(direction_deg=d * 45, baseline_trials=int(n_trials[d]),
                         preferred_channels=int(pd.sum()), tested_time_channels=int(eligible[d].sum()),
                         time_channels=int(sig[d].sum()) if eligible[d].any() else None,
                         direction_and_time_channels=int((groups['direction'] & sig[d]).sum()) if eligible[d].any() else None,
                         preferred_and_time_channels=int((pd & sig[d]).sum()) if eligible[d].any() else None))
        for j, electrode in enumerate(ctx.session.electrode_ids):
            channel_rows.append(dict(electrode=int(electrode), direction_deg=d * 45,
                                     preferred=bool(pd[j]), direction_anova_significant=bool(groups['direction'][j]),
                                     temporal_tested=bool(eligible[d, j]),
                                     temporal_q=float(q[d, j]) if eligible[d, j] else None,
                                     temporal_significant=bool(sig[d, j]) if eligible[d, j] else None,
                                     peak_abs_delta_time_s=float(peak_time[d, j]) if eligible[d, j] else None))
    return dict(rows=rows, channels=channel_rows, preferred=preferred, sig=sig, q=q,
                eligible=eligible, curves=curves, sem=sem, peak_time=peak_time,
                times=info['centres'][post], n_trials=n_trials, window_means=means,
                unassigned=int((groups['direction'] & (preferred < 0)).sum()))


def draw_counts(ctx, info):
    fig = new_figure(13, 4.5)
    axes = fig.subplots(1, 3)
    for ax, field, title, color in zip(axes,
            ('preferred_channels', 'time_channels', 'preferred_and_time_channels'),
            ('A  Preferred direction (S05 significant)', 'B  Time response within direction',
             'C  Direction + time response'), ('#c0392b', '#d77b35', '#806ba5')):
        values = [r[field] for r in info['rows']]
        ax.bar(range(8), [v or 0 for v in values], color=color)
        ax.set_xticks(range(8), [f'{d * 45}°' for d in range(8)], rotation=45)
        ax.set(title=title, ylabel='Channels', ylim=(0, max([v or 0 for v in values] + [1]) * 1.22))
        for d, v in enumerate(values):
            ax.text(d, v or 0, 'N/A' if v is None else str(v), ha='center', va='bottom', fontsize=8)
    ax = axes[2]
    ax.clear()
    for offset, field, label, color in [(-.2, 'direction_and_time_channels', 'S05 significant ∩ time(d)', '#806ba5'),
                                       (.2, 'preferred_and_time_channels', 'Preferred(d) ∩ time(d)', '#b7a6d3')]:
        values = [r[field] for r in info['rows']]
        ax.bar(np.arange(8) + offset, [v or 0 for v in values], width=.38, label=label, color=color)
        for d, v in enumerate(values):
            ax.text(d + offset, v or 0, 'N/A' if v is None else str(v), ha='center', va='bottom', fontsize=6)
    ax.set_xticks(range(8), [f'{d*45}°' for d in range(8)], rotation=45)
    ax.set(title='C  Direction + time response', ylabel='Channels',
           ylim=(0, max([r['direction_and_time_channels'] or 0 for r in info['rows']] + [1]) * 1.28))
    ax.legend(fontsize=6, frameon=False)
    fig.suptitle(f'{ctx.session.label} · direction-specific tuning', fontsize=10)
    fig.supxlabel('A: one unique preferred direction per significant channel; ties unassigned: '
                  f'{info["unassigned"]}. B/C dark: electrodes can repeat across directions. C light: A ∩ B.\n'
                  'Baseline-eligible trials (0° to 315°): ' + ', '.join(map(str, info['n_trials'])) +
                  '. Time tests: bin Bonferroni then joint BH across channel × direction.', fontsize=7)
    return fig


def draw_examples(ctx, groups, info):
    """Select exploratory examples without treating selection as inferential evidence."""
    choices = []
    # Look for two significant temporal directions with similar window means and separated peaks.
    candidates = []
    for j in range(len(groups['active'])):
        ds = np.flatnonzero(info['sig'][:, j])
        for a in ds:
            for b in ds[ds > a]:
                ma, mb = info['window_means'][[a, b], j]
                if not np.isfinite([ma, mb]).all():
                    continue
                similarity = abs(ma - mb) / max(abs(ma), abs(mb), 1)
                gap = abs(info['peak_time'][a, j] - info['peak_time'][b, j])
                if similarity <= .2 and gap >= .1:
                    strength = -np.log10(max(info['q'][a, j], info['q'][b, j], 1e-300))
                    candidates.append((gap * strength / (1 + similarity), j, int(a), int(b)))
    if candidates:
        _, j, a, b = max(candidates)
        choices.append(('Similar window means; different peaks', j, [a, b]))
    for title, mask in [('Temporal response; S05 not significant', ~groups['direction']),
                        ('Direction preference + temporal response', groups['direction'])]:
        js = np.flatnonzero(mask & info['sig'].any(axis=0))
        js = [j for j in js if all(j != choice[1] for choice in choices)]
        if js:
            j = min(js, key=lambda j: np.nanmin(info['q'][:, j]))
            ds = np.flatnonzero(info['sig'][:, j])
            ds = sorted(ds, key=lambda d: info['q'][d, j])[:2]
            choices.append((title, j, ds))
        else:
            choices.append((title, None, []))
    if not candidates:
        choices.insert(0, ('Similar window means; different peaks', None, []))
    fig = new_figure(13, 4.5)
    axes = fig.subplots(1, 3)
    data = []
    for ax, (title, j, ds) in zip(axes, choices):
        ax.set_title(title, fontsize=9)
        if j is None:
            style.text_panel(ax, ['No eligible example in this recording', 'No synthetic example substituted.'])
            continue
        electrode = int(ctx.session.electrode_ids[j])
        ax.set_title(f'{title}\nElectrode {electrode}', fontsize=9)
        for d in ds:
            mean, se = info['curves'][d, :, j], info['sem'][d, :, j]
            color = style.direction_color(d)
            ax.plot(info['times'], mean, color=color,
                    label=f'{d*45}°: q={info["q"][d,j]:.2g}, peak={info["peak_time"][d,j]:.2f}s')
            ax.fill_between(info['times'], mean - se, mean + se, color=color, alpha=.13)
            for t, m, s in zip(info['times'], mean, se):
                data.append(dict(example=title, electrode=electrode, direction_deg=int(d*45),
                                 time_s=float(t), delta_hz=float(m), sem_hz=float(s),
                                 window_mean_hz=float(info['window_means'][d,j]),
                                 temporal_q=float(info['q'][d,j])))
        ax.axhline(0, color=style.GREY, lw=.6)
        ax.set(xlabel='Time from S06 alignment / s', ylabel='Rate − paired baseline / Hz')
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle(f'{ctx.session.label} · exploratory temporal response examples', fontsize=10)
    fig.supxlabel('Raw post-event mean ± SEM; examples selected on these same data, not held-out evidence.\n'
                  'Similar means: within 20%; peak gap ≥100 ms (descriptive selection). '
                  'Different peaks do not establish direction × time interaction or decoding.', fontsize=7)
    return fig, data


def append_directional(ctx, result, groups):
    info = direction_statistics(ctx, groups)
    result.add_figure('Direction-specific tuning counts', draw_counts(ctx, info), method=METHOD,
                      data={'direction_counts': info['rows'], 'channel_direction_tests': info['channels']})
    result.tables['direction_specific_tuning_counts'] = info['rows']
    result.tables['direction_specific_tuning_channels'] = info['channels']
    from .paired import append_paired
    append_paired(ctx,result,groups,info)
    fig, data = draw_examples(ctx, groups, info)
    result.add_figure('Temporal response examples', fig, method=METHOD +
                      ' Exploratory selection on the same data; no inferential peak-latency comparison. '
                      'Similar window means: relative difference <=20%; absolute-delta peak gap >=0.1 s. '
                      'No qualifying example is explicitly shown as unavailable. Other examples selected '
                      'by smallest directional temporal q. SEM describes trial variability, not selection uncertainty.',
                      data={'selected_curves': data})
    result.tables['temporal_example_curves'] = data
    result.summary['direction_specific_tuning'] = info['rows']
    result.summary['preferred_direction_unassigned'] = info['unassigned']
