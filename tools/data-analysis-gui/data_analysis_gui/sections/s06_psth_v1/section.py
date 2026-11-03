"""S06 v1 — event-aligned PSTH, population heatmap and a temporal response test.

Rates are frame-weighted bin means (`Session.binned_rates`, NaN where a bin has
no frame). Smoothing (Gaussian, NaN-aware) is applied to trial-averaged curves
for display only; every test uses the raw bins.

Temporal response test, per channel: for every post-event bin a paired t-test of
bin rate vs the same trial's pre-event baseline (trial-paired); the channel p is
the minimum bin p, Bonferroni-corrected over the post-event bins, then BH across
active channels. Latency = first post-event bin that is Bonferroni-significant
with the same sign as the channel's strongest bin (two consecutive bins required
when available). This is a test of temporal modulation, not of direction tuning.
"""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from scipy import stats

from ...core import analysis as A
from ...core import style
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure

UP, DOWN = '#c0392b', '#2471a3'


def anchors(cohort, align):
    if align == 'target_acquired':
        return cohort.starts + cohort.array('acquire_time')
    return cohort.starts


def smooth(values, sigma_bins, axis):
    """NaN-aware Gaussian smoothing along `axis` (sigma in bins; 0 = none)."""
    if sigma_bins <= 0:
        return values
    from scipy.ndimage import gaussian_filter1d
    finite = np.isfinite(values)
    num = gaussian_filter1d(np.where(finite, values, 0.0), sigma_bins, axis=axis, mode='nearest')
    den = gaussian_filter1d(finite.astype(float), sigma_bins, axis=axis, mode='nearest')
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[den < 1e-3] = np.nan
    return out


def paired_t(delta):
    """Vectorised paired t-test over axis 0 ignoring NaN. Returns mean, p, n."""
    n = np.sum(np.isfinite(delta), 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.nanmean(delta, 0)
        sd = np.nanstd(delta, 0, ddof=1)
        t = mean / (sd / np.sqrt(n))
    ok = (n > 2) & (sd > 1e-12)
    p = np.ones(mean.shape)
    p[ok] = 2 * stats.t.sf(np.abs(t[ok]), n[ok] - 1)
    return np.where(np.isfinite(mean), mean, 0.0), p, n


def temporal_statistics(ctx):
    """Shared exact S06 test, cached across S06 and S07 for identical settings."""
    key = ('s06_temporal_statistics', id(ctx.cohort), tuple(sorted(ctx.params.items())))
    # Epoch arrays can be large: retain only the most recent recording/settings.
    for previous in list(ctx.cache):
        if isinstance(previous, tuple) and previous and previous[0] == key[0] and previous != key:
            del ctx.cache[previous]
    return ctx.cached(key, lambda: _temporal_statistics(ctx))


def _temporal_statistics(ctx):
    p, cohort, session = ctx.params, ctx.cohort, ctx.session
    if p['tmax'] - p['tmin'] < 2 * p['bin']:
        raise ValueError('时间范围至少需要两个 bin')
    if p['baseline_end'] <= p['baseline_start']:
        raise ValueError('基线终点必须大于起点')
    if cohort.n < 8:
        raise ValueError(f'保留的 trial 太少（{cohort.n}）')
    notes = []
    anchor = anchors(cohort, p['align'])
    keep = np.isfinite(anchor)
    if (~keep).any():
        notes.append(f'{int((~keep).sum())} 个 trial 缺少对齐事件，已剔除。')
    anchor, dirs = anchor[keep], cohort.directions[keep]
    ctx.progress('binning')
    binned, edges = session.binned_rates(anchor, p['tmin'], p['tmax'], p['bin'])
    centres = (edges[:-1] + edges[1:]) / 2
    base, nb = session.window_mean_rates(anchor + p['baseline_start'], anchor + p['baseline_end'])
    valid = nb > 0
    if valid.sum() < 8:
        raise ValueError('基线窗口内有神经帧的 trial 太少')
    if (~valid).any():
        notes.append(f'{int((~valid).sum())} 个 trial 基线窗口无神经帧，不参与时间响应检验。')
    sigma = p['smooth_sd'] / p['bin']
    mean_psth = np.nanmean(binned, 0)                    # bins x 512
    active = np.nanmean(binned.reshape(-1, binned.shape[2]), 0) > 0

    # ------------------------------------------------------------- test
    ctx.progress('temporal response test')
    post = np.flatnonzero(centres > 0)
    if not len(post):
        raise ValueError('结束时间需大于 0 才能做时间响应检验')
    delta = binned[valid][:, post, :] - base[valid][:, None, :]
    dmean, pbin, _ = paired_t(delta)                       # post x 512
    n_post = len(post)
    best = np.argmin(pbin, 0)
    cols = np.arange(pbin.shape[1])
    p_ch = np.minimum(1.0, pbin[best, cols] * n_post)
    sign = np.sign(dmean[best, cols])
    q = A.bh_fdr(p_ch, np.flatnonzero(active))
    sig = active & (q < p['alpha'])
    hit = (pbin < p['alpha'] / n_post) & (np.sign(dmean) == sign[None, :])
    latency = np.full(len(cols), np.nan)
    for j in np.flatnonzero(sig):
        h = hit[:, j]
        run2 = np.flatnonzero(h[:-1] & h[1:]) if n_post > 1 else np.zeros(0, int)
        first = run2[0] if len(run2) else (np.flatnonzero(h)[0] if h.any() else -1)
        if first >= 0:
            latency[j] = centres[post[first]]
    peak_idx = np.argmax(np.abs(dmean), 0)
    peak_delta = dmean[peak_idx, cols]
    peak_time = centres[post[peak_idx]]

    info = dict(binned=binned, centres=centres, edges=edges, dirs=dirs, active=active, sig=sig, sign=sign,
                q=q, p_ch=p_ch, latency=latency, pbin=pbin, dmean=dmean, post=post, mean_psth=mean_psth,
                sigma=sigma, base=base, valid=valid, peak_delta=peak_delta, peak_time=peak_time)
    info.update(anchor=anchor, notes=notes)
    return info


class Psth(Section):
    id = 's06'
    title = 'PSTH'
    description = '事件对齐 PSTH：群体通道×时间热图、代表通道八方向 PSTH、群体平均、逐通道时间响应检验与潜伏期'
    params = (
        Param('align', '对齐事件', 'choice', 'trial_start', choices=('trial_start', 'target_acquired')),
        Param('tmin', '起始时间 / s', 'float', -0.5),
        Param('tmax', '结束时间 / s', 'float', 2.0),
        Param('bin', 'PSTH bin / s', 'float', 0.05, minimum=0.02, maximum=0.5),
        Param('smooth_sd', '高斯平滑 SD / s（0=不平滑，仅用于显示）', 'float', 0.05, minimum=0, maximum=0.5),
        Param('baseline_start', '基线起点 / s（相对对齐事件）', 'float', -0.3),
        Param('baseline_end', '基线终点 / s', 'float', 0.0),
        Param('sort', '热图通道排序', 'choice', 'peak_latency', choices=('peak_latency', 'anova_F', 'electrode')),
        Param('zscore', '热图逐通道 z-score', 'bool', True),
        Param('top_n', '代表通道数', 'int', 8, minimum=1, maximum=32),
        Param('alpha', '显著性水平 α（BH q）', 'float', 0.05, minimum=1e-6, maximum=0.5),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)     # all-NaN bins (neural gaps)
            return self._run(ctx)

    def _run(self, ctx: AnalysisContext) -> SectionResult:
        p, cohort, session = ctx.params, ctx.cohort, ctx.session
        result = SectionResult()
        info = temporal_statistics(ctx)
        result.notes.extend(info['notes'])
        binned = info['binned']
        centres = info['centres']
        edges = info['edges']
        dirs = info['dirs']
        active = info['active']
        sig = info['sig']
        sign = info['sign']
        q = info['q']
        p_ch = info['p_ch']
        latency = info['latency']
        pbin = info['pbin']
        dmean = info['dmean']
        post = info['post']
        mean_psth = info['mean_psth']
        sigma = info['sigma']
        base = info['base']
        valid = info['valid']
        peak_delta = info['peak_delta']
        peak_time = info['peak_time']
        anchor = info['anchor']
        n_post = len(post)
        banner = (f'{session.label} · {len(anchor)} trials · aligned to {p["align"]} · bin {p["bin"] * 1000:.0f} ms'
                  f' · smoothing SD {p["smooth_sd"] * 1000:.0f} ms (display only) · baseline '
                  f'[{p["baseline_start"]:g}, {p["baseline_end"]:g}) s')
        F = None
        if p['sort'] == 'anova_F':
            x = cohort.window_rates
            ok = np.isfinite(x).all(1)
            st = (ctx.cached(('anova', id(cohort)), lambda: A.oneway_anova(x, cohort.directions)) if ok.all()
                  else ctx.cached(('anova_finite', id(cohort)), lambda: A.oneway_anova(x[ok], cohort.directions[ok])))
            F = st['F']
        ctx.progress('figures')
        ids = session.electrode_ids
        common = (
            f'Input: {len(anchor)} included trials aligned to {p["align"]} '
            + ('(T_TRIAL_START + first T_TARGET_ACQUIRED time)' if p['align'] == 'target_acquired' else '(T_TRIAL_START)')
            + f'. Rates: per trial, bin and channel, mean spike count over the neural frames whose timestamps fall in '
            f'the bin divided by the frame duration (Hz); bins of {p["bin"] * 1000:g} ms from {p["tmin"]:g} to '
            f'{p["tmax"]:g} s; a bin without frames is NaN and ignored in means. Trial-averaged curves are smoothed '
            f'for display with a NaN-aware Gaussian (SD {p["smooth_sd"] * 1000:g} ms; 0 = none); tests use raw bins. '
            f'Active channels = mean rate > 0 over all trials and bins ({int(active.sum())}). ')
        test_txt = (
            f'Temporal response test: baseline = per-trial mean rate in [{p["baseline_start"]:g}, '
            f'{p["baseline_end"]:g}) s around the event ({int(valid.sum())} trials with baseline frames). For each of '
            f'the {n_post} post-event bins (centre > 0) a two-sided paired t-test of (bin rate - baseline) across '
            f'trials; channel p = min over bins x {n_post} (Bonferroni within channel, capped at 1); q = '
            f'Benjamini-Hochberg over active channels; responsive = q < {p["alpha"]:g}; sign = sign of the mean '
            f'difference in the minimum-p bin. Latency = centre of the first bin with p < {p["alpha"]:g}/{n_post} and '
            f'the same sign, requiring two consecutive such bins when any exist. This tests temporal modulation, not '
            f'direction tuning. ')
        fig, hm_data, how = self.heatmap(info, ids, p, F, banner)
        result.add_figure(
            'Population PSTH heatmap', fig,
            method=common + test_txt + (
                'Heatmap: trial-averaged PSTH pooled over all directions, smoothed, one row per active channel, '
                + ('z-scored per channel over time bins' if p['zscore'] else 'in Hz') + f'; rows {how}'
                + ('(peak = bin of max |z| among post-event bins)' if p['sort'] == 'peak_latency' else '')
                + '. Linear diverging colour limit = 99th percentile of |z|. Left strip = test result.'),
            data=hm_data)
        order = [j for j in np.lexsort((-np.abs(peak_delta), p_ch)) if sig[j]][:p['top_n']]
        if not order:
            order = [j for j in np.lexsort((-np.abs(peak_delta), p_ch)) if active[j]][:p['top_n']]
            result.notes.append('没有通道通过时间响应检验，代表通道按未校正 p 值展示。')
        fig, grid_data = self.direction_grid(info, ids, order, p, banner)
        result.add_figure(
            'Per-direction PSTH of responsive channels', fig,
            method=common + test_txt + (
                f'Channels: up to {p["top_n"]} responsive channels ranked by channel p then by |peak delta| '
                '(peak delta = largest |mean (bin - baseline)| over post-event bins). Each panel: trial-averaged '
                'PSTH per target direction (smoothed); shaded = baseline window; dashed = latency.'),
            data=grid_data)
        fig, pop_data = self.population(info, p, banner)
        result.add_figure(
            'Population mean PSTH by direction', fig,
            method=common + test_txt + (
                'Per trial the rate is averaged over the channels of a group (all active; responsive up; responsive '
                'down), then mean +- SEM (SD / sqrt(n finite trials)) over trials of each direction, both smoothed.'),
            data=pop_data)
        t_post = centres[post]
        thr = pbin < p['alpha'] / n_post
        n_act = max(int(active.sum()), 1)
        act = np.flatnonzero(active)
        result.add_figure(
            'Temporal response test', self.test_figure(info, p, banner),
            method=common + test_txt + (
                'Panels: (a) number of active channels responsive up / down / not; (b) histogram of -log10 q; '
                f'(c) % of active channels whose bin p < {p["alpha"]:g}/{n_post} with mean above / below baseline, '
                'per post-event bin; (d) stacked histogram of latency of responsive channels by sign, dashed = median.'),
            data=dict(
                channels=dict(stored_column=act + 1, electrode=ids[act], p=p_ch[act], q=q[act], significant=sig[act],
                              sign=sign[act], latency_s=latency[act], peak_delta_hz=peak_delta[act],
                              peak_time_s=peak_time[act]),
                per_bin=dict(t_s=t_post, pct_above=100 * (thr & (dmean > 0))[:, active].sum(1) / n_act,
                             pct_below=100 * (thr & (dmean < 0))[:, active].sum(1) / n_act),
                counts=dict(category=['up', 'down', 'not_responsive'],
                            channels=[int((sig & (sign > 0)).sum()), int((sig & (sign < 0)).sum()),
                                      int((active & ~sig).sum())])))

        ids = session.electrode_ids
        result.tables['psth_channels'] = [
            dict(stored_column=j + 1, electrode=int(ids[j]), active=bool(active[j]), p=float(p_ch[j]),
                 q=float(q[j]), significant=bool(sig[j]),
                 direction=('up' if sign[j] > 0 else 'down') if sig[j] else '',
                 latency_s=round(float(latency[j]), 4) if np.isfinite(latency[j]) else '',
                 peak_delta_hz=round(float(peak_delta[j]), 3), peak_time_s=round(float(peak_time[j]), 4),
                 baseline_hz=round(float(np.nanmean(base[valid, j])), 3))
            for j in range(len(ids))]
        lat = latency[np.isfinite(latency)]
        result.summary.update(
            trials=int(len(anchor)), trials_with_baseline=int(valid.sum()), active_channels=int(active.sum()),
            responsive_channels=int(sig.sum()), responsive_up=int((sig & (sign > 0)).sum()),
            responsive_down=int((sig & (sign < 0)).sum()), post_bins_tested=int(n_post),
            median_latency_s=round(float(np.median(lat)), 3) if len(lat) else None,
            top_responsive_electrodes=[int(ids[j]) for j in order])
        from ...core.channel_atlas import append_psth_atlas
        append_psth_atlas(result, info, ids, p, banner, smooth)
        return result

    # ------------------------------------------------------------------ figures
    @staticmethod
    def _events(ax, p, criteria=None):
        ax.axvline(0, color=style.INK, lw=0.6)
        ax.axvspan(p['baseline_start'], p['baseline_end'], color='#e8e6df', zorder=-2, lw=0)

    def heatmap(self, info, ids, p, F, banner):
        active, sig, sign = info['active'], info['sig'], info['sign']
        m = smooth(info['mean_psth'], info['sigma'], axis=0)        # bins x 512
        cols = np.flatnonzero(active)
        post = info['post']
        if p['zscore']:
            mu, sd = np.nanmean(m, 0), np.nanstd(m, 0)
            m = (m - mu) / np.where(sd > 1e-12, sd, 1.0)
        if p['sort'] == 'anova_F' and F is not None:
            cols = cols[np.argsort(-F[cols])]
            how = 'sorted by window ANOVA F (top = most direction-tuned)'
        elif p['sort'] == 'electrode':
            cols = cols[np.argsort(ids[cols])]
            how = 'sorted by electrode id'
        else:
            mm = m if p['zscore'] else m - np.nanmean(m[info['centres'] <= 0], 0)
            peak = np.argmax(np.nan_to_num(np.abs(mm[post]), nan=-1), 0)
            cols = cols[np.argsort(peak[cols], kind='stable')]
            how = 'sorted by post-event peak latency of |response|'
        fig = new_figure(12, 8.5)
        grid = fig.add_gridspec(1, 2, width_ratios=[0.04, 1], wspace=0.01)
        side = fig.add_subplot(grid[0, 0])
        ax = fig.add_subplot(grid[0, 1], sharey=side)
        code = np.where(sig[cols], np.where(sign[cols] > 0, 2, 0), 1)[:, None]
        side.imshow(code, aspect='auto', cmap=ListedColormap([DOWN, '#f2f2f2', UP]), vmin=0, vmax=2,
                    interpolation='nearest')
        side.set_xticks([])
        side.set_ylabel(f'active channels (n={len(cols)}) · {how}', fontsize=7)
        side.set_title('test', fontsize=6)
        data = m[:, cols].T
        e = info['edges']
        if p['zscore']:
            lim = np.nanpercentile(np.abs(data), 99) or 1
            im = ax.imshow(data, aspect='auto', cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim),
                           extent=[e[0], e[-1], len(cols) - 0.5, -0.5], interpolation='nearest')
            label = 'z-scored rate (per channel, over time)'
        else:
            im = ax.imshow(data, aspect='auto', cmap='magma', vmax=np.nanpercentile(data, 99),
                           extent=[e[0], e[-1], len(cols) - 0.5, -0.5], interpolation='nearest')
            label = 'rate / Hz'
        ax.axvline(0, color='black', lw=0.8)
        ax.tick_params(labelleft=False)
        ax.set_xlabel(f't from {p["align"]} / s')
        ax.set_title('Trial-averaged PSTH, all directions pooled · left strip: red = up, blue = down, '
                     'grey = not responsive', fontsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01)
        cb.set_label(label, fontsize=7)
        fig.suptitle(banner, fontsize=8)
        rows = []
        for r, j in enumerate(cols):
            row = dict(row=r + 1, electrode=int(ids[j]), stored_column=int(j) + 1,
                       test=('up' if sign[j] > 0 else 'down') if sig[j] else 'not_responsive')
            row.update({f't_{t:+.3f}s': float(v) for t, v in zip(info['centres'], m[:, j])})
            rows.append(row)
        return fig, dict(heatmap=rows), how

    def direction_grid(self, info, ids, order, p, banner):
        n = max(len(order), 1)
        n_cols = min(4, n)
        n_rows = int(np.ceil(n / n_cols))
        fig = new_figure(3.3 * n_cols + 0.5, 2.6 * n_rows + 0.9)
        axes = fig.subplots(n_rows, n_cols, squeeze=False).ravel()
        c, b, dirs = info['centres'], info['binned'], info['dirs']
        rows = []
        for ax, j in zip(axes, order):
            for d in range(8):
                sel = dirs == d
                if sel.any():
                    ax.plot(c, smooth(np.nanmean(b[sel, :, j], 0), info['sigma'], 0), color=style.direction_color(d),
                            lw=1, label=f'{d * 45}°')
            self._events(ax, p)
            lat = info['latency'][j]
            if np.isfinite(lat):
                ax.axvline(lat, color=UP if info['sign'][j] > 0 else DOWN, ls='--', lw=0.8)
            ax.set_title(f'E{int(ids[j])} · q={info["q"][j]:.1e} · peak Δ {info["peak_delta"][j]:+.1f} Hz', fontsize=7)
            ax.tick_params(labelsize=6)
            for d in range(8):
                sel = dirs == d
                if sel.any():
                    raw = np.nanmean(b[sel, :, j], 0)
                    shown = smooth(raw, info['sigma'], 0)
                    rows.extend(dict(electrode=int(ids[j]), stored_column=int(j) + 1, direction_deg=d * 45,
                                     n_trials=int(sel.sum()), t_s=float(t), rate_hz_raw=float(r_),
                                     rate_hz_plotted=float(v), latency_s=float(lat))
                                for t, r_, v in zip(c, raw, shown))
        for ax in axes[len(order):]:
            ax.axis('off')
        axes[0].legend(fontsize=5.5, ncol=2, frameon=False, loc='upper right', handlelength=1)
        fig.supxlabel(f't from {p["align"]} / s  (shaded = baseline, dashed = latency)', fontsize=8)
        fig.supylabel('rate / Hz', fontsize=8)
        fig.suptitle(banner + f' · top {len(order)} responsive channels', fontsize=8)
        return fig, dict(psth=rows)

    def population(self, info, p, banner):
        b, c, dirs, active, sig, sign = (info['binned'], info['centres'], info['dirs'], info['active'],
                                         info['sig'], info['sign'])
        groups = [('All active channels', active), ('Responsive · up', sig & (sign > 0)),
                  ('Responsive · down', sig & (sign < 0))]
        fig = new_figure(13, 4.6)
        axes = fig.subplots(1, 3)
        rows = []
        for ax, (title, mask) in zip(axes, groups):
            if not mask.any():
                ax.text(0.5, 0.5, 'no channel', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title} (n=0)', fontsize=8)
                continue
            pop = np.nanmean(b[:, :, mask], 2)                  # trials x bins
            for d in range(8):
                sel = dirs == d
                if not sel.any():
                    continue
                y = pop[sel]
                mu = smooth(np.nanmean(y, 0), info['sigma'], 0)
                se = smooth(np.nanstd(y, 0) / np.sqrt(np.maximum(np.isfinite(y).sum(0), 1)), info['sigma'], 0)
                ax.fill_between(c, mu - se, mu + se, color=style.direction_color(d), alpha=0.18, lw=0)
                ax.plot(c, mu, color=style.direction_color(d), lw=1.1, label=f'{d * 45}°')
                rows.extend(dict(group=title.replace(' · ', ' '), n_channels=int(mask.sum()), direction_deg=d * 45,
                                 n_trials=int(sel.sum()), t_s=float(t), mean_hz=float(a), sem_hz=float(e_))
                            for t, a, e_ in zip(c, mu, se))
            self._events(ax, p)
            ax.set_title(f'{title} (n={int(mask.sum())})', fontsize=8)
            ax.set_xlabel(f't from {p["align"]} / s')
            ax.set_ylabel('mean rate across channels / Hz')
        axes[0].legend(fontsize=6, ncol=2, frameon=False)
        fig.suptitle(banner + ' · mean ± SEM over trials', fontsize=8)
        return fig, dict(population=rows)

    def test_figure(self, info, p, banner):
        active, sig, sign, q = info['active'], info['sig'], info['sign'], info['q']
        alpha = p['alpha']
        fig = new_figure(12, 7.5)
        grid = fig.add_gridspec(2, 2)
        ax = fig.add_subplot(grid[0, 0])
        vals = [int((sig & (sign > 0)).sum()), int((sig & (sign < 0)).sum()), int((active & ~sig).sum())]
        ax.bar(range(3), vals, color=[UP, DOWN, style.GREY])
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha='center', va='bottom')
        ax.set_xticks(range(3))
        ax.set_xticklabels(['up', 'down', 'not responsive'])
        ax.set_ylabel('active channels')
        ax.set_title(f'Responsive channels: {int(sig.sum())}/{int(active.sum())} (BH q < {alpha:g})')

        ax = fig.add_subplot(grid[0, 1])
        ax.hist(-np.log10(np.clip(q[active], 1e-300, 1)), bins=40, color='#30638e')
        ax.axvline(-np.log10(alpha), color='#c76d4b', ls='--', lw=0.8, label=f'q = {alpha:g}')
        ax.set_yscale('log')
        ax.set_xlabel('−log10 q (min-bin Bonferroni, then BH)')
        ax.set_ylabel('channels (log)')
        ax.set_title('Channel-level significance')
        ax.legend(frameon=False)

        ax = fig.add_subplot(grid[1, 0])
        t = info['centres'][info['post']]
        thr = (info['pbin'] < alpha / len(t))
        n_act = max(int(active.sum()), 1)
        ax.plot(t, 100 * (thr & (info['dmean'] > 0))[:, active].sum(1) / n_act, color=UP, label='above baseline')
        ax.plot(t, 100 * (thr & (info['dmean'] < 0))[:, active].sum(1) / n_act, color=DOWN, label='below baseline')
        ax.set_xlabel(f't from {p["align"]} / s')
        ax.set_ylabel('% active channels')
        ax.set_title(f'Per-bin paired t-test vs baseline (p < {alpha:g}/{len(t)})')
        ax.legend(frameon=False)

        ax = fig.add_subplot(grid[1, 1])
        lat = info['latency']
        bins = np.r_[info['edges'][info['edges'] >= 0]]
        up, down = sig & (sign > 0) & np.isfinite(lat), sig & (sign < 0) & np.isfinite(lat)
        ax.hist([lat[up], lat[down]], bins=bins, color=[UP, DOWN], stacked=True, label=['up', 'down'])
        finite = lat[np.isfinite(lat)]
        if len(finite):
            ax.axvline(np.median(finite), color=style.INK, ls='--', lw=0.8, label=f'median {np.median(finite):.2f} s')
        ax.set_xlabel(f'response latency from {p["align"]} / s (bin centre)')
        ax.set_ylabel('responsive channels')
        ax.set_title('Response latency')
        ax.legend(frameon=False)
        fig.suptitle(banner, fontsize=8)
        return fig


SECTION = Psth()
