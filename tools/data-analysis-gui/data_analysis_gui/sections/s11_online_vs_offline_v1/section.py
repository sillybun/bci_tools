"""S11 — recorded online velocity compared with held-out Ridge and Vel-KF."""
from __future__ import annotations

import numpy as np
from matplotlib.colors import LogNorm

from ... import decoders as D
from ...core import analysis as A
from ...core import style
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure

COLORS = dict(style.MODEL_COLORS, **D.COLORS)
BASE_NAMES = {'gt': 'GT (cursor)', 'online': 'Online decoder'}


def name_of(key):
    return BASE_NAMES.get(key) or 'Offline ' + D.label(key)


def spec_of(p) -> D.BinSpec:
    return D.BinSpec(offset=float(p['offset']), bin_s=float(p['bin']),
                     n_bins=int(p['n_bins']) if p['until'] == 'window' else None, until=p['until'], sqrt=False)


def bin_online(session, vb, bin_s, min_samples=2):
    dt, dv = session.decoder_times, session.decoder_velocity
    lo = np.searchsorted(dt, vb.time - bin_s, side='left')
    hi = np.searchsorted(dt, vb.time, side='left')
    cs = np.vstack([np.zeros((1, dv.shape[1])), np.cumsum(dv, 0)])
    n = hi - lo
    with np.errstate(invalid='ignore', divide='ignore'):
        out = (cs[hi] - cs[lo]) / n[:, None]
    out[n < min_samples] = np.nan
    return out[:, :2], lo, hi


def scores(y, p):
    ok = np.isfinite(y).all(1) & np.isfinite(p).all(1)
    y, p = y[ok], p[ok]
    if len(y) < 3:
        return dict(n=int(len(y)), r2_x=np.nan, r2_y=np.nan, r2_xy=np.nan, rmse=np.nan, r_x=np.nan, r_y=np.nan)
    r2 = A.r2_score(y, p)
    r = A.pearson(y, p)
    return dict(n=int(len(y)), r2_x=float(r2[0]), r2_y=float(r2[1]),
                r2_xy=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean(0)) ** 2)),
                rmse=float(np.sqrt(np.mean(np.sum((y - p) ** 2, 1)))), r_x=float(r[0]), r_y=float(r[1]))


class OnlineVsOffline(Section):
    id = 's11'
    title = '在线闭环decoder结果复现与离线结果对比'
    description = '同一 trial bin 上比较 GT、在线输出和 Ridge / Vel-KF 留出预测，并核查已保存的在线 KF 权重回放'
    requires = ('online',)
    params = (
        Param('online_model_dir', '在线模型目录（空=自动查找）', 'text', '',
              help='优先使用本次记录同级的 refit，也可指定 refit 或它所在的 run 目录。'),
        Param('initial_model_path', '初始权重文件或目录（空=自动查找）', 'file', '',
              help='优先使用 refit 同级 initial_weights 下的 INITIAL_position_velocity_kf.pkl。'),
        Param('saved_position_feedback', '保存权重回放：位置反馈', 'choice', 'recorded', choices=('recorded','decoded'),
              help='recorded：逐帧使用 XDF 位置；decoded：仅 trial 开始用位置初始化，之后由 KF 自行估计'),
        Param('feedback_max_age', '保存权重回放：位置最大陈旧时间 / s', 'float', .25, minimum=.001, maximum=2),
        Param('bin', 'bin 宽 / s', 'float', 0.1, minimum=0.02, maximum=0.5),
        Param('offset', '首个 bin 相对 T_TRIAL_START / s', 'float', 0.2, minimum=0.0),
        Param('n_bins', '每 trial bin 数（until=window）', 'int', 10, minimum=1, maximum=200),
        Param('until', 'bin 截止', 'choice', 'window', choices=('window', 'end', 'acquire')),
        Param('lam_ridge', '离线 Ridge λ', 'float', 1.0, minimum=1e-8),
        Param('lam_kf', '离线 KF λ', 'float', 1e-4, minimum=0.0),
        Param('q_shrinkage', '离线 KF 观测协方差收缩 α', 'float', 0.5, minimum=0.0, maximum=1.0),
        Param('zoom_start', '时间轴放大起点 / s（相对首个 trial）', 'float', 60.0, minimum=0.0),
        Param('zoom_len', '时间轴放大长度 / s', 'float', 60.0, minimum=2.0),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        p, session = ctx.params, ctx.session
        names = ('ridge', 'velocity_kf')
        result = SectionResult()
        ctx.progress('velocity bins')
        spec = spec_of(p)
        vb = D.build_data(ctx.cohort, spec, cache=ctx.cached)
        if vb.n < 50 or len(np.unique(vb.block)) < 4:
            raise ValueError('有效速度 bin 太少（需 ≥50 个 bin、≥4 个 block）')
        online, _, _ = bin_online(session, vb, p['bin'])
        ctx.progress('offline decoders: ' + ', '.join(names))
        cv = D.section_cv(ctx, ctx.cohort, names, spec, p, data=vb)
        series = dict(online=online, **{m: cv['results'][m]['pred'] for m in names})
        ctx.check_cancel()
        gt = vb.velocity
        comparisons = [(f'{name_of(k)} vs GT', k, 'gt') for k in series]
        comparisons += [(f'{name_of(m)} vs online', m, 'online') for m in names]
        ref = dict(series, gt=gt)
        table = []
        for label, a, b in comparisons:
            table.append(dict(comparison=label, prediction=a, reference=b, **scores(ref[b], ref[a])))
        ctx.progress('figures')
        base = self.base_method(spec, names, p, vb) + ' ' + D.cohort_rule_text(ctx.cohort, spec)
        fig, tl_data = self.timeline(session, ctx.cohort, vb, gt, series, names, p)
        result.add_figure('Velocity timeline (zoom)', fig,
                          method=base + f' Timeline: bins whose right edge lies in the zoom window ({p["zoom_start"]:g}'
                          f'–{p["zoom_start"] + p["zoom_len"]:g} s after the first included trial start; falls back '
                          'to the first bins if empty), one line per trial (lines break between trials, shaded = trial '
                          'start–end).', data=tl_data)
        fig, sc_data = self.scatter(session, gt, series, names)
        result.add_figure('Scatter: online and offline', fig,
                          method=base + ' Each column: log-density 2-D histogram (60×60 bins, ±1.1×99.5th percentile of '
                          '|GT| and |online|) of prediction (y) vs reference (x) per axis on bins where both are finite; '
                          'R² = 1 − SSE/SST of the prediction against the reference, r = Pearson.', data=sc_data)
        result.add_figure('Metrics summary', self.metrics(session, vb, table, p),
                          method=base + ' Left: for each comparison, R² (x,y pooled: 1 − ΣSSE/ΣSST around the '
                          'reference mean) and mean Pearson r over x/y, on bins where prediction and reference are both '
                          'finite (n in the table). Middle: vector RMSE sqrt(mean ‖v̂ − v‖²) vs GT.',
                          data=dict(comparisons=table))
        cmp_fig, cmp_data = self.comparison(session, ctx.cohort, vb, gt, series, names, p)
        result.add_figure('Decoder comparison', cmp_fig,
                          method=base + ' Decoder comparison on ONE common set of bins (GT, online and every offline '
                          'decoder all finite; n in the table): (a) R² (x,y pooled) of each series vs GT and of each '
                          'offline decoder vs the online decoder output; (b) vector RMSE sqrt(mean ‖v̂ − v_GT‖²); (c) '
                          'mean Pearson r over x/y vs GT and vs online; (d, e) time strip starting at the first bin at or '
                          'after the zoom start, covering '
                          f'{min(float(p["zoom_len"]), 20.0):g} s; GT, online and every offline '
                          'decoder overlaid (lines break between trials). The online decoder drove the cursor, so its '
                          'agreement with GT is descriptive, not held-out.', data=cmp_data)
        result.tables['decoder_comparison'] = cmp_data['common_bin_scores']
        result.tables['comparisons'] = table
        result.tables['decoders'] = [dict(decoder=m, source=D.source_text(m), cv_seconds=cv['results'][m]['seconds'])
                                     for m in names]
        result.summary.update(recording=session.label, decoders=list(names), n_bins=int(vb.n),
                              n_bins_with_online=int(np.isfinite(online).all(1).sum()),
                              decoder_samples=int(len(session.decoder_times)),
                              neural_frames=int(len(session.neural_times)))
        for row in table:
            result.summary[f'{row["prediction"]}_vs_{row["reference"]}_r2_xy'] = row['r2_xy']
            result.summary[f'{row["prediction"]}_vs_{row["reference"]}_r'] = (row['r_x'], row['r_y'])
        result.notes.append('在线 decoder 处于闭环控制中（cursor 由它驱动），其与 GT 的拟合只是描述性的，'
                            '不是独立的 held-out 评估。')
        from .model_replay import append_replay
        append_replay(ctx, result)
        return result

    # ------------------------------------------------------------------ method text
    @staticmethod
    def base_method(spec, names, p, vb):
        text = (f'Inputs: included trials; {spec.bin_s * 1000:g} ms bins from T_TRIAL_START+{spec.offset:g} s '
                f'({spec.until}' + (f', {spec.n_bins} bins/trial' if spec.n_bins else '') + '). GT = cursor '
                'displacement velocity over the bin (interpolated active-cursor positions). Online = mean of '
                'cursortasks_decoder_velocity samples with receive time in [left, right) of the bin (≥2 samples, else '
                'NaN); the online decoder drove the cursor (closed loop), so online-vs-GT is descriptive. Offline = '
                'held-out predictions: 5 chronological block folds, ±1 block purge, features = trailing '
                f'{vb.extra["neural_window"] * 1000:g} ms mean rate (Hz) of all transmitted electrodes, normalised on '
                'training '
                'rows only. Offline decoders: ' + D.describe(names, p, spec))
        return text

    # ------------------------------------------------------------------ figures
    def timeline(self, session, cohort, vb, gt, series, names, p):
        t0 = float(cohort.starts.min())
        lo, hi = t0 + p['zoom_start'], t0 + p['zoom_start'] + p['zoom_len']
        if not np.any((vb.time >= lo) & (vb.time <= hi)):
            lo, hi = float(vb.time.min()) - 1, float(vb.time.min()) - 1 + p['zoom_len']
        fig = new_figure(14, 5.4)
        axes = fig.subplots(2, 1, sharex=True)
        sel = np.flatnonzero((vb.time >= lo) & (vb.time <= hi))
        trials = np.unique(vb.trial[sel])
        for axis in range(2):
            ax = axes[axis]
            for k, data in [('gt', gt)] + [(k, series[k]) for k in ('online',) + tuple(names)]:
                for j, t in enumerate(trials):
                    r = sel[vb.trial[sel] == t]
                    ax.plot(vb.time[r] - t0, data[r, axis], color=COLORS[k], lw=1.6 if k == 'gt' else 1.0,
                            label=name_of(k) if j == 0 else None, marker='.', ms=2)
            for t in trials:
                ax.axvspan(cohort.starts[t] - t0, min(cohort.ends[t], hi) - t0, color='#f1efe8', zorder=-1)
            ax.axhline(0, color=style.GREY, lw=0.4)
            ax.set_ylabel(f'v{"xy"[axis]} / units·s⁻¹')
            ax.set_title(f'Matched {p["bin"] * 1000:.0f} ms trial bins · v{"xy"[axis]} (lines break between trials; '
                         'shaded = trial)', fontsize=8, loc='left')
        axes[0].legend(frameon=False, ncol=min(6, 2 + len(names)), loc='lower right', bbox_to_anchor=(1, 1),
                       fontsize=6)
        bins_table = dict(time_s=vb.time[sel] - t0, trial=vb.trial[sel], gt_vx=gt[sel, 0], gt_vy=gt[sel, 1])
        for k in ('online',) + tuple(names):
            bins_table[f'{k}_vx'], bins_table[f'{k}_vy'] = series[k][sel, 0], series[k][sel, 1]
        data = dict(zoom_bins=bins_table)
        axes[-1].set_xlabel(f'time / s from first included trial start (zoom {lo - t0:.0f}–{hi - t0:.0f} s)')
        axes[-1].set_xlim(lo - t0, hi - t0)
        fig.suptitle(f'{session.label} · GT vs online closed-loop decoder vs offline held-out decoders', fontsize=10)
        return fig, data

    def scatter(self, session, gt, series, names):
        pairs = [('online', 'gt')] + [(m, 'gt') for m in names] + [(m, 'online') for m in names]
        fig = new_figure(2.3 * len(pairs) + 0.5, 5.4)
        axes = fig.subplots(2, len(pairs))
        ref = dict(series, gt=gt)
        lim = np.nanpercentile(np.abs(np.r_[gt.ravel(), series['online'].ravel()]), 99.5) * 1.1
        panel_rows = []
        for j, (a, b) in enumerate(pairs):
            xs, ys = ref[b], ref[a]
            xname, yname, color = name_of(b), name_of(a), COLORS[a]
            ok = np.isfinite(xs).all(1) & np.isfinite(ys).all(1)
            s = scores(xs[ok], ys[ok])
            panel_rows.append(dict(prediction=a, reference=b, **s))
            for axis in range(2):
                ax = axes[axis, j]
                ax.hist2d(xs[ok, axis], ys[ok, axis], bins=60, range=[[-lim, lim], [-lim, lim]], cmap='Greys',
                          cmin=1, norm=LogNorm())
                ax.plot([-lim, lim], [-lim, lim], color=color, lw=0.8)
                ax.set_aspect('equal')
                ax.set_title(f'v{"xy"[axis]} · R² {s["r2_" + "xy"[axis]]:.2f} · r {s["r_" + "xy"[axis]]:.2f}',
                             fontsize=7)
                ax.set_xlabel(xname, fontsize=7)
                ax.set_ylabel(yname, fontsize=7)
                ax.tick_params(labelsize=6)
        fig.suptitle(f'{session.label} · log-density scatter (matched trial bins)', fontsize=9)
        bins_table = dict(gt_vx=gt[:, 0], gt_vy=gt[:, 1])
        for k in ('online',) + tuple(names):
            bins_table[f'{k}_vx'], bins_table[f'{k}_vy'] = series[k][:, 0], series[k][:, 1]
        return fig, dict(panels=panel_rows, matched_bins=bins_table)

    def metrics(self, session, vb, table, p):
        fig = new_figure(14, 7.6)
        grid = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1.1])
        ax = fig.add_subplot(grid[:, 0])
        labels = [r['comparison'] for r in table]
        ys = np.arange(len(table))
        for off, key, col in ((-0.2, 'r2_xy', '#2176ae'), (0.2, 'r_mean', '#e07b39')):
            vals = [r['r2_xy'] if key == 'r2_xy' else np.nanmean([r['r_x'], r['r_y']]) for r in table]
            ax.barh(ys + off, vals, 0.38, color=col, label='R² (x,y pooled)' if key == 'r2_xy' else 'mean r')
            for y, v in zip(ys, vals):
                if np.isfinite(v):
                    ax.text(max(v, 0), y + off, f' {v:.2f}', va='center', fontsize=6)
        ax.set_yticks(ys, labels, fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color=style.GREY, lw=0.5)
        ax.set_xlim(min(0, np.nanmin([r['r2_xy'] for r in table]) - 0.05), 1.12)
        ax.set_title('Agreement on matched trial bins (prediction vs reference)', fontsize=8)
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.04), ncol=2)

        ax = fig.add_subplot(grid[0, 1])
        vs_gt = [r for r in table if r['reference'] == 'gt']
        rm = [r['rmse'] for r in vs_gt]
        ax.bar(range(len(rm)), rm, color=[COLORS[r['prediction']] for r in vs_gt])
        ax.set_xticks(range(len(rm)), [r['comparison'].replace(' vs GT', '').replace('Offline ', '')
                                       for r in vs_gt], fontsize=6, rotation=25, ha='right')
        ax.set_ylabel('RMSE ‖v‖ vs GT / units·s⁻¹')
        ax.set_title('Error vs GT (matched bins)', fontsize=8)

        ax = fig.add_subplot(grid[:, 2])
        lines = [session.label, '',
                 f'bins           {vb.n} ({p["bin"] * 1000:.0f} ms, +{p["offset"]:g} s, {p["until"]})',
                 f'decoder samples {len(session.decoder_times)}',
                 f'neural frames  {len(session.neural_times)}', '']
        for r in table:
            lines.append(f'{r["prediction"][:12]:>12} vs {r["reference"]:<6} R² {r["r2_xy"]:.3f}  '
                         f'r {r["r_x"]:.2f}/{r["r_y"]:.2f}')
        lines += ['', 'The online decoder drives the cursor', '(closed loop): its fit to GT is',
                  'descriptive, not a held-out evaluation.',
                  'Offline models: Ridge and Vel-KF.']
        style.text_panel(ax, lines, fontsize=6.8)
        fig.suptitle(f'{session.label} · online vs offline agreement', fontsize=10)
        return fig


    def comparison(self, session, cohort, vb, gt, series, names, p):
        keys = ('online',) + tuple(names)
        common = np.isfinite(gt).all(1)
        for k in keys:
            common &= np.isfinite(series[k]).all(1)
        rows = []
        for k in keys:
            vs_gt = scores(gt[common], series[k][common])
            vs_on = scores(series['online'][common], series[k][common]) if k != 'online' else None
            rows.append(dict(series=k, label=name_of(k), n=vs_gt['n'], r2_vs_gt=vs_gt['r2_xy'], rmse_vs_gt=vs_gt['rmse'],
                             r_vs_gt=float(np.nanmean([vs_gt['r_x'], vs_gt['r_y']])),
                             r2_vs_online=np.nan if vs_on is None else vs_on['r2_xy'],
                             r_vs_online=np.nan if vs_on is None else float(np.nanmean([vs_on['r_x'], vs_on['r_y']]))))
        fig = new_figure(15, 9.5)
        grid = fig.add_gridspec(3, 3, height_ratios=[1.2, 0.8, 0.8])
        xs = np.arange(len(keys))
        colors = [COLORS[k] for k in keys]
        short = [name_of(k).replace('Offline ', '') for k in keys]
        for col, (title, a_key, b_key, ylabel) in enumerate((
                ('(a) R² (x,y pooled)', 'r2_vs_gt', 'r2_vs_online', 'R²'),
                ('(b) RMSE vs GT', 'rmse_vs_gt', None, 'RMSE ‖v‖ / units·s⁻¹'),
                ('(c) mean Pearson r', 'r_vs_gt', 'r_vs_online', 'r'))):
            ax = fig.add_subplot(grid[0, col])
            w = 0.38 if b_key else 0.7
            va = [r[a_key] for r in rows]
            ax.bar(xs - (w / 2 if b_key else 0), va, w, color=colors, label='vs GT')
            for x, v in zip(xs - (w / 2 if b_key else 0), va):
                ax.text(x, v, f'{v:.2f}' if v < 10 else f'{v:.0f}', ha='center', va='bottom', fontsize=5.5)
            if b_key:
                vb_ = [r[b_key] for r in rows]
                ax.bar(xs + w / 2, vb_, w, color=colors, alpha=0.45, hatch='//', edgecolor='white',
                       label='vs online decoder')
                for x, v in zip(xs + w / 2, vb_):
                    if np.isfinite(v):
                        ax.text(x, v, f'{v:.2f}', ha='center', va='bottom', fontsize=5.5)
                ax.set_ylim(min(0, np.nanmin(va + vb_) - 0.05), 1.08)
                ax.legend(frameon=False, fontsize=6, loc='lower right')
            ax.set_xticks(xs, short, rotation=25, ha='right', fontsize=6.5)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=8)
        t0 = float(cohort.starts.min())
        # Start at the first bin at/after the zoom start (the window may fall in a gap between blocks).
        order = np.sort(vb.time)
        i0 = min(int(np.searchsorted(order, t0 + p['zoom_start'])), len(order) - 1)
        lo = float(order[i0]) - float(p['bin'])
        hi = lo + min(float(p['zoom_len']), 20.0)
        sel = np.flatnonzero((vb.time >= lo) & (vb.time <= hi))
        strip = dict(time_s=vb.time[sel] - t0, trial=vb.trial[sel], gt_vx=gt[sel, 0], gt_vy=gt[sel, 1])
        for axis in range(2):
            ax = fig.add_subplot(grid[1 + axis, :])
            for k in ('gt',) + keys:
                data = gt if k == 'gt' else series[k]
                for j, t in enumerate(np.unique(vb.trial[sel])):
                    r = sel[vb.trial[sel] == t]
                    ax.plot(vb.time[r] - t0, data[r, axis], color=COLORS[k], lw=1.8 if k == 'gt' else 1.0,
                            label=name_of(k) if j == 0 else None)
                if k != 'gt':
                    strip[f'{k}_v{"xy"[axis]}'] = data[sel, axis]
            ax.axhline(0, color=style.GREY, lw=0.4)
            ax.set_ylabel(f'v{"xy"[axis]} / units·s⁻¹')
            ax.set_title(f'({"de"[axis]}) v{"xy"[axis]} · all series overlaid', fontsize=8, loc='left')
            if axis == 0:
                ax.legend(frameon=False, ncol=len(keys) + 1, fontsize=6, loc='lower right', bbox_to_anchor=(1, 1))
            else:
                ax.set_xlabel('time / s from first included trial start')
        note = '' if len(names) > 1 else ' · only one offline decoder selected'
        fig.suptitle(f'{session.label} · online decoder vs offline decoders on {int(common.sum())} common bins{note}',
                     fontsize=10)
        return fig, dict(common_bin_scores=rows, time_strip=strip)


SECTION = OnlineVsOffline()
