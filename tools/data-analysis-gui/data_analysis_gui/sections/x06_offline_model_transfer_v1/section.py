"""X06 v1 — offline model transfer across days.

A reference recording trains Ridge and Vel-KF velocity decoders on its included
trial bins, plus a one-hot ridge direction classifier. The
models are applied unchanged to each test recording, using the electrodes both
recordings transmitted. An explicit reference is tested against all other selected
recordings. The first and each modes evaluate forward in recording order. Recalibration
options that do not refit any weight:
  fixed          features standardized with the reference statistics
  re-centred     test-day feature means removed (reference SD kept)
  re-standardized test-day mean and SD
Ceiling: the test recording's own within-day block CV restricted to the same
common electrodes (so the gap is due to time, not to the electrode subset).
"""
from __future__ import annotations

import numpy as np

from ... import config as C
from ... import decoders as D
from ...core import analysis as A
from ...core import crossday as X
from ...core import style
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure

MODES = X.STANDARDIZE
MODE_COLORS = {'train': '#d1495b', 'recentre': '#edae49', 'test_own': '#2176ae'}
MODE_SHORT = {'train': 'fixed', 'recentre': 're-centred', 'test_own': 're-standardized'}


def metrics_for(names):
    return (('direction', 'acc', 'direction balanced accuracy'),) + tuple(
        (m, 'r2', f'{D.label(m)} velocity R²') for m in names)


def spec_of(ctx, p) -> D.BinSpec:
    # Velocity-bin quality thresholds and policy come from the cohort TrialFilter (None fields).
    return D.BinSpec(offset=float(p['offset']), bin_s=float(p['bin']), n_bins=int(p['n_bins']), until='window')


def ceiling(ctx, j, cj, dir_j, names, spec, p):
    """Within-day CV of recording j restricted to stored columns cj."""
    key = ('xday_ceiling_dir', id(ctx.cohorts[j]), tuple(map(int, cj)), p['lam'])

    def direction():
        scores, _, _, tested = A.ridge_classifier_cv(dir_j['x'][:, cj], dir_j['y'], dir_j['blocks'], p['lam'])
        return dict(acc=A.balanced_accuracy(dir_j['y'][tested], scores.argmax(1)[tested]))
    out = dict(direction=ctx.cached(key, direction))
    cv = D.section_cv(ctx, ctx.cohorts[j], names, spec, p, columns=cj)
    for m in names:
        pr = cv['results'][m]['pred']
        t = np.isfinite(pr).all(1)
        out[m] = X.velocity_scores(cv['data'].velocity[t], pr[t])
    return out


def online_check(session, vb):
    """Online decoder output vs cursor-displacement velocity at the bin right edges."""
    t, v = session.decoder_times, np.asarray(session.decoder_velocity, float)
    if v.ndim != 2 or v.shape[1] < 2 or len(t) < 10:
        return None
    order = np.argsort(t)
    t, v = t[order], v[order, :2]
    inside = (vb.time >= t[0]) & (vb.time <= t[-1])
    if inside.sum() < 10:
        return None
    dv = np.column_stack([np.interp(vb.time[inside], t, v[:, k]) for k in range(2)])
    cur = vb.velocity[inside]
    s = X.velocity_scores(cur, dv)
    return dict(n_bins=int(inside.sum()), r=s['r'], r2=s['r2'], rmse=s['rmse'],
                gain=float(np.sum(dv * cur) / max(np.sum(dv * dv), 1e-12)), decoder=dv, cursor=cur)


class OfflineModelTransfer(Section):
    id = 'x06'
    mode = 'cross'
    title = '离线模型跨天分析'
    description = ('可指定任意一份已勾选的记录作参考，测试其余全部记录（含更早记录）。'
                   'first 从第一份向后测试，each 对每份有后续记录的参考向后测试。'
                   '用参考记录的全部 trial 训练所选速度解码器与方向分类器；'
                   '对比每天自身交叉验证上限，并比较简单再校准（逐日特征去均值 / 重新标准化）；含在线记录说明')
    params = (
        Param('reference', '参考记录', 'recording', 'first', choices=('first', 'each')),
        Param('lam', '方向分类 ridge λ', 'float', 1.0, minimum=1e-6),
        Param('lam_ridge', '速度 Ridge λ', 'float', 1.0, minimum=1e-8),
        Param('lam_kf', 'KF 最小二乘 λ', 'float', 1e-4, minimum=0.0),
        Param('q_shrinkage', 'KF 观测协方差收缩 α', 'float', 0.5, minimum=0.0, maximum=1.0),
        Param('bin', '速度 bin 宽 / s', 'float', 0.1, minimum=0.02, maximum=0.5),
        Param('offset', '首个 bin 相对 T_TRIAL_START / s', 'float', 0.2, minimum=0.0),
        Param('n_bins', '每 trial bin 数', 'int', 10, minimum=1, maximum=200),
    )

    def run(self, ctx: AnalysisContext) -> SectionResult:
        p = ctx.params
        names = ('ridge', 'velocity_kf')
        metrics = metrics_for(names)
        spec = spec_of(ctx, p)
        sessions, n = ctx.sessions, len(ctx.sessions)
        labels = X.recording_labels(s.recording for s in sessions)
        reference = p['reference']
        if reference == 'first':
            refs = [0]
        elif reference == 'each':
            refs = list(range(n - 1))
        else:
            refs = [i for i, s in enumerate(sessions) if s.recording.key == reference]
            if len(refs) != 1:
                raise ValueError(f'参考记录不在当前选择中或无法唯一识别：{reference}')
        forward_only = reference in ('first', 'each')
        result = SectionResult()
        dirs, bins = [], []
        for k, s in enumerate(sessions):
            ctx.check_cancel()
            ctx.progress(f'features {labels[k]}')
            d = X.direction_data(ctx, k)
            vb = D.build_data(ctx.cohorts[k], spec, cache=ctx.cached)
            if len(d['x']) < 24 or vb.n < 50 or len(np.unique(d['blocks'])) < 3:
                raise ValueError(f'{s.label}: trial 或速度 bin 太少，无法训练/评估')
            dirs.append(d)
            bins.append(vb)
        days = np.array([X.day_number(s) for s in sessions])
        rows = []
        for r in refs:
            ctx.check_cancel()
            ctx.progress(f'reference {labels[r]}')
            tests = range(r + 1, n) if forward_only else (j for j in range(n) if j != r)
            for j in tests:
                ctx.check_cancel()
                _, cr, cj = X.common_columns(sessions[r].electrode_ids, sessions[j].electrode_ids)
                sv = X.fit_scaler(bins[r].features[:, cr])
                if sv['keep'].sum() < 5:
                    continue
                sd, clf = X.common_direction_model(ctx, r, j, dirs[r]['x'][:, cr], dirs[r]['y'], p['lam'])
                if clf is None:
                    continue
                ceil = ceiling(ctx, j, cj, dirs[j], names, spec, p)
                same_day_rank = (int(np.sum(days[r:j] == days[j])) if j > r
                                 else -int(np.sum(days[j + 1:r + 1] == days[j])))
                row = dict(reference=sessions[r].label, test=sessions[j].label, ref_index=r, test_index=j,
                           days_elapsed=int(days[j] - days[r]), same_day_rank=same_day_rank,
                           common_electrodes=int(len(cr)),
                           ceiling_direction_acc=round(ceil['direction']['acc'], 4),
                           **{f'ceiling_{m}_{q}': round(ceil[m][q], 4) for m in names for q in ('r2', 'rmse')})
                for mode in MODES:
                    acc = A.balanced_accuracy(dirs[j]['y'],
                                              X.predict_classifier(clf, X.transform(sd, dirs[j]['x'][:, cj], mode)))
                    row[f'{mode}_direction_acc'] = round(acc, 4)
                for m in names:
                    scores = self.velocity_transfer(ctx, r, j, cr, cj, m, spec, p)
                    for mode in MODES:
                        row.update({f'{mode}_{m}_r2': round(scores[mode]['r2'], 4),
                                    f'{mode}_{m}_rmse': round(scores[mode]['rmse'], 4)})
                rows.append(row)
        if not rows:
            raise ValueError('没有可评估的参考/测试记录对')
        # reference recordings' own CV ceiling on all electrodes (x = 0)
        own = {}
        for r in refs:
            all_cols = np.arange(C.N_COLUMNS)
            own[r] = ceiling(ctx, r, all_cols, dirs[r], names, spec, p)

        online = {}
        for k, s in enumerate(sessions):
            if s.decoder_velocity is not None:
                key = ('xday_online_check', id(ctx.cohorts[k]), spec.key(ctx.criteria))
                check = ctx.cached(key, lambda: online_check(s, bins[k]))
                if check:
                    online[k] = check
        ctx.progress('figures')
        base = self.base_method(spec, names, p, bins[0]) + ' ' + D.cohort_rule_text(ctx.cohorts[0], spec)
        plotted = [{k: v for k, v in r.items()} for r in rows]
        result.add_figure('Fixed-model performance vs days elapsed', self.vs_days(rows, own, labels, refs, metrics),
                          method=base + ' For each reference r and test recording j: models fitted once on all '
                          'of r (electrodes both transmitted) and applied unchanged to all of j under the three test '
                          'standardisations (lines). Direction = balanced accuracy of the one-hot ridge classifier '
                          '(trial window rates, chance 1/8); velocity = mean per-axis R² on all j bins the decoder '
                          'predicted. Hollow diamonds = ceiling: j\'s own 5-fold block CV (±1 purge) on the same common '
                          'electrodes; star at x = 0 = reference\'s own CV on all electrodes. R² < −1 drawn at the floor '
                          'as triangles. x = days since reference + 0.12 × same-day run rank.',
                          data=dict(transfer=plotted, reference_own_cv=[dict(reference=labels[r], **{
                              f'{m}_{k}': own[r][m][k] for m, k, _ in metrics}) for r in refs]))
        result.add_figure('Retention relative to within-day ceiling', self.retention(rows, labels, metrics),
                          method=base + ' Retention = (transfer − baseline) / (ceiling − baseline), baseline = 1/8 for '
                          'direction accuracy and 0 for R²; undefined (dropped) when ceiling − baseline ≤ 1e-6; clipped '
                          'to [−1, 1.5]; one point per (reference, test, standardisation).',
                          data=dict(transfer=plotted))
        if online:
            result.add_figure('Online recordings: decoder output vs cursor velocity',
                              self.online_figure(online, labels),
                              method='For recordings with a cursortasks_decoder_velocity stream: the online decoder '
                              'output linearly interpolated at each evaluation-bin right edge vs the cursor-displacement '
                              f'velocity of that {spec.bin_s * 1000:g} ms bin; r = mean Pearson r over x/y, R² = mean '
                              'per-axis R² of decoder vs cursor, gain = least-squares Σ(v_dec·v_cur)/Σ‖v_dec‖². Right: '
                              'first 300 bins of vx. Closed loop: the cursor was driven by this decoder.',
                              data={f'{labels[k]}': dict(decoder_vx=o['decoder'][:, 0], decoder_vy=o['decoder'][:, 1],
                                                         cursor_vx=o['cursor'][:, 0], cursor_vy=o['cursor'][:, 1])
                                    for k, o in online.items()})
        cmp_fig, cmp_data = self.comparison(rows, own, refs, labels, names)
        result.add_figure('Decoder comparison', cmp_fig,
                          method=base + ' Decoder comparison: (top, one panel per test standardisation) velocity R² '
                          '(mean of per-axis R²) of each decoder fitted once on the reference and applied unchanged to '
                          'each test recording, x = days since reference (+0.12 × signed same-day run rank); with several '
                          'references, values at the same x are averaged; hollow diamonds = the test recording\'s own '
                          'block-CV ceiling on the same common electrodes; stars at x = 0 = reference\'s own CV. '
                          '(bottom) retention = transfer R² / ceiling R² (baseline 0; pairs with ceiling ≤ 1e-6 '
                          'dropped), mean over pairs per decoder × standardisation (bars) and every pair (dots), clipped '
                          'to [−1, 1.5].', data=cmp_data)
        result.tables['decoder_comparison'] = cmp_data['retention_summary']
        result.tables['transfer'] = rows
        if online:
            result.tables['online_decoder'] = [dict(recording=sessions[k].label, bins=o['n_bins'],
                                                    pearson_r=round(o['r'], 4), r2=round(o['r2'], 4),
                                                    rmse=round(o['rmse'], 4), ls_gain_cursor_per_decoder=round(o['gain'], 4))
                                               for k, o in online.items()]
        first = [r for r in rows if r['ref_index'] == refs[0]]
        result.summary.update(
            reference=p['reference'], references=[labels[r] for r in refs], decoders=list(names),
            **{f'{m}_{metric}_{key}_first_ref': [r[f'{m}_{metric}_{key}'] for r in first]
               for m in MODES for metric, key, _ in metrics},
            ceiling_direction_acc_first_ref=[r['ceiling_direction_acc'] for r in first],
            **{f'ceiling_{m}_r2_first_ref': [r[f'ceiling_{m}_r2'] for r in first] for m in names},
            test_recordings_first_ref=[labels[r['test_index']] for r in first],
            days_elapsed_first_ref=[r['days_elapsed'] for r in first])
        if online:
            names = '、'.join(labels[k] for k in online)
            result.notes.append(
                f'在线记录（{names}）中光标由当时的在线解码器驱动：其“光标速度”标签本身就是在线解码输出的结果（闭环），'
                '与离线手动/开环记录的速度不是同一种行为量，离线模型在这些记录上的表现不能直接与开环日比较。'
                '在线解码输出与光标位移速度的一致性见最后一张图和 online_decoder 表。')
        result.notes.append('模型在参考记录的全部 trial 上训练后固定；“再校准”只改变测试日特征的均值/标准差，不重新拟合权重。'
                            '上限 = 测试记录在相同共同电极上的日内 block 交叉验证。')
        return result

    @staticmethod
    def velocity_transfer(ctx, r, j, cr, cj, name, spec, p):
        """Fit decoder `name` once on all of r (columns cr); scores on j for every standardisation."""
        dec = D.get(name)
        dp = D.decoder_params(dec, p)
        key = ('xday_model_transfer', id(ctx.cohorts[r]), id(ctx.cohorts[j]), name, spec.key(ctx.cohorts[r].criteria),
               tuple(sorted(dp.items())), tuple(map(int, cr)), tuple(map(int, cj)))

        def compute():
            ctx.progress(f'{D.label(name)}: {ctx.sessions[r].label} → {ctx.sessions[j].label}')
            tr = D.build_data(ctx.cohorts[r], spec, cache=ctx.cached).columns(cr)
            te = D.build_data(ctx.cohorts[j], spec, cache=ctx.cached).columns(cj)
            tr_n = te_n = None
            if D.needs_native(dec, te):
                tr_n = D.build_data(ctx.cohorts[r], spec, dec.BIN_SECONDS, cache=ctx.cached).columns(cr)
                te_n = D.build_data(ctx.cohorts[j], spec, dec.BIN_SECONDS, cache=ctx.cached).columns(cj)
            fitted = D.fit_model(dec, tr, dp, train_native=tr_n)
            out = {}
            for mode in MODES:
                pred, _, _ = D.predict_grid(fitted, te, test_native=te_n, test_mode=mode)
                out[mode] = X.velocity_scores(te.velocity, pred)
            return out
        ctx.check_cancel()
        return ctx.cached(key, compute)

    @staticmethod
    def base_method(spec, names, p, vb):
        return (f'Inputs: included trials of each recording. Direction: trial window rates (topbar window), one-hot '
                f'ridge classifier λ={p["lam"]:g} on reference-standardised features. Velocity: {spec.n_bins} bins of '
                f'{spec.bin_s * 1000:g} ms from T_TRIAL_START+{spec.offset:g} s, GT = cursor-displacement velocity, '
                f'features = trailing {vb.extra["neural_window"] * 1000:g} ms mean rate (Hz). Electrodes: ids both '
                'the '
                'reference and the test recording transmitted. Standardisation of the test day: fixed = reference '
                'mean/SD; re-centred = test-day mean, reference SD; re-standardized = test-day mean/SD (for velocity '
                'decoders applied by rescaling test features so the decoder\'s own training normalisation yields '
                'exactly that; no weights refitted). Velocity decoders: ' + D.describe(names, p, spec))

    # ------------------------------------------------------------------ figures
    @staticmethod
    def _x(row):
        return row['days_elapsed'] + 0.12 * row['same_day_rank']

    def label_test_axis(self, ax, rows, labels, *, reference=None):
        ticks = [(self._x(row), row['days_elapsed'], labels[row['test_index']]) for row in rows]
        if reference is not None:
            ticks.append((0, 0, labels[reference]))
        positions, days, names = zip(*sorted(ticks))
        X.recording_ticks(ax, positions, names)
        # Same-day runs are only 0.12 apart. Stagger their labels below the axis.
        ranks = {}
        for tick, day in zip(ax.xaxis.get_major_ticks(), days):
            rank = ranks.get(day, 0)
            tick.set_pad(tick.get_pad() + 8 * rank)
            ranks[day] = rank + 1

    def vs_days(self, rows, own, labels, refs, metrics):
        fig = new_figure(4.5 * len(metrics), 4.7)
        axes = fig.subplots(1, len(metrics))
        ref_styles = ['-', '--', ':', '-.']
        for ax, (metric, key, name) in zip(axes, metrics):
            for ri, r in enumerate(refs):
                sub = sorted([row for row in rows if row['ref_index'] == r], key=self._x)
                if not sub:
                    continue
                x = [self._x(row) for row in sub]
                ls = ref_styles[ri % len(ref_styles)]
                for m in MODES:
                    v = np.array([row[f'{m}_{metric}_{key}'] for row in sub], float)
                    clipped = v < -1
                    v = np.maximum(v, -0.97)
                    ax.plot(x, v, ls, marker='o', ms=3.5, lw=1,
                            color=MODE_COLORS[m], alpha=1 if len(refs) == 1 else 0.75,
                            label=MODE_SHORT[m] if ri == 0 else None)
                    if clipped.any():
                        ax.scatter(np.asarray(x)[clipped], v[clipped], marker='v', s=40, color=MODE_COLORS[m],
                                   zorder=5)
                ax.scatter(x, [row[f'ceiling_{metric}_{key}'] for row in sub], marker='D', s=26, facecolors='none',
                           edgecolors=style.INK, lw=0.9, zorder=4, label='test-day CV ceiling' if ri == 0 else None)
                ax.scatter([0], [own[r][metric][key]], marker='*', s=70, color=style.INK, zorder=5,
                           label='reference own CV' if ri == 0 else None)
            if metric == 'direction':
                ax.axhline(1 / 8, color=style.INK2, ls=':', lw=0.8, label='chance')
                ax.set_ylim(0, 1)
            else:
                ax.axhline(0, color=style.INK2, ls=':', lw=0.8)
                ax.set_ylim(-1.0, 1)
            if len(refs) == 1:
                self.label_test_axis(ax, rows, labels, reference=refs[0])
            ax.set_xlabel('days since reference (same-day runs offset by 0.12)')
            ax.set_ylabel(name)
            ax.set_title(name)
        axes[0].legend(frameon=False, fontsize=6.5, loc='upper right')
        if len(refs) == 1:
            subtitle = f'reference {labels[refs[0]]}'
        else:
            subtitle = ('line style by reference: ' +
                        ', '.join(f'{["solid", "dashed", "dotted", "dash-dot"][i % 4]} = {labels[r]}'
                                  for i, r in enumerate(refs[:4])) + (' …' if len(refs) > 4 else ''))
        fig.suptitle(f'Fixed offline models applied to test recordings · R² < −1 drawn as triangles at the floor'
                     f'\n{subtitle}',
                     fontsize=9)
        return fig

    def retention(self, rows, labels, metrics):
        fig = new_figure(4.5 * len(metrics), 4.4)
        axes = fig.subplots(1, len(metrics))
        for ax, (metric, key, name) in zip(axes, metrics):
            x = np.array([self._x(r) for r in rows])
            ceil = np.array([r[f'ceiling_{metric}_{key}'] for r in rows], float)
            base = 1 / 8 if metric == 'direction' else 0.0
            for k, m in enumerate(MODES):
                v = np.array([r[f'{m}_{metric}_{key}'] for r in rows], float)
                frac = (v - base) / np.where(ceil - base > 1e-6, ceil - base, np.nan)
                ax.scatter(x + (k - 1) * 0.05, np.clip(frac, -1, 1.5), s=16, color=MODE_COLORS[m], label=MODE_SHORT[m])
            ax.axhline(1, color=style.INK, lw=0.7, ls='--')
            ax.axhline(0, color=style.INK2, lw=0.6, ls=':')
            ax.set_ylim(-1.05, 1.55)
            if len({row['ref_index'] for row in rows}) == 1:
                self.label_test_axis(ax, rows, labels)
            ax.set_xlabel('days since reference')
            ax.set_ylabel('fraction of ceiling' + (' above chance' if metric == 'direction' else ''))
            ax.set_title(name)
        axes[0].legend(frameon=False, fontsize=6.5)
        fig.suptitle('Retention: (transfer − baseline) / (test-day CV ceiling − baseline); baseline = chance for '
                     'direction, 0 for R² · clipped to [−1, 1.5]', fontsize=9)
        return fig

    def online_figure(self, online, labels):
        k = len(online)
        fig = new_figure(11, 3.4 * k + 0.4)
        axes = np.atleast_2d(fig.subplots(k, 3, squeeze=False))
        for row, (idx, o) in zip(axes, online.items()):
            for c in range(2):
                ax = row[c]
                ax.scatter(o['decoder'][:, c], o['cursor'][:, c], s=3, alpha=0.3, color=style.MODEL_COLORS['online'],
                           lw=0)
                ax.set_xlabel(f'online decoder v{"xy"[c]}')
                ax.set_ylabel(f'cursor v{"xy"[c]} (bins)')
                ax.set_title(f'{labels[idx]} · {"xy"[c]}', fontsize=8, color=X.session_label_color(labels[idx]))
            ax = row[2]
            seg = slice(0, min(300, len(o['cursor'])))
            ax.plot(o['cursor'][seg, 0], color=style.MODEL_COLORS['gt'], lw=0.8, label='cursor vx')
            ax.plot(o['decoder'][seg, 0], color=style.MODEL_COLORS['online'], lw=0.8, label='decoder vx')
            ax.set_xlabel('bin (first 300)')
            ax.set_title(f"r = {o['r']:.2f} · R² = {o['r2']:.2f} · {o['n_bins']} bins", fontsize=8)
            ax.legend(frameon=False, fontsize=6)
        fig.suptitle('Online recordings: the cursor was driven by the online decoder (closed loop), so cursor '
                     'velocity is not an independent behavioural label', fontsize=9)
        return fig


    def comparison(self, rows, own, refs, labels, names):
        k = len(names)
        fig = new_figure(15, 9)
        grid = fig.add_gridspec(2, len(MODES))
        colors = [D.COLORS[m] for m in names]
        curves = []
        for c, mode in enumerate(MODES):
            ax = fig.add_subplot(grid[0, c])
            xs = sorted({self._x(r) for r in rows})
            span = max(map(abs, xs), default=1.0)
            for j, m in enumerate(names):
                dx = (j - (k - 1) / 2) * 0.025 * max(span, 1.0)        # side-by-side decoders at the same x
                ys = [float(np.mean([r[f'{mode}_{m}_r2'] for r in rows if self._x(r) == x])) for x in xs]
                cs = [float(np.mean([r[f'ceiling_{m}_r2'] for r in rows if self._x(r) == x])) for x in xs]
                ref = float(np.mean([own[r][m]['r2'] for r in refs]))
                px = np.r_[0.0, np.asarray(xs, float)] + dx
                py = np.r_[ref, ys]
                order = np.argsort(px)
                ax.plot(px[order], np.maximum(py[order], -0.97), lw=1.1, color=colors[j],
                        label=D.label(m), zorder=2)
                ax.scatter(px[:1], py[:1], marker='*', s=60, color=colors[j], zorder=5)
                v = np.asarray(ys)
                low = v < -1
                ax.scatter(px[1:][~low], v[~low], marker='o', s=16, color=colors[j], zorder=4)
                ax.scatter(px[1:][low], np.full(low.sum(), -0.97), marker='v', s=40, color=colors[j], zorder=4)
                for x_, y_ in zip(px[1:][low], v[low]):
                    ax.text(x_, -0.9 + 0.07 * j, f'{y_:.1f}', color=colors[j], fontsize=5.5, ha='center')
                ax.scatter(px[1:], cs, marker='D', s=24, facecolors='none', edgecolors=colors[j], lw=0.9, zorder=4)
                curves += [dict(standardisation=mode, decoder=m, x=x, transfer_r2=y, ceiling_r2=cc,
                                reference_cv_r2=ref) for x, y, cc in zip(xs, ys, cs)]
            ax.axhline(0, color=style.INK2, ls=':', lw=0.8)
            ax.set_ylim(-1, 1)
            if len(refs) == 1:
                self.label_test_axis(ax, rows, labels, reference=refs[0])
            ax.set_xlabel('days since reference')
            ax.set_ylabel('velocity R²')
            ax.set_title(f'{MODE_SHORT[mode]} test features (★ reference CV at 0, ◇ ceiling, ▼ R² < −1)',
                         fontsize=8)
            if c == 0:
                ax.legend(frameon=False, fontsize=6, loc='lower left')
        ax = fig.add_subplot(grid[1, :])
        width = 0.8 / len(MODES)
        summary = []
        for j, m in enumerate(names):
            for c, mode in enumerate(MODES):
                ceil = np.array([r[f'ceiling_{m}_r2'] for r in rows], float)
                v = np.array([r[f'{mode}_{m}_r2'] for r in rows], float)
                frac = v / np.where(ceil > 1e-6, ceil, np.nan)
                frac = np.clip(frac[np.isfinite(frac)], -1, 1.5)
                x = j + (c - (len(MODES) - 1) / 2) * width
                mean = float(np.mean(frac)) if len(frac) else np.nan
                ax.bar(x, mean, width * 0.9, color=colors[j], alpha=(0.35, 0.65, 1.0)[c], edgecolor=MODE_COLORS[mode],
                       lw=1.2)
                if np.isfinite(mean):
                    ax.text(x, mean, f'{mean:.2f}', ha='center', va='bottom' if mean >= 0 else 'top', fontsize=5.5)
                ax.scatter(np.full(len(frac), x), frac, s=9, color=style.INK, zorder=3)
                summary.append(dict(decoder=m, label=D.label(m), standardisation=mode, n_pairs=int(len(frac)),
                                    mean_retention=mean))
        ax.axhline(1, color=style.INK, ls='--', lw=0.7)
        ax.axhline(0, color=style.INK2, ls=':', lw=0.6)
        ax.set_ylim(-1.05, 1.55)
        ax.set_xticks(range(k), [D.label(m) for m in names], fontsize=7)
        ax.set_ylabel('transfer R² / test-day ceiling R²')
        ax.set_title('Retention per decoder: transfer R² / test-day ceiling R² (bars = mean over pairs, dots = pairs; '
                     'clipped to [−1, 1.5])', fontsize=8)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor=style.GREY, alpha=a, edgecolor=MODE_COLORS[mode], lw=1.2,
                                 label=f'{MODE_SHORT[mode]} test features')
                           for a, mode in zip((0.35, 0.65, 1.0), MODES)], frameon=False, fontsize=6.5)
        note = '' if k > 1 else ' · only one decoder selected'
        reference_title = labels[refs[0]] if len(refs) == 1 else f'{len(refs)} reference recordings'
        fig.suptitle('Decoder comparison: fixed offline velocity models across days'
                     f' ({reference_title}){note}', fontsize=10)
        return fig, dict(curves=curves, retention_summary=summary)


SECTION = OfflineModelTransfer()
