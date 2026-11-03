"""S12 v1: five source panels on one page, with decoding in supplementary pages."""
from __future__ import annotations

import numpy as np

from ... import config as C
from ... import decoders as D
from ...core import analysis as A
from ...core import style
from ...core.channels import column_tiers, read_channel_config
from ...core.target_decoding import grid_target_decoding
from ..base import AnalysisContext, Param, Section, SectionResult, new_figure
from ..shared_frame_loss import fpga_analysis, metadata_frame_indices, overlap
from ..s08_channel_config_v1.section import SUBSETS, TIER_COLORS
from ..s09_direction_decoding_v1.section import decode as direction_decode
from ..s11_online_vs_offline_v1.section import bin_online, scores as velocity_scores

RED, GREY, INK = '#c0392b', style.GREY, style.INK
FALLBACK_SUBSETS = ('all 512', 'active ≥1 Hz', 'sat. ≤1%')


class Summary(Section):
    id = 's12'
    title = 'Overview 一图流 Summary'
    description = ('首页依次汇总 S01 任务成功率、S02 通道饱和分区、S03 传输概览、'
                   'S04 降维概览和 S07 互斥通道组成。解码结果放在补充页。')
    cache_code_dependencies = (
        'sections/shared_frame_loss.py',
        'sections/s08_channel_config_v1',
        'sections/s09_direction_decoding_v1',
        'sections/s11_online_vs_offline_v1',
    )
    source_sections = ('s01', 's02', 's03', 's04', 's05', 's06', 's07')
    params = (
        Param('n_perm', '方向解码置换次数', 'int', 50, minimum=0, maximum=2000),
        Param('lam', '方向 / 目标解码 ridge λ', 'float', 1.0, minimum=1e-6),
        Param('channel_impact', '计算通道子集影响', 'bool', True),
        Param('zoom_start', '在线时间轴起点 / s（相对首个 trial）', 'float', 60.0, minimum=0.0),
        Param('zoom_len', '在线时间轴长度 / s', 'float', 30.0, minimum=2.0),
    )

    # ================================================================== compute
    def run(self, ctx: AnalysisContext) -> SectionResult:
        p, cohort, session = ctx.params, ctx.cohort, ctx.session
        names = ('ridge', 'velocity_kf')
        result = SectionResult()
        S = {}
        status = cohort.session_status()
        summ = cohort.summary()
        nt = session.neural_times
        grid_task = session.task_mode == 'grid'
        S.update(recording=session.label, date=session.recording.date, task_mode=session.task_mode,
                 session_type='online' if status['online'] else 'offline', online=bool(status['online']),
                 session_qualified=bool(status['qualified']), session_issues='; '.join(status['issues']),
                 n_outages=len(status['outages']),
                 outage_total_s=float(sum(o['duration_s'] for o in status['outages'])),
                 frames=int(len(nt)), duration_min=float((nt[-1] - nt[0]) / 60) if len(nt) > 1 else 0.0,
                 criteria_version=summ['criteria_version'], total_trial_rows=summ['total_trial_rows'],
                 complete_trials=summ['complete_trials'], success_trials=summ['success_trials'],
                 direction_trials=summ['included_trials'], velocity_trials=summ['velocity_trials'],
                 blocks=summ['blocks'])
        for k, v in summ['primary_reason_counts'].items():
            S[f'primary_{k}'] = int(v)
        for k, v in summ['flag_counts'].items():
            S[f'flag_{k}'] = int(v)

        ctx.progress('frame loss')
        loss = self.frame_loss(ctx)
        S.update({k: v for k, v in loss.items() if np.isscalar(v)})
        from . import modules
        fig, methods, data = modules.overview(ctx, result, S)
        result.add_figure('Overview', fig, method=methods, data=data)

        enough = cohort.n >= 16 and len(np.unique(cohort.blocks)) >= 4
        direction = grid = None
        if enough:
            ctx.progress('direction / target decoding')
            if grid_task:
                grid = ctx.cached(('grid_target_decoding', id(cohort), p['lam'], p['n_perm']),
                                  lambda: grid_target_decoding(cohort, p['lam'], p['n_perm']))
                S.update(grid_n_targets=grid['n_targets'], grid_target_r2_xy=grid['position']['r2_xy'],
                         grid_target_median_error=grid['position']['median_error'],
                         grid_displacement_r2_xy=grid['displacement_metrics']['r2_xy'])
                if grid['identity'] is not None:
                    S.update(grid_identity_balanced_accuracy=grid['identity']['balanced_accuracy'],
                             grid_identity_chance=grid['identity']['chance'], grid_identity_p=grid['identity']['p'])
            else:
                x = np.asarray(cohort.window_rates, float)
                ok = np.isfinite(x).all(1)
                direction = ctx.cached(('summary_direction', id(cohort), p['lam'], p['n_perm']),
                                       lambda: direction_decode(x[ok], cohort.directions[ok], cohort.blocks[ok],
                                                                p['lam'], p['n_perm']))
                S.update(direction_balanced_accuracy=direction['observed'], direction_p=direction['p'],
                         direction_within_45deg=direction['within45'], direction_n_perm=int(p['n_perm']))
        else:
            result.notes.append('方向队列 trial 太少（<16 或 block <4），跳过方向 / 目标解码。')

        ctx.progress('velocity decoding')
        spec = D.BinSpec()
        vel = None
        data = D.build_data(cohort, spec, cache=ctx.cached) if cohort.n else None
        if data is not None and data.n >= 50 and len(np.unique(data.block)) >= 4:
            cv = D.section_cv(ctx, cohort, names, spec, p, data=data)
            metrics = {m: velocity_scores(data.velocity, cv['results'][m]['pred']) for m in names}
            vel = dict(data=data, cv=cv, metrics=metrics)
            for m, s in metrics.items():
                S.update({f'velocity_{m}_r2_x': s['r2_x'], f'velocity_{m}_r2_y': s['r2_y'],
                          f'velocity_{m}_r2_mean': float(np.mean([s['r2_x'], s['r2_y']])),
                          f'velocity_{m}_rmse': s['rmse']})
            S['velocity_bins'] = int(data.n)
        else:
            result.notes.append('速度队列有效 bin 太少（<50 或 block <4），跳过速度解码。')

        impact = None
        if p['channel_impact'] and enough and vel is not None:
            ctx.progress('channel subsets')
            impact = self.channel_impact(ctx, data, spec, grid_task, p)
            for row in impact['rows']:
                key = row['key']
                S[f'channels_{key}_n'] = row['n_channels']
                S[f'channels_{key}_{impact["metric_key"]}'] = row['metric']
                S[f'channels_{key}_velocity_r2'] = row['velocity_r2']
            S['channel_subset_source'] = impact['source']

        online = None
        if status['online'] and session.decoder_velocity is not None and vel is not None:
            ctx.progress('online vs offline')
            on, _, _ = bin_online(session, data, spec.bin_s)
            on_scores = velocity_scores(data.velocity, on)
            best = max(names, key=lambda m: np.nan_to_num(vel['metrics'][m]['r2_xy'], nan=-np.inf))
            online = dict(pred=on, scores=on_scores, best=best,
                          agreement=velocity_scores(on, vel['cv']['results'][best]['pred']))
            S.update(online_r2_x=on_scores['r2_x'], online_r2_y=on_scores['r2_y'], online_r2_xy=on_scores['r2_xy'],
                     online_rmse=on_scores['rmse'], best_offline_decoder=best,
                     best_offline_r2_xy=vel['metrics'][best]['r2_xy'],
                     best_offline_vs_online_r2_xy=online['agreement']['r2_xy'])

        ctx.check_cancel()
        ctx.progress('figure')
        self._data = {}
        if direction is not None or grid is not None or vel is not None:
            decoding = self.decoding_figure(ctx, direction, grid, vel, online, names, p)
            result.add_figure('Supplementary decoding overview', decoding,
                              method=self.method(ctx, spec, names, None, grid_task, online, p), data=self._data)
        if impact is not None:
            self._data={}
            supplement=new_figure(12,5)
            self.impact_panel(supplement,supplement.add_gridspec(1,1)[0],impact)
            supplement.suptitle(f'{session.label} · Supplementary channel-subset decoding impact',fontsize=10)
            result.add_figure('Supplementary channel-subset decoding impact',supplement,
                              method=self.method(ctx,spec,names,impact,grid_task,None,p),data=self._data)
        result.summary.update(S)
        result.tables['summary'] = [dict(key=k, value=str(v)) for k, v in S.items()]
        result.tables['trial_selection'] = cohort.exclusion_table()
        return result

    def frame_loss(self, ctx):
        session, cohort = ctx.session, ctx.cohort
        nt = session.neural_times
        thr = ctx.params['_source_sections']['s03']['params']['gap_threshold']
        intervals = np.diff(nt)
        gap_idx = np.flatnonzero(intervals > thr)
        frame_indices = metadata_frame_indices(session)
        fpga = (ctx.cached(('s03_fpga', id(session)), lambda: fpga_analysis(session, frame_indices))
                if frame_indices is not None else None)
        lefts = cohort.starts + cohort.criteria.window_start
        rights = cohort.starts + cohort.criteria.window_end
        all_starts = np.array([r['start'] for r in cohort.table if np.isfinite(r['start'])])
        al, ar = all_starts + cohort.criteria.window_start, all_starts + cohort.criteria.window_end
        gap_dir = overlap(nt[gap_idx], nt[gap_idx + 1], lefts, rights).any(1)
        gap_all = overlap(nt[gap_idx], nt[gap_idx + 1], al, ar).any(1)
        out = dict(gap_threshold_s=thr, receive_gaps=int(len(gap_idx)),
                   receive_gap_total_s=float(intervals[gap_idx].sum()),
                   receive_interval_median_ms=float(np.median(intervals) * 1000),
                   fpga_clock=fpga is not None,
                   pct_direction_trials_window_gap=float(100 * gap_dir.mean()) if len(gap_dir) else float('nan'),
                   pct_all_trials_window_gap=float(100 * gap_all.mean()) if len(gap_all) else float('nan'),
                   _gap_idx=gap_idx, _intervals=intervals)
        if fpga is not None:
            drop_idx = np.flatnonzero(fpga['lost'] > 0)
            drop_dir = overlap(nt[drop_idx], nt[drop_idx + 1], lefts, rights).any(1)
            drop_all = overlap(nt[drop_idx], nt[drop_idx + 1], al, ar).any(1)
            out.update(fpga_frames_lost=fpga['frames_lost'], fpga_drop_rate=fpga['drop_rate'],
                       fpga_lost_seconds=fpga['lost_seconds'],
                       pct_direction_trials_window_drop=float(100 * drop_dir.mean()) if len(drop_dir) else float('nan'),
                       pct_all_trials_window_drop=float(100 * drop_all.mean()) if len(drop_all) else float('nan'),
                       _drop_idx=drop_idx, _lost=fpga['lost'])
        return out

    def channel_impact(self, ctx, data, spec, grid_task, p):
        session, cohort = ctx.session, ctx.cohort
        ids = session.electrode_ids
        subsets = []
        if ctx.channel_config_path:
            cfg = read_channel_config(ctx.channel_config_path)
            tiers = column_tiers(ids, cfg)
            for name, members in SUBSETS:
                subsets.append((name, name.replace('+', '_').replace(' ', '_'), np.flatnonzero(np.isin(tiers, members)),
                                TIER_COLORS.get(members[0], GREY) if len(members) == 1 else '#5b8c5a'))
            source = 'channel config: ' + ctx.channel_config_path.replace(chr(92), '/').split('/')[-1]
        else:
            mean_hz = session.counts.mean(0, dtype=np.float64) / session.frame_seconds
            sat = np.count_nonzero(session.counts >= C.MAX_CODE, axis=0) / len(session.counts)
            subsets = [(FALLBACK_SUBSETS[0], 'all', np.arange(len(ids)), '#5b6670'),
                       (FALLBACK_SUBSETS[1], 'active_1hz', np.flatnonzero(mean_hz >= 1.0), '#2e8b57'),
                       (FALLBACK_SUBSETS[2], 'unsaturated', np.flatnonzero(sat <= 0.01), '#2a78d6')]
            source = 'no channel config (fallback subsets)'
        x = np.nan_to_num(np.asarray(cohort.window_rates, float))
        y, blocks = cohort.directions, cohort.blocks
        rows = []
        for name, key, cols, color in subsets:
            row = dict(subset=name, key=key, n_channels=int(len(cols)), metric=np.nan, velocity_r2=np.nan,
                       velocity_r2_x=np.nan, velocity_r2_y=np.nan, color=color)
            if len(cols) >= 2:
                if grid_task:
                    from ...core.target_decoding import cv_ridge_regression
                    target = np.c_[cohort.array('target_x'), cohort.array('target_y')]
                    pred, folds = cv_ridge_regression(x[:, cols], target, blocks, p['lam'])
                    ok = folds >= 0
                    row['metric'] = float(1 - np.sum((target[ok] - pred[ok]) ** 2)
                                          / np.sum((target[ok] - target[ok].mean(0)) ** 2))
                else:
                    sc, _, _, tested = A.ridge_classifier_cv(x[:, cols], y, blocks, lam=p['lam'])
                    row['metric'] = A.balanced_accuracy(y[tested], sc[tested].argmax(1))
                cv = D.section_cv(ctx, cohort, ('ridge',), spec, p, columns=cols)
                s = velocity_scores(cv['data'].velocity, cv['results']['ridge']['pred'])
                row.update(velocity_r2=float(np.mean([s['r2_x'], s['r2_y']])), velocity_r2_x=s['r2_x'],
                           velocity_r2_y=s['r2_y'])
            rows.append(row)
        metric = 'target position R² (xy)' if grid_task else 'direction balanced accuracy'
        return dict(rows=rows, source=source, metric=metric,
                    metric_key='target_r2' if grid_task else 'direction_ba')

    # ================================================================== figure
    def decoding_figure(self, ctx, direction, grid, vel, online, names, p):
        fig = new_figure(16, 12 if online is not None else 6)
        layout = fig.add_gridspec(2 if online is not None else 1, 1)
        panels = layout[0].subgridspec(1, 2)
        if grid is not None:
            self.grid_panel(fig, panels[0], grid)
        else:
            self.direction_panel(fig, panels[0], direction, p)
        self.velocity_panel(fig, panels[1], vel, names)
        if online is not None:
            self.online_row(fig, layout[1], ctx, vel, online, names, p)
        fig.suptitle(f'{ctx.session.label} · Supplementary decoding overview', fontsize=12)
        return fig

    def impact_panel(self, fig, spec, impact):
        g = spec.subgridspec(1, 2, wspace=0.02)
        if impact is None:
            ax = fig.add_subplot(g[:])
            ax.axis('off')
            ax.text(0.5, 0.5, 'Channel-subset impact not computed\n(too few trials/bins or disabled)', ha='center',
                    va='center', transform=ax.transAxes)
            ax.set_title('2 · 通道筛选影响', fontsize=9, loc='left', fontweight='bold')
            return
        rows = impact['rows']
        labels = [f'{r["subset"]}\n(n={r["n_channels"]})' for r in rows]
        colors = [r['color'] for r in rows]
        for k, (key, title) in enumerate((('metric', impact['metric']), ('velocity_r2', 'velocity R² (ridge, mean x,y)'))):
            ax = fig.add_subplot(g[k])
            vals = [r[key] for r in rows]
            ax.bar(range(len(rows)), np.nan_to_num(vals), color=colors)
            for i, v in enumerate(vals):
                ax.text(i, max(np.nan_to_num(v), 0), f'{v:.2f}' if np.isfinite(v) else 'n/a', ha='center',
                        va='bottom', fontsize=6.5)
            if key == 'metric' and 'balanced' in impact['metric']:
                ax.axhline(0.125, color=RED, ls='--', lw=0.8)
            ax.axhline(0, color=GREY, lw=0.5)
            ax.set_xticks(range(len(rows)), labels, fontsize=6, rotation=0)
            ax.set_ylabel(title, fontsize=7)
            ax.margins(y=0.15)
            ax.tick_params(labelsize=6)
            if k == 0:
                ax.set_title(f'2 · 通道筛选影响 — {impact["source"]}', fontsize=9, loc='left', fontweight='bold')
        self._data['TR_channel_subsets'] = [{k: v for k, v in r.items() if k != 'color'} for r in rows]

    # -------------------------------------------------------------- BL direction
    def direction_panel(self, fig, spec, r, p):
        g = spec.subgridspec(1, 2, width_ratios=[1.1, 1], wspace=0.02)
        ax = fig.add_subplot(g[0])
        if r is None:
            ax.axis('off')
            ax.text(0.5, 0.5, 'not enough trials', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('3 · 方向解码', fontsize=9, loc='left', fontweight='bold')
            return
        ax.imshow(r['fraction'], cmap='Blues', vmin=0, vmax=max(0.5, r['fraction'].max()))
        for t in range(8):
            for q in range(8):
                v = r['fraction'][t, q]
                if v >= 0.05:
                    ax.text(q, t, f'{100 * v:.0f}', ha='center', va='center', fontsize=5.5,
                            color='white' if v > 0.45 else INK)
        deg = [str(d * 45) for d in range(8)]
        ax.set_xticks(range(8), deg, fontsize=6)
        ax.set_yticks(range(8), deg, fontsize=6)
        ax.set_xlabel('predicted / °', fontsize=7)
        ax.set_ylabel('target / °', fontsize=7)
        ax.set_title('3 · 方向解码 — confusion (row %, held-out)', fontsize=9, loc='left', fontweight='bold')
        ax = fig.add_subplot(g[1])
        null = r['null']
        if len(null):
            ax.hist(100 * null, bins=20, color=GREY, label=f'null ({len(null)} perms)')
        ax.axvline(100 * r['observed'], color='#2176ae', lw=2, label='observed')
        ax.axvline(12.5, color=RED, ls='--', lw=0.8, label='chance')
        ax.set_xlabel('balanced accuracy / %', fontsize=7)
        ax.legend(frameon=False, fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_title(f'BA {100 * r["observed"]:.1f}% · ±45° {100 * r["within45"]:.0f}%\n'
                     f'p = {r["p"]:.3g}  (n_perm={len(null)}; min p = {1 / (1 + len(null)):.3g})', fontsize=7.5,
                     loc='left')
        self._data['BL_confusion'] = [dict(target_deg=t * 45, predicted_deg=q * 45, n=int(r['counts'][t, q]),
                                           fraction=float(r['fraction'][t, q])) for t in range(8) for q in range(8)]
        self._data['BL_permutation_null'] = dict(permutation=np.arange(len(null)), balanced_accuracy=null)
        self._data['BL_statistics'] = [dict(balanced_accuracy=r['observed'], accuracy=r['accuracy'],
                                            within_45deg=r['within45'], p=r['p'], n_perm=len(null),
                                            n_trials=int(r['tested'].sum()))]

    def grid_panel(self, fig, spec, gres):
        g = spec.subgridspec(1, 2, width_ratios=[1.2, 1], wspace=0.02)
        ax = fig.add_subplot(g[0])
        targets, labels_xy = gres['targets'], gres['target']
        pred = gres['position_pred']
        cls = np.unique(np.round(labels_xy, 3), axis=0, return_inverse=True)[1].ravel()
        rows = []
        for k, t in enumerate(targets):
            sel = (cls == k) & np.isfinite(pred).all(1)
            if not sel.any():
                continue
            m = pred[sel].mean(0)
            ax.plot([t[0], m[0]], [t[1], m[1]], color=GREY, lw=0.7)
            rows.append(dict(target_x=t[0], target_y=t[1], mean_pred_x=m[0], mean_pred_y=m[1], n_trials=int(sel.sum())))
        ax.scatter(targets[:, 0], targets[:, 1], marker='s', s=26, facecolors='none', edgecolors=INK, lw=0.8,
                   label='target')
        ax.scatter([r['mean_pred_x'] for r in rows], [r['mean_pred_y'] for r in rows], s=12, color='#2176ae',
                   label='mean held-out prediction')
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(frameon=False, fontsize=6, loc='upper left', bbox_to_anchor=(0, -0.12), ncol=2)
        ax.tick_params(labelsize=6)
        pm = gres['position']
        ax.set_title(f'3 · 目标解码 (Grid) — target position R² {pm["r2_xy"]:.2f}', fontsize=9, loc='left',
                     fontweight='bold')
        ax = fig.add_subplot(g[1])
        lines = [f'Grid: {gres["n_targets"]} unique targets, {gres["n_trials"]} trials',
                 '(8-direction confusion not used: targets',
                 ' lie on a lattice and starts vary)', '',
                 'Target position (ridge regression, block CV)',
                 f'  R² x / y      {pm["r2_x"]:.2f} / {pm["r2_y"]:.2f}',
                 f'  median error  {pm["median_error"]:.0f} (mean-only {pm["baseline_median_error"]:.0f})',
                 'Reach displacement (target − start)',
                 f'  R² xy         {gres["displacement_metrics"]["r2_xy"]:.2f}', '']
        ident = gres['identity']
        if ident is not None:
            lines += ['Target identity (one-hot ridge, argmax)',
                      f'  balanced acc  {100 * ident["balanced_accuracy"]:.1f}%',
                      f'  chance        {100 * ident["chance"]:.1f}% (1/{gres["n_targets"]})',
                      f'  p (perm)      {ident["p"]:.3g} (n={len(ident["null"])})',
                      f'  median class error {ident["median_class_error"]:.0f}']
        else:
            lines += ['Target identity: skipped', f'  (needs ≤ 64 targets and ≥ 3 trials each)']
        style.text_panel(ax, lines, fontsize=6.8)
        self._data['BL_grid_target_means'] = rows
        self._data['BL_grid_trials'] = dict(target_x=labels_xy[:, 0], target_y=labels_xy[:, 1], pred_x=pred[:, 0],
                                            pred_y=pred[:, 1], fold=gres['folds'])
        stats = dict(n_targets=gres['n_targets'], **{f'position_{k}': v for k, v in pm.items()},
                     **{f'displacement_{k}': v for k, v in gres['displacement_metrics'].items()})
        if ident is not None:
            stats.update(identity_balanced_accuracy=ident['balanced_accuracy'], identity_chance=ident['chance'],
                         identity_p=ident['p'], identity_n_perm=len(ident['null']))
        self._data['BL_grid_statistics'] = [stats]

    # -------------------------------------------------------------- BR velocity
    def velocity_panel(self, fig, spec, vel, names):
        g = spec.subgridspec(1, 2, width_ratios=[1.3, 1], wspace=0.02)
        ax = fig.add_subplot(g[0])
        if vel is None:
            ax.axis('off')
            ax.text(0.5, 0.5, 'not enough velocity bins', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('4 · 速度解码', fontsize=9, loc='left', fontweight='bold')
            return
        width = 0.8 / len(names)
        rows = []
        for j, m in enumerate(names):
            s = vel['metrics'][m]
            vals = [s['r2_x'], s['r2_y'], float(np.mean([s['r2_x'], s['r2_y']]))]
            xs = np.arange(3) + (j - (len(names) - 1) / 2) * width
            ax.bar(xs, vals, width, color=D.COLORS.get(m, GREY), label=f'{D.label(m)} (RMSE {s["rmse"]:.0f})')
            for x, v in zip(xs, vals):
                ax.text(x, max(v, 0), f'{v:.2f}', ha='center', va='bottom', fontsize=6)
            rows.append(dict(decoder=m, r2_x=vals[0], r2_y=vals[1], r2_mean=vals[2], r2_xy_pooled=s['r2_xy'],
                             rmse=s['rmse'], r_x=s['r_x'], r_y=s['r_y'], n_bins=s['n']))
        ax.set_xticks(range(3), ['x', 'y', 'mean'])
        ax.axhline(0, color=GREY, lw=0.5)
        ax.set_ylabel('held-out R² vs GT', fontsize=7)
        ax.margins(y=0.18)
        ax.legend(frameon=False, fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
        data = vel['data']
        ax.set_title(f'4 · 速度解码 — {data.n} bins, {len(np.unique(data.trial))} velocity-cohort trials',
                     fontsize=9, loc='left', fontweight='bold')
        self._data['BR_velocity_metrics'] = rows
        best = max(names, key=lambda m: np.nan_to_num(vel['metrics'][m]['r2_xy'], nan=-np.inf))
        ax = fig.add_subplot(g[1])
        pred = vel['cv']['results'][best]['pred']
        traj = []
        for d in range(8):
            for src, values, ls in (('GT', data.velocity, '-'), (best, pred, '--')):
                paths = []
                for t in np.unique(data.trial[data.direction == d]):
                    r = np.flatnonzero(data.trial == t)
                    if data.bin[r][0] != 0 or len(np.unique(data.sequence[r])) != 1 or not np.isfinite(values[r]).all():
                        continue
                    paths.append(np.vstack([data.position[r[0]],
                                            data.position[r[0]] + np.cumsum(values[r], 0) * data.bin_seconds]))
                n = min((len(q) for q in paths), default=0)
                if not n:
                    continue
                mean = np.mean([q[:n] for q in paths], 0)
                ax.plot(*mean.T, color=style.direction_color(d), ls=ls, lw=1.4 if ls == '-' else 1.0)
                traj += [dict(direction_deg=d * 45, source=src, point=k, x=float(a), y=float(b), n_trials=len(paths))
                         for k, (a, b) in enumerate(mean)]
        ax.plot([], [], color=INK, label='GT')
        ax.plot([], [], color=INK, ls='--', label=D.label(best))
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(frameon=False, fontsize=6)
        ax.tick_params(labelsize=6)
        ax.set_title('Mean integrated trajectory per direction', fontsize=7.5, loc='left')
        self._data['BR_mean_trajectories'] = traj

    # -------------------------------------------------------------- online row
    def online_row(self, fig, spec, ctx, vel, online, names, p):
        cohort, data = ctx.cohort, vel['data']
        g = spec.subgridspec(2, 2, width_ratios=[1, 2.4], wspace=0.02, hspace=0.02)
        ax = fig.add_subplot(g[:, 0])
        keys = ['online'] + list(names)
        sc = dict(online=online['scores'], **vel['metrics'])
        colors = ['#009e73'] + [D.COLORS.get(m, GREY) for m in names]
        r2 = [sc[k]['r2_xy'] for k in keys]
        ax.bar(range(len(keys)), r2, color=colors)
        for i, k in enumerate(keys):
            ax.text(i, max(r2[i], 0), f'R² {r2[i]:.2f}\nRMSE {sc[k]["rmse"]:.0f}', ha='center', va='bottom',
                    fontsize=6)
        ax.set_xticks(range(len(keys)), ['online\n(closed loop)'] + [D.label(m) + '\n(offline CV)' for m in names],
                      fontsize=6.5)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel('R² vs GT (x,y pooled), same 100 ms bins', fontsize=7)
        ax.set_title('5 · 在线 vs 离线 — online = closed-loop performance,\ncursor driven by the decoder '
                     '(descriptive, not held-out)', fontsize=8.5, loc='left', fontweight='bold')
        self._data['ON_metrics'] = [dict(source=k, **sc[k]) for k in keys] + [
            dict(source=f'{online["best"]} vs online', **online['agreement'])]
        t0 = float(cohort.starts.min())
        lo, hi = t0 + p['zoom_start'], t0 + p['zoom_start'] + p['zoom_len']
        sel = np.flatnonzero((data.time >= lo) & (data.time <= hi))
        if not len(sel):
            lo, hi = float(data.time.min()) - 0.5, float(data.time.min()) - 0.5 + p['zoom_len']
            sel = np.flatnonzero((data.time >= lo) & (data.time <= hi))
        best = online['best']
        series = [('GT', data.velocity, INK, 1.6), ('online', online['pred'], '#009e73', 1.1),
                  (D.label(best), vel['cv']['results'][best]['pred'], D.COLORS.get(best, GREY), 1.0)]
        strip = dict(time_s=data.time[sel] - t0, trial=data.trial[sel])
        for axis in range(2):
            ax = fig.add_subplot(g[axis, 1])
            for j, (name, values, color, lw) in enumerate(series):
                for k, t in enumerate(np.unique(data.trial[sel])):
                    r = sel[data.trial[sel] == t]
                    ax.plot(data.time[r] - t0, values[r, axis], color=color, lw=lw,
                            label=name if k == 0 and axis == 0 else None)
                strip[f'{name}_v{"xy"[axis]}'] = values[sel, axis]
            ax.axhline(0, color=GREY, lw=0.4)
            ax.set_ylabel(f'v{"xy"[axis]}', fontsize=7)
            ax.set_xlim(lo - t0, hi - t0)
            ax.tick_params(labelsize=6)
            if axis == 0:
                ax.legend(frameon=False, fontsize=6, ncol=3, loc='lower right', bbox_to_anchor=(1, 1))
                ax.set_title(f'Velocity strip (100 ms bins; best offline = {D.label(best)})', fontsize=7.5, loc='left')
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel('time / s from first direction-cohort trial start', fontsize=7)
        self._data['ON_velocity_strip'] = strip

    # -------------------------------------------------------------- method
    @staticmethod
    def method(ctx, spec, names, impact, grid_task, online, p):
        crit = ctx.cohort.criteria
        parts = [f'Single recording {ctx.session.label}; trial filter {crit.criteria_version}.']
        if impact is not None:
            if impact['source'].startswith('channel config:'):
                sub = ('Channel subsets from INI or Neuralinsight log tiers (zero-based indices converted to one-based '
                       'electrode ids, intersected with the transmitted electrodes): good, good+uncertain, all 512, '
                       'bad only.')
            else:
                sub = ('No channel config given, fallback subsets: all 512 transmitted columns; columns with mean rate '
                       '≥ 1 Hz over the whole file; columns with ≤ 1% of frames at the maximum code 3.')
            metric = ('target position R² (ridge regression of target x,y from the window rates, pooled x,y)'
                      if grid_task else 'eight-direction balanced accuracy (one-hot ridge, argmax)')
            parts.append(f'Panel 2 (channel impact): {sub} For each subset: {metric} on the direction cohort, and '
                         'velocity R² (mean of x and y) of the ridge decoder on the velocity cohort bins; both with '
                         'the same 5 chronological block folds (±1 block purge), lambda '
                         f'{p["lam"]:g} for the trial-level model.')
        if grid_task:
            parts.append('Panel 3 (Grid): target position (x, y) and reach displacement (target − start) regressed '
                         f'from the window mean rates with standardized ridge (penalty n·lambda, lambda {p["lam"]:g}) '
                         'over 5 block folds (±1 purge); grey lines join each target to the mean held-out prediction '
                         'of its trials; median error = median Euclidean distance (task units), mean-only baseline '
                         'predicts the target mean. Target identity: one-hot ridge over the unique targets (rounded '
                         f'to 1e-3) when ≤ 64 targets with ≥ 3 trials each; balanced accuracy vs 1/K; p from '
                         f'{p["n_perm"]} within-block label permutations with exact refits.')
        else:
            parts.append('Panel 3 (direction): window mean rate (Hz) per channel, z-scored on training folds, '
                         f'one-hot ridge (penalty n·lambda, lambda {p["lam"]:g}), argmax; 5 chronological block folds '
                         '(±1 purge); confusion = row-normalised held-out counts (%); balanced accuracy = mean '
                         f'per-direction recall; p = (1 + #null ≥ observed)/(1 + n_perm) from {p["n_perm"]} '
                         'within-block label permutations with exact refits (small n_perm: p cannot go below '
                         '1/(1+n_perm)).')
        parts.append(f'Panel 4 (velocity): velocity-cohort bins of {spec.bin_s * 1000:g} ms from '
                     f'T_TRIAL_START+{spec.offset:g} s, {spec.n_bins} per trial; GT = cursor displacement / bin; '
                     f'neural feature = mean Hz over the trailing {(spec.neural_window or crit.velocity_neural_window) * 1000:g} ms. Decoders: '
                     + D.describe(names, p, spec) + ' Held-out over 5 block folds (±1 purge). R² per axis, mean of '
                     'x,y; RMSE = sqrt(mean ‖v̂ − v‖²). Inset: per direction, mean over trials with complete '
                     'contiguous bins of the position integrated from the first-bin cursor position (GT solid, '
                     'best decoder dashed).')
        if online is not None:
            parts.append('Panel 5 (online): online = mean of cursortasks_decoder_velocity samples whose receive time '
                         'falls in each evaluation bin (≥ 2 samples). The online decoder drives the cursor (closed '
                         'loop), so its agreement with GT is closed-loop performance, not a held-out evaluation; '
                         'offline decoders are held-out CV on the same bins. Strip: bins in the zoom window '
                         f'({p["zoom_start"]:g} s + {p["zoom_len"]:g} s after the first trial start), lines break '
                         'between trials.')
        return ' '.join(parts)



SECTION = Summary()
