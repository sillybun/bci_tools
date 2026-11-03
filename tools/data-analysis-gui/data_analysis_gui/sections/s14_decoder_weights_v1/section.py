"""Descriptive full-run weights using S10 inputs and decoder parameters."""
from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from scipy import linalg

from ... import decoders as D
from ... import config as C
from ...core.channels import layout_grid
from ...core import style
from ...decoders._blas import single_thread
from ..base import Param, Section, SectionResult
from ..s10_velocity_decoding_v1.section import spec_of


def steady_gain(par, *, tolerance=1e-10, max_steps=10000):
    """Converged gain for the exact local velocity_kf recursion (not encoding C)."""
    A, C, W, Q = (np.asarray(par[k], float) for k in ('A', 'C', 'W', 'Q'))
    ctqi = linalg.cho_solve(linalg.cho_factor(Q), C).T
    information = ctqi @ C
    covariance = np.zeros((2, 2))
    previous = None
    for step in range(1, max_steps + 1):
        prior = A @ covariance @ A.T + W
        covariance = linalg.inv(linalg.inv(prior) + information)
        gain = covariance @ ctqi
        if not np.isfinite(gain).all():
            raise ValueError('KF 有效增益出现非有限值')
        if previous is not None and linalg.norm(gain - previous) <= tolerance * max(linalg.norm(gain), 1e-12):
            return gain, step
        previous = gain.copy()
    raise ValueError('KF 增益未收敛，无法用单张固定权重图概括；请改选 Ridge 或调整 S10 参数')


def fit_weights(data, name, params, basis):
    if data.n < 50 or len(np.unique(data.trial)) < 2:
        raise ValueError('拟合权重至少需要 50 个有效速度 bin 和 2 个 trial')
    if not np.isfinite(data.features).all() or not np.isfinite(data.velocity).all():
        raise ValueError('拟合数据含非有限特征或速度')
    if not np.any(np.std(data.features, axis=0) > 1e-12):
        raise ValueError('所有传输通道的拟合特征均为常数，无法拟合权重')
    with single_thread():
        model = D.get(name).fit(data, params)
        keep = np.asarray(model['keep'], bool)
        steps = None
        if name == 'ridge':
            standardized = np.asarray(model['coef'], float)
        elif name == 'velocity_kf':
            gain, steps = steady_gain(model['par'])
            standardized = gain.T * model['scale']
        else:
            raise ValueError(f'不支持提取权重的模型：{name}')
    weights = np.zeros((len(data.electrode_ids), 2))
    weights[keep] = standardized if basis == 'per_SD' else standardized / model['sd'][:, None]
    return dict(weights=weights, keep=keep, model_info=model['info'], gain_steps=steps)


def channel_rows(layout, ids, weights, keep):
    ids = np.asarray(ids, int)
    if weights.shape != (len(ids), 2) or len(set(ids)) != len(ids) or np.any((ids < 1) | (ids > 1024)):
        raise ValueError('模型权重与 XDF 电极 ID 不匹配')
    lookup = {int(e): j for j, e in enumerate(ids)}
    rows = []
    for index, (r, c) in enumerate(layout['position']):
        e = index + 1
        j = lookup.get(e)
        vx, vy = weights[j] if j is not None else (np.nan, np.nan)
        rows.append(dict(channel_0based=index, electrode_id=e, stored_column_0based=j,
                         shank=int(c + 1), elec_index=int(r + 1), transmitted=j is not None,
                         status='untransmitted' if j is None else 'fitted' if keep[j] else 'constant_dropped',
                         vx_signed=float(vx), vy_signed=float(vy),
                         vx=float(abs(vx)), vy=float(abs(vy)), norm=float(np.hypot(vx, vy))))
    return sorted(rows, key=lambda row: (row['elec_index'], row['shank']))


def depth_rows(rows, size):
    total = sum(row['norm'] for row in rows if row['transmitted'])
    result = []
    for start in range(1, 129, size):
        end = min(start + size - 1, 128)
        sites = [r for r in rows if start <= r['elec_index'] <= end]
        selected = [r for r in sites if r['transmitted']]
        row = dict(elec_start=start, elec_end=end, elec_mid=(start + end) / 2,
                   n_sites=len(sites), n_transmitted=len(selected),
                   n_fitted=sum(r['status'] == 'fitted' for r in selected),
                   transmitted_fraction=len(selected) / len(sites))
        for key in ('vx', 'vy', 'norm'):
            values = np.array([r[key] for r in selected])
            row[f'{key}_mean_abs'] = float(np.mean(np.abs(values))) if len(values) else np.nan
            row[f'{key}_median_abs'] = float(np.median(np.abs(values))) if len(values) else np.nan
        row['norm_weight_share'] = sum(r['norm'] for r in selected) / total if total > 0 else np.nan
        result.append(row)
    return result


def weight_figure(layout, ids, values, component, numeric, banner, unit, *, color_limits=(0., 25.)):
    """Fixed reversed-hot 0-25 magnitude map; high weights use dark colors."""
    fig = Figure(figsize=(8.27, 11.69), dpi=110)
    ax = fig.add_axes([.10, .11, .70, .77])
    grid = layout_grid(layout, ids, np.abs(values))
    cmap = colormaps['hot_r'].with_extremes(bad=style.MISSING)
    norm = Normalize(*color_limits)
    upper = float(color_limits[1])
    im = ax.imshow(np.ma.masked_invalid(grid), origin='lower', aspect='auto', cmap=cmap,
                   norm=norm, interpolation='nearest')
    cax = fig.add_axes([.84, .11, .025, .77])
    bar = fig.colorbar(im, cax=cax, label=unit, extend='max', ticks=MaxNLocator(nbins=6))
    bar.ax.tick_params(labelsize=6)
    if numeric:
        for r, c in zip(*np.nonzero(np.isfinite(grid))):
            red, green, blue, _ = cmap(norm(grid[r, c]))
            color = 'black' if .2126 * red + .7152 * green + .0722 * blue > .5 else 'white'
            ax.text(c, r, f'{grid[r,c]:.2f}', ha='center', va='center', fontsize=3.5, color=color)
        fig.text(.5, .047, f'Absolute values; multiplier x1; numbers above {upper:g} are retained',
                 fontsize=6, ha='center')
        ax.set_xticks(np.arange(-.5, 8, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 128, 1), minor=True)
        ax.grid(which='minor', color='#DFE3E7', lw=.25)
        ax.tick_params(which='minor', length=0)
    ax.set_xticks(range(8), [str(i) for i in range(1, 9)])
    ax.xaxis.tick_bottom();ax.xaxis.set_label_position('bottom')
    ax.set_xlabel('Shank', fontsize=7)
    ticks = np.r_[0, np.arange(15, 128, 16)]
    ax.set_yticks(ticks, ticks + 1)
    ax.set_ylim(-.5, 127.5)
    ax.set_ylabel('Elec# (1 near fan-in; 128 tip)', fontsize=7)
    ax.tick_params(length=0, labelsize=6)
    label = '|wx|' if component == 'vx' else '|wy|' if component == 'vy' else 'norm(wx, wy)'
    fig.suptitle(f'{label} weighting - {"values" if numeric else "heatmap"}\n{banner}',
                 fontsize=10, y=.978)
    fig.text(.5, .025, f'hot_r: {color_limits[0]:g}-{upper:g}; high = dark, zero = white. Grey = untransmitted; >{upper:g} saturates.\n'
             'Elec# is a position index; magnitude is model sensitivity, not causal contribution.',
             ha='center', fontsize=6)
    return fig


def append_weight_figures(result, layout, rows, banner, unit, method, *, progress=lambda _: None,
                          check_cancel=lambda: None):
    """Shared rendering for newly fitted models and saved-weight redraws."""
    selected = sorted((r for r in rows if r['transmitted']), key=lambda r: r['stored_column_0based'])
    ids = np.array([r['electrode_id'] for r in selected])
    for component in ('vx', 'vy', 'norm'):
        values = np.array([abs(r[component]) for r in selected])
        for numeric in (False, True):
            check_cancel()
            title = f'{component} weighting {"values" if numeric else "heatmap"}'
            progress(f'S14: {title}')
            result.add_figure(title, weight_figure(layout, ids, values, component, numeric, banner, unit),
                              method=method, data={'channel_weights': rows})


class DecoderWeights(Section):
    id = 's14'
    title = '解码权重与电极深度'
    description = '六张解码权重图，加一张按整段录制非零事件通道计数的深度分布图。'
    source_sections = ('s10',)
    params = (
        Param('decoder', '权重模型', 'choice', 'ridge', choices=('ridge', 'velocity_kf')),
        Param('weight_basis', '权重单位', 'choice', 'per_SD', choices=('per_SD', 'per_feature_unit'),
              help='per_SD：输入增加 1 个拟合标准差；per_feature_unit：输入增加 1 Hz 或 1 sqrt(Hz)。'),
        Param('depth_group_size', 'CSV 位置统计分组 / shank', 'int', 16, minimum=1, maximum=128),
        Param('depth_csv', '通道深度 CSV', 'file', str(C.DEFAULT_DEPTH_CSV)),
        Param('active_depth_bin_um', '有效通道深度 bin / µm', 'float', 100., minimum=1.),
        Param('shank1_depth_offset_um', 'Shank1 深度修正 / µm（固定106850）', 'float', 106850.),
    )

    def run(self, ctx):
        layout = ctx.layout
        if layout is None or layout.get('kind') != 'shank_table':
            raise ValueError('S14 需要启用新天枢 3.0 mapping（Channel# / Shank / Elec# 表），请在顶栏选择天枢3.0_ChannelMap.xlsx')
        source = ctx.params['_source_sections']['s10']
        p = source['params']
        spec = spec_of(p)
        ctx.progress('S14: S10 velocity bins and full-run model fit')
        data = D.build_data(ctx.cohort, spec, cache=ctx.cached)
        name, basis = ctx.params['decoder'], ctx.params['weight_basis']
        fitted = ctx.cached(('s14_weights', id(ctx.cohort), spec.key(ctx.criteria), name, basis,
                             tuple(sorted(D.decoder_params(D.get(name), p).items()))),
                            lambda: fit_weights(data, name, p, basis))
        ctx.check_cancel()
        weights = fitted['weights']
        rows = channel_rows(layout, data.electrode_ids, weights, fitted['keep'])
        depth = depth_rows(rows, ctx.params['depth_group_size'])
        unit = 'velocity / input SD' if basis == 'per_SD' else f'velocity / {data.feature_unit}'
        banner = f'{ctx.session.label}\n{D.label(name)}; {basis}; {data.n} bins'
        method = (f'Current run, full-data descriptive fit (not CV or saved online weights). '
                  f'S10 {source["version"]} inputs/parameters: bin={spec.bin_s:g}s, offset={spec.offset:g}s, '
                  f'until={spec.until}, n_bins={spec.n_bins}, sqrt={spec.sqrt}; {data.n} bins, '
                  f'{len(np.unique(data.trial))} trials. {D.get(name).METHOD} '
                  + ('Weights = Ridge coefficients on standardized inputs. ' if name == 'ridge' else
                     f'Weights = scale * steady-state Kalman gain K (not observation C); '
                     f'Riccati recursion starts P=0, converged in {fitted["gain_steps"]} steps; '
                     'transient gains depend on sequence age and are not represented here. ')
                  + ('Displayed per input SD. ' if basis == 'per_SD' else
                     'Divide each retained coefficient by its fitted input SD to display per feature unit. ')
                  + 'norm = sqrt(vx_weight^2 + vy_weight^2). Constant training features have weight zero; '
                  'untransmitted sites are missing/grey. XDF electrode ID minus one joins Channel#; '
                  f'Shank and Elec# come from {layout["source"]}. '
                  'All six plots display absolute values with the same hot_r colormap and fixed 0-25 color limits. '
                  'Values above 25 saturate in color only; numeric labels and CSV retain their magnitudes. '
                  'Numeric labels use multiplier x1 and two decimals. No position-group subplot. '
                  'channel_weights exports vx/vy as absolute values and retains signs in vx_signed/vy_signed. '
                  'Magnitude is model sensitivity, not unique explained variance or an ablation effect. '
                  + D.cohort_rule_text(ctx.cohort, spec))
        result = SectionResult()
        append_weight_figures(result, layout, rows, banner, unit, method,
                              progress=ctx.progress, check_cancel=ctx.check_cancel)
        candidates = [r for r in depth if np.isfinite(r['norm_mean_abs']) and r['norm_mean_abs'] > 0]
        peak = max(candidates, key=lambda r: r['norm_mean_abs']) if candidates else {}
        result.summary.update(recording=ctx.session.label, decoder=name, weight_basis=basis,
                              fit_scope='full_run_descriptive', source_section=source, layout_source=layout['source'],
                              n_bins=int(data.n), n_trials=int(len(np.unique(data.trial))),
                              n_transmitted=len(data.electrode_ids), n_fitted=int(fitted['keep'].sum()),
                              gain_steps=fitted['gain_steps'], model_info=fitted['model_info'],
                              display_absolute=True, color_limits=[0., 25.], colormap='hot_r',
                              peak_elec_start=peak.get('elec_start'), peak_elec_end=peak.get('elec_end'),
                              peak_norm_mean_abs=peak.get('norm_mean_abs'), peak_n_transmitted=peak.get('n_transmitted'))
        result.tables.update(channel_weights=rows, depth_groups=depth)
        result.notes.extend(['按当前 run 的 S10 有效样本重新全量拟合，仅用于描述权重，不是 CV 性能评估。',
                             '默认 per_SD 权重表示输入增加一个拟合标准差的速度敏感度。相关通道会分摊权重；权重大小不能替代删通道后的解码性能变化。',
                             'Elec# 1 靠近 fan-in、128 靠近尖端。没有植入方向和深度标定时，只比较沿杆位点位置，不推断解剖深度。',
                             '灰色为未传输通道，常数通道权重为零。位置分组均值只在已传输位点上计算，需同时查看覆盖数量。'])
        if name == 'velocity_kf':
            result.notes.append('KF 图显示稳态输入增益；实际 trial 开始阶段的增益随状态协方差变化。观测矩阵 C 不作为神经到速度的解码权重。')
        from .depth_distribution import append_distribution
        append_distribution(result,ctx.session,path=ctx.params['depth_csv'],
                            bin_um=ctx.params['active_depth_bin_um'],
                            shank1_offset_um=ctx.params['shank1_depth_offset_um'])
        result.notes.append('深度分布图中的 single unit count 按非零事件通道计数，不代表 spike sorting 后的神经元数。')
        return result


SECTION = DecoderWeights()
