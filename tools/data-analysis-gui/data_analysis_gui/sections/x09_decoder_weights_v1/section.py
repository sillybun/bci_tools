"""Equal-run channelwise median and average; missing electrodes remain missing."""
import numpy as np
from matplotlib.ticker import MaxNLocator

from ... import decoders as D
from ..base import Section, SectionResult
from ..s10_velocity_decoding_v1.section import spec_of
from ..s14_decoder_weights_v1.section import fit_weights, channel_rows, weight_figure


def aggregate_result(layout, runs, labels, unit, *, progress=lambda _: None):
    if not runs:
        raise ValueError('请选择至少一份可用记录')
    cube = np.full((len(runs), 1024, 3), np.nan)
    constant = np.zeros((len(runs), 1024), bool)
    for i, rows in enumerate(runs):
        seen = set()
        for row in rows:
            ch = int(row['channel_0based'])
            if ch in seen or not 0 <= ch < 1024:
                raise ValueError('记录通道编号重复或越界')
            seen.add(ch)
            if row['transmitted']:
                values = np.abs([row['vx'], row['vy'], row['norm']])
                if not np.isfinite(values).all():
                    raise ValueError('已传输通道含非有限权重')
                cube[i, ch] = values
                constant[i, ch] = row['status'] == 'constant_dropped'
    counts = np.isfinite(cube[:, :, 0]).sum(0)
    valid = counts > 0
    median, mean = np.full((1024, 3), np.nan), np.full((1024, 3), np.nan)
    median[valid] = np.nanmedian(cube[:, valid], axis=0)
    mean[valid] = np.nanmean(cube[:, valid], axis=0)
    maximum = max(float(np.max(median[valid])), float(np.max(mean[valid]))) if valid.any() else 0.
    upper = float(MaxNLocator(nbins=6).tick_values(0., maximum)[-1]) if maximum > 0 else 1.
    limits = (0., upper)
    rows = []
    for ch, (r, c) in enumerate(layout['position']):
        row = dict(channel_0based=ch, electrode_id=ch+1, shank=int(c+1), elec_index=int(r+1),
                   n_runs_observed=int(counts[ch]), n_runs_selected=len(runs),
                   n_runs_constant=int(constant[:, ch].sum()),
                   n_runs_fitted=int(counts[ch] - constant[:, ch].sum()))
        for j, component in enumerate(('vx', 'vy', 'norm')):
            row[f'{component}_median'] = float(median[ch, j])
            row[f'{component}_average'] = float(mean[ch, j])
        rows.append(row)
    result = SectionResult()
    method = ('Each selected recording is fitted independently with S14/S10 settings. '
              'Join by Channel# (XDF electrode ID minus 1). Compute per-channel median and arithmetic average '
              'of absolute weights, with equal weight per recording. Untransmitted channels are missing and '
              'excluded; transmitted constant-input channels retain zero weight. norm is computed within '
              'each run before aggregation, not from aggregated wx/wy. No clipping before aggregation. '
              f'hot_r, shared color range 0-{upper:g} across all six plots, covering their global maximum; '
              'high dark, zero white, missing grey; Elec# 1 bottom and 128 top. '
              'Channel coverage and both statistics are exported to CSV. Weight magnitude is sensitivity, '
              'not a held-out ablation effect.')
    for statistic, matrix in (('median', median), ('average', mean)):
        for j, component in enumerate(('vx', 'vy', 'norm')):
            title = f'{component} weighting {statistic}'
            progress(title)
            fig = weight_figure(layout, np.arange(1, 1025), matrix[:, j], component, False,
                                f'{statistic} across {len(runs)} recordings; equal recording weight', unit,
                                color_limits=limits)
            label = '|wx|' if component == 'vx' else '|wy|' if component == 'vy' else 'norm(wx, wy)'
            fig._suptitle.set_text(f'{label} weighting - {statistic}\n{len(runs)} recordings; equal recording weight')
            result.add_figure(title, fig, method=method, data={'channel_statistics': rows})
    per_run = [dict(recording=labels[i], channel_0based=ch, vx=float(cube[i,ch,0]),
                    vy=float(cube[i,ch,1]), norm=float(cube[i,ch,2]))
               for i in range(len(runs)) for ch in range(1024) if np.isfinite(cube[i,ch,0])]
    result.tables.update(channel_statistics=rows, per_run_weights=per_run)
    result.summary.update(recordings=labels, n_recordings=len(runs), n_channels_observed=int(valid.sum()),
                          n_channels_all_runs=int(np.sum(counts == len(runs))),
                          min_coverage=int(counts[valid].min()) if valid.any() else 0,
                          max_coverage=int(counts.max()), color_limits=list(limits), colormap='hot_r')
    result.notes += ['每份所选录制等权；不是按 trial 数、bin 数或天数加权。',
                     '未传输位点排除，已传输但恒定输入的零权重参与统计。每通道实际参与 run 数见 CSV。',
                     '先计算每个 run 的 norm，再计算跨 run 的 median / average。',
                     'Elec# 是沿杆位置序号，不是毫米解剖深度。']
    return result


class CrossDecoderWeights(Section):
    id = 'x09'
    mode = 'cross'
    min_recordings = 1
    title = '解码权重 median / average'
    description = '对勾选 run 独立拟合，按通道等权汇总 |wx|、|wy|、norm 的 median 与 average，共六张图。'
    source_sections = ('s14',)
    cache_code_dependencies = ('sections/s10_velocity_decoding_v1',)

    def run(self, ctx):
        if ctx.layout is None or ctx.layout.get('kind') != 'shank_table':
            raise ValueError('X09 需要启用新天枢 Channel# / Shank / Elec# mapping')
        settings = ctx.params['_source_sections']['s14']['params']
        p = settings['_source_sections']['s10']['params']
        spec = spec_of(p)
        name, basis = settings['decoder'], settings['weight_basis']
        runs, labels = [], []
        if len({s.recording.key for s in ctx.sessions}) != len(ctx.sessions):
            raise ValueError('不能重复选择同一份录制')
        for session, cohort in zip(ctx.sessions, ctx.cohorts):
            ctx.check_cancel()
            ctx.progress(f'X09 fit: {session.label}')
            data = D.build_data(cohort, spec, cache=ctx.cached)
            key = ('s14_weights', id(cohort), spec.key(ctx.criteria), name, basis,
                   tuple(sorted(D.decoder_params(D.get(name), p).items())))
            fitted = ctx.cached(key, lambda: fit_weights(data, name, p, basis))
            runs.append(channel_rows(ctx.layout, data.electrode_ids, fitted['weights'], fitted['keep']))
            labels.append(session.label)
        unit = 'velocity / input SD' if basis == 'per_SD' else ('velocity / sqrt(Hz)' if spec.sqrt else 'velocity / Hz')
        result = aggregate_result(ctx.layout, runs, labels, unit, progress=ctx.progress)
        result.summary.update(decoder=name, weight_basis=basis, source_settings=settings)
        return result


SECTION = CrossDecoderWeights()
