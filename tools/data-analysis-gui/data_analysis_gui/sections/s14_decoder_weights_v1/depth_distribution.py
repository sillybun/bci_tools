"""Active-channel depth histogram; one channel contributes at most one count/run."""
import csv
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from ...core.channels import read_layout

TITLE = 'Active channel depth distribution'


def load_depths(path, shank1_offset_um=106850.):
    layout = read_layout()
    if layout is None or layout.get('kind') != 'shank_table':
        raise ValueError('深度映射需要最新天枢 Shank 表')
    depths = {}
    with Path(path).open(encoding='utf-8-sig') as f:
        for row_index, row in enumerate(csv.DictReader(f)):
            for shank in range(1,9):
                ch = int(row[f'shank{shank}_ID'])
                csv_channel = ch
                elec_row, plot_column = map(int, layout['position'][ch])
                if plot_column + 1 != shank:
                    raise ValueError('CSV Shank 与最新 mapping 不一致')
                raw = float(row[f'shank{shank}_depth_um'])
                # Confirmed convention: latest Shank1 starts at zero after this fixed correction.
                offset = 106850. if shank == 1 else 0.
                if ch in depths or not np.isfinite(raw + offset):
                    raise ValueError('深度 CSV 通道重复或深度非有限')
                depths[ch] = dict(channel_0based=ch, csv_channel_0based=csv_channel,
                                  csv_shank=shank, shank=plot_column+1, elec_index=elec_row+1, depth_original_um=raw,
                                  depth_offset_um=offset, depth_um=raw+offset)
    if set(depths) != set(range(1024)):
        raise ValueError('深度 CSV 必须完整覆盖 0–1023 通道')
    return depths


def distribution(ids, maxima, depths, bin_um):
    ids = np.asarray(ids, int)
    maxima = np.asarray(maxima)
    if len(ids) != len(set(ids)) or maxima.shape != ids.shape or np.any(~np.isfinite(maxima)):
        raise ValueError('最大 code 与传输电极 ID 不匹配')
    width = float(bin_um)
    if not np.isfinite(width) or width <= 0:
        raise ValueError('深度分箱宽度必须大于 0')
    all_depth = np.array([d['depth_um'] for d in depths.values() if d['depth_um'] >= 0])
    start = np.floor(all_depth.min()/width)*width
    stop = (np.floor(all_depth.max()/width)+1)*width
    edges = np.arange(start,stop+width*.5,width)
    channel_rows = []
    for e, code in zip(ids, maxima):
        depth = depths.get(int(e)-1)
        if depth is None:
            raise ValueError(f'深度 CSV 缺少电极 ID {e}')
        if depth['depth_um'] < 0:
            continue
        channel_rows.append(dict(depth, electrode_id=int(e), max_code=int(code), active=bool(code>0)))
    active_depth = [r['depth_um'] for r in channel_rows if r['active']]
    counts, _ = np.histogram(active_depth, bins=edges)
    transmitted, _ = np.histogram([r['depth_um'] for r in channel_rows],bins=edges)
    assert int(counts.sum()) == sum(r['active'] for r in channel_rows)
    bins = [dict(depth_left_um=float(a), depth_right_um=float(b), depth_center_um=float((a+b)/2),
                 single_unit_count=int(n), n_transmitted=int(t))
            for a,b,n,t in zip(edges[:-1],edges[1:],counts,transmitted)]
    return bins, channel_rows


def figure(bins, label, *, y_max=None):
    fig = Figure(figsize=(9,6),dpi=110,layout='constrained')
    ax = fig.subplots()
    x = np.array([r['depth_center_um'] for r in bins])
    y = np.array([r['single_unit_count'] for r in bins])
    ax.plot(x,y,color='#0072BD',lw=2.2,label=f'n = {int(y.sum())}')
    ax.set_xlabel('Depth (\u00b5m)',fontsize=12)
    ax.set_ylabel('single unit count',fontsize=12)
    ax.set_xlim(bins[0]['depth_left_um'],bins[-1]['depth_right_um'])
    ax.set_ylim(0, y_max if y_max is not None else max(1.,float(y.max())*1.15))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis='y',alpha=.15)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False)
    ax.set_title(f'Active channel depth distribution\n{label}',fontsize=13)
    return fig


def append_distribution(result, session, *, path, bin_um=100., shank1_offset_um=106850.):
    depths = load_depths(path,shank1_offset_um)
    if not len(session.counts):
        raise ValueError('本记录没有可统计的神经帧')
    maxima = np.max(session.counts,axis=0)
    bins, channels = distribution(session.electrode_ids,maxima,depths,bin_um)
    n_active = sum(r['active'] for r in channels)
    result.add_figure(TITLE,figure(bins,session.label),method=(
        f'Whole loaded recording, {len(session.counts)} received frames; no trial selection or model-coefficient threshold. '
        'Each transmitted channel contributes 1 if any frame has code > 0, otherwise 0; codes 1/2/3 count equally. '
        f'Channel# = XDF electrode ID - 1, joined to {path}. Latest Shank1 +106850 um (0..6350 um), included. '
        f'Depth histogram: {bin_um:g} um left-closed/right-open bins, equal depth ranges across runs. '
        'single unit count is the requested display label for active-channel count, not spike-sorted neuron count.'),
        data={'depth_bins':bins,'active_channels':channels})
    result.tables.update(active_depth_bins=bins,active_depth_channels=channels)
    result.summary.update(active_depth_channels=n_active,active_depth_bin_um=bin_um,
                          active_depth_scope='whole_loaded_recording',active_depth_source=str(path),
                          active_depth_shank1_offset_um=106850,
                          active_depth_exclusion='none; latest Shank1 corrected +106850 um')
