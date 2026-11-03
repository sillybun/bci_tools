"""Replot saved channel statistics at calibrated depths, without interpolation or refitting."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import config as C, storage
from data_analysis_gui.sections.s14_decoder_weights_v1.depth_distribution import load_depths

ROOT = Path('E:/AitBCI/data_analysis_gui_output/keaiduo/single/S14/summary_45runs')
DEPTH = C.DEFAULT_DEPTH_CSV


def read_rows(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(row for row in f if not row.startswith('#')))


def joined_rows():
    depth = load_depths(DEPTH)
    if set(depth) != set(range(1024)):
        raise ValueError('Depth CSV must contain channels 0-1023 exactly once')
    rows = []
    for raw in read_rows(ROOT/'tables/channel_statistics.csv'):
        ch = int(raw['channel_0based'])
        row = dict(raw)
        row.update(depth[ch])
        row['channel_0based'] = ch
        row['shank'] = int(raw['shank'])
        row['n_runs_observed'] = int(raw['n_runs_observed'])
        for stat in ('median','average'):
            for component in ('vx','vy','norm'):
                key = f'{component}_{stat}'
                row[key] = float(raw[key]) if raw[key] else np.nan
        rows.append(row)
    assert len(rows) == 1024
    for shank in range(1,9):
        group = [r for r in rows if r['shank']==shank]
        if not group:
            continue
        assert len(group)==128 and len({r['csv_shank'] for r in group})==1
        depths = np.sort([r['depth_um'] for r in group])
        np.testing.assert_allclose(np.diff(depths),50)
    return rows


def depth_figure(rows, component, statistic, limits):
    fig = Figure(figsize=(8.27,11.69),dpi=110)
    ax = fig.add_axes([.12,.11,.68,.77])
    ax.set_facecolor('#d6d4cf')
    cmap = colormaps['hot_r'].with_extremes(bad='#d6d4cf')
    norm = Normalize(*limits)
    key = f'{component}_{statistic}'
    for shank in range(1,9):
        group = sorted((r for r in rows if r['shank']==shank),key=lambda r:r['depth_um'])
        if not group:
            continue
        centres = np.array([r['depth_um'] for r in group])
        edges = np.r_[centres[0]-25,(centres[:-1]+centres[1:])/2,centres[-1]+25]
        values = np.array([r[key] for r in group])[:,None]
        mesh = ax.pcolormesh([shank-.5,shank+.5],edges,np.ma.masked_invalid(values),
                             cmap=cmap,norm=norm,shading='flat',edgecolors='none',linewidth=0)
        # Every cell's centre must be its CSV depth, not an electrode-index rescaling.
        np.testing.assert_allclose((edges[:-1]+edges[1:])/2,centres)
    ax.set_xlim(.5,8.5)
    ax.set_ylim(-25,8500)
    ax.set_xticks(range(1,9))
    ax.set_yticks(np.arange(0,8500,1000))
    ax.set_xlabel('Shank',fontsize=10)
    ax.set_ylabel('Depth (\u00b5m)',fontsize=10)
    ax.tick_params(length=3,labelsize=8)
    ax.spines[['top','right']].set_visible(False)
    cax=fig.add_axes([.84,.11,.025,.77])
    fig.colorbar(mesh,cax=cax,ticks=np.arange(0,13,2),label='velocity / input SD')
    label='|wx|' if component=='vx' else '|wy|' if component=='vy' else 'norm(wx, wy)'
    fig.suptitle(f'{label} weighting - {statistic}\n45 recordings; equal recording weight',fontsize=13,y=.963)
    fig.text(.5,.045,'Grey = no site at that depth or no observed weight.',ha='center',fontsize=8)
    fig.text(.5,.025,'Shank1 depth correction: +106850 um (0-6350 um).',ha='center',fontsize=8)
    return fig


def main():
    original_digest=storage.file_digest(DEPTH)
    rows=joined_rows()
    metadata=json.loads((ROOT/'result.json').read_text(encoding='utf-8'))
    limits=metadata['summary']['color_limits']
    out=ROOT/'figures_depth'
    out.mkdir(exist_ok=True)
    exported=[]
    with PdfPages(out/'Weighting_Projection_depth.pdf') as combined:
        for index,(stat,component) in enumerate(((s,c) for s in ('median','average')
                                               for c in ('vx','vy','norm')),1):
            stem=f'X09_{index:02d}_{component}_weighting_{stat}_depth'
            fig=depth_figure(rows,component,stat,limits)
            FigureCanvasAgg(fig).draw()
            fig.savefig(out/(stem+'.png'),dpi=300,facecolor='white')
            fig.savefig(out/(stem+'.pdf'),facecolor='white')
            combined.savefig(fig,facecolor='white')
            fig.clear()
            exported.append(stem)
            print(stem,flush=True)
    with (out/'channel_statistics_with_depth.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    assert storage.file_digest(DEPTH)==original_digest
    (out/'depth_manifest.json').write_text(json.dumps(dict(source_depth=str(DEPTH),
        source_depth_sha256=original_digest,source_weights=str(ROOT/'tables/channel_statistics.csv'),
        color_limits=limits,unit='um',csv_shank1_offset_um=106850,excluded_csv_shank=None,
        x_order='Latest mapping Shank1..8; joined by Channel ID',
        excluded_plot_shank=None,
        y_order='Depth increases upward; lower origin',
        csv_shank_to_plot_shank={str(s):sorted({r['shank'] for r in rows if r['csv_shank']==s})
                                 for s in range(1,9)},
        depth_range_um=[min(r['depth_um'] for r in rows),max(r['depth_um'] for r in rows)],
        figures=exported),ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'DONE: {out}',flush=True)


if __name__=='__main__':
    main()
