"""Restore the large S05 typography and apply it to the six S14 7shank depth maps."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path('E:/AitBCI/data_analysis_gui_output/keaiduo/single')
S05 = ROOT/'S05/tuning_layout_depth_7shank/summary'
S14 = ROOT/'S14/summary_45runs/figures_depth_7shank'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reference = S05/'style_before_alignment_v1'
    data_files = [S05/'channel_tuning_summary_7shank.csv', S14/'channel_statistics_with_depth.csv']
    original_hashes = {str(p): digest(p) for p in data_files}
    # Restore the exact original S05 PNGs, PDFs and manifest.
    for p in reference.iterdir():
        if p.suffix in ('.png', '.pdf', '.json'):
            shutil.copy2(p, S05/p.name)
            assert digest(p) == digest(S05/p.name)
    for n in range(1, 10000):
        backup = S14/f'style_before_alignment_v{n}'
        if not backup.exists():
            break
    backup.mkdir()
    for p in [*S14.glob('*.png'), *S14.glob('*.pdf'), S14/'7shank_manifest.json']:
        shutil.copy2(p, backup/p.name)
    with (S14/'channel_statistics_with_depth.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 896 and {int(r['shank']) for r in rows} == set(range(2, 9))
    limits = json.loads((S14.parent/'result.json').read_text(encoding='utf-8'))['summary']['color_limits']
    exports = []
    with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11}):
        with PdfPages(S14/'Weighting_Projection_depth_7shank.pdf') as combined:
            for index, (statistic, component) in enumerate(((s, c) for s in ('median', 'average') for c in ('vx', 'vy', 'norm')), 1):
                fig = Figure(figsize=(8.27, 11.69), dpi=110)
                ax = fig.add_axes([.13, .16, .71, .72])
                ax.set_facecolor('#d6d4cf')
                cmap = colormaps['hot_r'].with_extremes(bad='#d6d4cf')
                norm = Normalize(*limits)
                for shank in range(2, 9):
                    group = sorted((r for r in rows if int(r['shank']) == shank), key=lambda r: float(r['depth_um']))
                    centres = np.array([float(r['depth_um']) for r in group])
                    edges = np.r_[centres[0]-25, (centres[:-1]+centres[1:])/2, centres[-1]+25]
                    values = np.array([float(r[f'{component}_{statistic}']) if r[f'{component}_{statistic}'] else np.nan for r in group])
                    np.testing.assert_allclose((edges[:-1]+edges[1:])/2, centres)
                    mesh = ax.pcolormesh([shank-.5, shank+.5], edges, np.ma.masked_invalid(values[:, None]),
                                        cmap=cmap, norm=norm, shading='flat', edgecolors='none', linewidth=0)
                    np.testing.assert_equal(np.asarray(mesh.get_array()).ravel(), values)
                ax.set(xlim=(1.5, 8.5), ylim=(-25, 8500), xticks=range(2, 9), yticks=np.arange(0, 8500, 1000))
                ax.set_xlabel('Shank', fontsize=14)
                ax.set_ylabel('Depth (µm)', fontsize=14)
                ax.tick_params(length=3, labelsize=11)
                ax.spines[['top', 'right']].set_visible(False)
                cax = fig.add_axes([.87, .16, .028, .72])
                cb = fig.colorbar(mesh, cax=cax, ticks=np.arange(0, 13, 2))
                cb.set_label('velocity / input SD', fontsize=12)
                cb.ax.tick_params(labelsize=11)
                label = {'vx': '|wx|', 'vy': '|wy|', 'norm': 'norm(wx, wy)'}[component]
                fig.suptitle(f'{label} weighting - {statistic}\n45 recordings; equal recording weight | 7shank', fontsize=18, y=.96)
                fig.text(.5, .075, 'Grey = no site at that depth or no observed weight.', ha='center', fontsize=11)
                fig.text(.5, .045, '7shank: Shank 2-8; Shank 1 excluded.', ha='center', fontsize=11)
                stem = S14/f'X09_{index:02d}_{component}_weighting_{statistic}_depth_7shank'
                fig.savefig(stem.with_suffix('.png'), dpi=300, facecolor='white')
                fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
                combined.savefig(fig, facecolor='white')
                exports.append(stem.name)
                fig.clear()
    for p in data_files:
        assert digest(p) == original_hashes[str(p)]
    manifest = dict(reference=str(reference/'S05_direction_tuning_mean_depth_7shank.pdf'),
        direction='S14 matches the original large S05 style; original S05 restored exactly',
        canvas_inches=[8.27, 11.69], png_pixels=[2481, 3507], png_dpi=300,
        axes_bounds=[.13, .16, .71, .72], colorbar_bounds=[.87, .16, .028, .72],
        title_pt=18, axis_label_pt=14, tick_label_pt=11, colorbar_label_pt=12, colorbar_tick_pt=11,
        footer_pt=11, backup=str(backup), color_limits=limits, data_sha256=original_hashes,
        figures=exports, calculation='Styling only; no recomputation of weights, tuning or depths')
    (S14/'style_alignment_to_s05.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Restored S05; updated six S14 PNG/PDF figures and combined PDF.')


if __name__ == '__main__':
    main()
