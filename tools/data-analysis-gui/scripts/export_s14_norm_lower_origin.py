"""Export an additional norm PNG set with Elec# 1 at the bottom (no refit)."""
from pathlib import Path
import csv
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import storage
from data_analysis_gui.core.channels import read_layout
from data_analysis_gui.core.style import finish_figure
from data_analysis_gui.sections.s14_decoder_weights_v1.section import weight_figure


def main():
    root = Path('E:/AitBCI/data_analysis_gui_output/keaiduo')
    source = json.loads((root / 's14_hot_r_0_25_redraw.json').read_text(encoding='utf-8'))
    out = root / 'single' / 'S14_05_norm_weighting_heatmap_origin_lower'
    out.mkdir(exist_ok=True)
    outputs = []
    for index, item in enumerate(source['completed'], 1):
        folder = Path(item['output'])
        entry = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        if storage.load_cached(folder, entry['signature']) is None:
            raise ValueError(f'Source verification failed: {folder}')
        summary = entry['summary']
        layout = read_layout(summary['layout_source'])
        with (folder / 'tables' / 'channel_weights.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(line for line in f if not line.startswith('#')))
        rows = sorted((r for r in rows if r['transmitted'] == 'True'),
                      key=lambda r: int(r['stored_column_0based']))
        ids = np.array([int(r['electrode_id']) for r in rows])
        values = np.array([float(r['norm']) for r in rows])
        unit = 'velocity / input SD' if summary['weight_basis'] == 'per_SD' else (
            'velocity / sqrt(Hz)' if summary['source_section']['params']['sqrt'] else 'velocity / Hz')
        banner = f'{summary["recording"]}\n{summary["decoder"]}; {summary["weight_basis"]}; {summary["n_bins"]} bins'
        fig = weight_figure(layout, ids, values, 'norm', False, banner, unit)
        ax, bar = fig.axes
        ax.set_ylim(-.5, 127.5)
        ax.xaxis.tick_bottom()
        ax.xaxis.set_label_position('bottom')
        ax.set_position([.10, .11, .70, .77])
        bar.set_position([.84, .11, .025, .77])
        ax.set_ylabel('Elec# (bottom 1 near fan-in; top 128 tip)', fontsize=7)
        assert ax.get_ylim() == (-.5, 127.5)
        grid = ax.images[0].get_array()
        for e, value in zip(ids, values):
            r, c = layout['position'][e - 1]
            assert grid[r, c] == value
        assert ax.images[0].get_clim() == (0., 25.)
        assert ax.images[0].cmap.name == 'hot_r'
        finish_figure(fig, entry['inputs']['font_percent'])
        name = Path(entry['inputs']['recordings'][0]['name']).stem
        date = folder.parents[1].name[:10]
        if not name.startswith(date):
            name = date + '_' + name
        target = out / (name + '.png')
        fig.savefig(target, dpi=storage.PNG_DPI, facecolor='white', bbox_inches='tight')
        fig.clear()
        outputs.append(dict(source=str(folder), png=str(target)))
        print(f'[{index}/{len(source["completed"])}] {target.name}', flush=True)
    (out / 'export_manifest.json').write_text(json.dumps(dict(
        orientation='Shank 1 left; Elec# 1 bottom, 128 top', colormap='hot_r', color_limits=[0,25],
        outputs=outputs), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'DONE: {len(outputs)} PNGs, {out}', flush=True)


if __name__ == '__main__':
    main()
