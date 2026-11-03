"""Collect all six S14 PNG types by run; aggregate the saved 45-run coefficients."""
from pathlib import Path
import csv
import json
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import config as C, storage
from data_analysis_gui.core.channels import read_layout
from data_analysis_gui.core.style import finish_figure
from data_analysis_gui.sections.s14_decoder_weights_v1.section import channel_rows, append_weight_figures
from data_analysis_gui.sections.x09_decoder_weights_v1.section import aggregate_result, SECTION


def main():
    root = Path('E:/AitBCI/data_analysis_gui_output/keaiduo')
    out = root / 'single' / ('S14_new_mapping_build' if '--staging' in sys.argv else 'S14')
    out.mkdir(exist_ok=True)
    manifest = json.loads((root / 's14_hot_r_0_25_redraw.json').read_text(encoding='utf-8'))
    runs, labels, sources = [], [], []
    common = None
    filenames = set()
    for i, job in enumerate(manifest['completed'], 1):
        folder = Path(job['output'])
        entry = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        if storage.load_cached(folder, entry['signature']) is None:
            raise ValueError(f'Incomplete source: {folder}')
        s = entry['summary']
        config = dict(decoder=s['decoder'], basis=s['weight_basis'],
                      params=s['source_section'], criteria=entry['inputs']['criteria'])
        if common is not None and common != config:
            raise ValueError('Selected results have inconsistent fit parameters or units')
        common = config
        layout = read_layout(str(C.DEFAULT_LAYOUT_XLSX))
        if layout is None:
            raise ValueError('Current default mapping is missing')
        with (folder / 'tables/channel_weights.csv').open(encoding='utf-8-sig') as f:
            raw = list(csv.DictReader(line for line in f if not line.startswith('#')))
        selected = sorted((r for r in raw if r['transmitted'] == 'True'),
                          key=lambda r: int(r['stored_column_0based']))
        ids = np.array([int(r['electrode_id']) for r in selected])
        weights = np.array([[float(r['vx_signed']), float(r['vy_signed'])] for r in selected])
        keep = np.array([r['status'] == 'fitted' for r in selected])
        rows = channel_rows(layout, ids, weights, keep)
        runs.append(rows); labels.append(s['recording'])
        unit = 'velocity / input SD' if s['weight_basis'] == 'per_SD' else (
            'velocity / sqrt(Hz)' if s['source_section']['params']['sqrt'] else 'velocity / Hz')
        from data_analysis_gui.sections.base import SectionResult
        result = SectionResult()
        append_weight_figures(result, layout, rows,
                              f'{s["recording"]}\n{s["decoder"]}; {s["weight_basis"]}; {s["n_bins"]} bins', unit,
                              'Redisplay saved coefficients; Elec# 1 bottom, 128 top.')
        xdf = Path(entry['inputs']['recordings'][0]['name']).stem
        name = folder.parents[1].name[:10] + '_' + xdf
        if name in filenames:
            raise ValueError('Duplicate recording name: ' + name)
        filenames.add(name)
        for n, (title, fig) in enumerate(result.figures, 1):
            if '--summary-only' in sys.argv:
                fig.clear()
                continue
            typ = f'S14_{n:02d}_' + title.replace(' ', '_')
            target = out / typ
            target.mkdir(exist_ok=True)
            finish_figure(fig, entry['inputs']['font_percent'])
            # Render to a temporary image, then COPY to the organized collection.
            temp = Path(__file__).resolve().parents[1] / 'outputs' / 's14_collection_rendered' / typ
            temp.mkdir(parents=True, exist_ok=True)
            source_png = temp / (name + '.png')
            fig.savefig(source_png, dpi=storage.PNG_DPI, facecolor='white', bbox_inches='tight')
            dest = target / source_png.name
            shutil.copy2(source_png, dest)
            if storage.file_digest(source_png) != storage.file_digest(dest):
                raise ValueError('Copy verification failed')
            fig.clear()
        sources.append(dict(run=s['recording'], source=str(folder), filename=name+'.png'))
        print(f'[{i}/{len(manifest["completed"])}] {name}', flush=True)
    result = aggregate_result(layout, runs, labels, unit)
    result.summary.update(decoder=common['decoder'], weight_basis=common['basis'], source_settings=common)
    for _, fig in result.figures:
        finish_figure(fig, 120)
    # Preserve source identity and settings beside the six summary plots.
    inputs = dict(format=storage.FORMAT_VERSION, section='x09', version='v1',
                  recordings=[], criteria=common['criteria'], params=common,
                  source_results=sources, fit_configuration=common,
                  layout=storage.file_digest(layout['source']), code=storage.source_digest(storage.section_folder(SECTION)))
    sig = storage.fingerprint(inputs)
    summary_dir = out / 'summary_45runs'
    # Merge only generated files; preserve any user-added documents in the output folder.
    with TemporaryDirectory(prefix='s14_summary_') as temporary:
        rendered = Path(temporary) / 'summary'
        storage.save_result(result, rendered, sig, inputs, SECTION, [])
        shutil.copytree(rendered, summary_dir, dirs_exist_ok=True)
    for _, fig in result.figures:
        fig.clear()
    (out/'collection_manifest.json').write_text(json.dumps(dict(recordings=sources,
        n_recordings=len(runs), n_pngs=len(runs)*6, orientation='Shank 1 left, 8 right; Elec# 1 bottom, 128 top',
        layout_source=layout['source'], layout_digest=storage.file_digest(layout['source']),
        summary=str(summary_dir)), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'DONE: {len(runs)*6} PNG copies; 6 summary figures. {out}', flush=True)


if __name__ == '__main__':
    main()
