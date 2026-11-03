"""Re-render saved S14 coefficients with current plotting rules, without fitting again.

Writes normal cache entries under new signatures, preserving the original results.
Only accepts artifacts with the same S10/core/decoder dependency hash.
Mapping is rejoined by channel identity, so updated coordinates need no refit.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import config as C, storage
from data_analysis_gui.core.catalog import describe
from data_analysis_gui.core.channels import read_layout
from data_analysis_gui.core.filters import TrialFilter
from data_analysis_gui.core.style import finish_figure
from data_analysis_gui.sections import versions
from data_analysis_gui.sections.base import SectionResult
from data_analysis_gui.sections.result_summaries import result_sentence
from data_analysis_gui.sections.s14_decoder_weights_v1.section import (
    append_weight_figures, channel_rows, depth_rows)


def read_rows(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=C.DEFAULT_DATA_ROOT)
    ap.add_argument('--output', type=Path, default=C.default_output_root(C.DEFAULT_DATA_ROOT))
    args = ap.parse_args()
    section = versions('s14')[-1]
    code = storage.source_digest(storage.section_folder(versions('s10')[-1]))
    lookup = {}
    for path in args.data.rglob('*.xdf'):
        lookup.setdefault(path.name, []).append(path)
    # Snapshot before writing any new cache entries; keep the newest saved configuration.
    selected = {}
    for path in args.output.glob('single/*/s14_v1/*/result.json'):
        entry = json.loads(path.read_text(encoding='utf-8'))
        old = entry['inputs']
        key = storage.fingerprint([old['recordings'], old['params'], old['criteria'], old['font_percent']])
        if key not in selected or entry['created'] > selected[key][1]['created']:
            selected[key] = (path, entry)
    completed, skipped = [], []
    for source, entry in selected.values():
        result = SectionResult()
        try:
            old = entry['inputs']
            if storage.load_cached(source.parent, entry['signature']) is None:
                raise ValueError('source cache digest verification failed')
            if old['source_sections']['s10']['code'] != code:
                raise ValueError('S10/core/decoder code differs from the saved fit')
            paths = lookup.get(old['recordings'][0]['name'], [])
            if len(paths) != 1:
                raise ValueError('source recording cannot be resolved uniquely')
            recording = describe(paths[0], args.data)
            summary = dict(entry['summary'])
            layout = read_layout(summary['layout_source'])
            if layout is None or layout.get('kind') != 'shank_table':
                raise ValueError('new shank mapping unavailable')
            params = section.resolve_params(old['params'])
            criteria = TrialFilter.from_dict(old['criteria'])
            inputs = storage.signature_inputs(section, [recording], criteria, params, layout['source'],
                                              font_percent=old['font_percent'])
            if inputs['recordings'] != old['recordings']:
                raise ValueError('source recording content differs from the saved fit')
            signature = storage.fingerprint(inputs)
            destination = storage.result_folder(args.output, section, [recording], signature)
            if storage.load_cached(destination, signature) is not None:
                skipped.append(dict(source=str(source), status='already_current', output=str(destination)))
                continue
            raw = read_rows(source.parent / 'tables' / 'channel_weights.csv')
            transmitted = sorted((r for r in raw if r['transmitted'] == 'True'),
                                 key=lambda r: int(r['stored_column_0based']))
            ids = np.array([int(r['electrode_id']) for r in transmitted])
            weights = np.array([[float(r.get('vx_signed', r['vx'])), float(r.get('vy_signed', r['vy']))]
                                for r in transmitted])
            keep = np.array([r['status'] == 'fitted' for r in transmitted])
            rows = channel_rows(layout, ids, weights, keep)
            if len(raw) != 1024 or len({int(r['channel_0based']) for r in raw}) != 1024 or len(ids) != summary['n_transmitted']:
                raise ValueError('saved channel table coverage mismatch')
            np.testing.assert_allclose(np.hypot(weights[:, 0], weights[:, 1]),
                                       [float(r['norm']) for r in transmitted], rtol=2e-8, atol=1e-8)
            depth = depth_rows(rows, params['depth_group_size'])
            peaks = [r for r in depth if np.isfinite(r['norm_mean_abs']) and r['norm_mean_abs'] > 0]
            peak = max(peaks, key=lambda r: r['norm_mean_abs']) if peaks else {}
            summary.update(display_absolute=True, color_limits=[0., 25.], colormap='hot_r',
                           weight_redraw_source=str(source),
                           weight_redraw_source_layout_digest=old['layout'],
                           peak_elec_start=peak.get('elec_start'), peak_elec_end=peak.get('elec_end'),
                           peak_norm_mean_abs=peak.get('norm_mean_abs'), peak_n_transmitted=peak.get('n_transmitted'),
                           weight_redraw_precision='saved CSV coefficients (10 significant digits)')
            result.summary.update(summary)
            result.summary['one_sentence_summary'] = result_sentence('s14', summary)
            result.tables.update(channel_weights=rows, depth_groups=depth)
            method = entry['figures'][0]['method']
            method = re.sub(r'Position groups of .*?coverage n\. ', '', method)
            method = method.replace('vx/vy share symmetric color limits; norm uses zero to its maximum. ', '')
            method = method.replace('hot colormap', 'hot_r colormap')
            method += (' Display: |wx|, |wy|, norm; hot_r colormap (high dark, low light), fixed 0-25 on all six plots, no position-group '
                       'subplot. Above 25 only the color saturates; numbers/CSV remain untruncated. '
                       'Numeric multiplier x1. vx/vy CSV fields are absolute; vx_signed/vy_signed retain signs. '
                       f'Redrawn from saved CSV coefficients without refitting: {source}.')
            unit = 'velocity / input SD' if summary['weight_basis'] == 'per_SD' else (
                'velocity / sqrt(Hz)' if summary['source_section']['params']['sqrt'] else 'velocity / Hz')
            banner = f'{summary["recording"]}\n{summary["decoder"]}; {summary["weight_basis"]}; {summary["n_bins"]} bins'
            append_weight_figures(result, layout, rows, banner, unit, method)
            result.notes = [note for note in entry['notes'] if '位置分组均值' not in note
                            and '色条固定' not in note and '零权重为黑色' not in note]
            result.notes += ['本次复用已保存 CSV 系数重画，未重新拟合。',
                             '六图均显示绝对值，hot_r 色条固定 0–25，高值深色、低值浅色；超过 25 仅颜色饱和，数值保持原值。',
                             '灰色为未传输；已传输但零权重为白色，具体状态见 CSV。']
            if inputs['layout'] != old['layout']:
                result.notes.append('mapping 文件已更新：用 Channel# 重新关联当前 Shank / Elec#，未使用旧 CSV 中的位置。')
            for _, figure in result.figures:
                finish_figure(figure, old['font_percent'])
            storage.save_result(result, destination, signature, inputs, section, [recording])
            completed.append(dict(source=str(source), output=str(destination)))
            print(f'[{len(completed)}/{len(selected)}] {recording.path.name}', flush=True)
        except Exception as exc:
            skipped.append(dict(source=str(source), error=str(exc)))
            print(f'ERROR {source}: {exc}', flush=True)
        finally:
            for _, figure in result.figures:
                figure.clear()
    manifest = args.output / 's14_hot_r_0_25_redraw.json'
    manifest.write_text(json.dumps(dict(completed=completed, skipped=skipped), ensure_ascii=False, indent=2),
                        encoding='utf-8')
    print(f'DONE: {len(completed)} redrawn, {len(skipped)} skipped; {manifest}', flush=True)
    return int(any('error' in row for row in skipped))


if __name__ == '__main__':
    raise SystemExit(main())
