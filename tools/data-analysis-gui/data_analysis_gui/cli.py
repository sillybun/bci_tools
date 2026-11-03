"""Run sections without the GUI, through the same cache-first storage as the GUI.

    uv run --locked python -m data_analysis_gui.cli --list
    uv run --locked python -m data_analysis_gui.cli --section s01 --match 20261026
    uv run --locked python -m data_analysis_gui.cli --section x04 --match 20261029 --match 20261030 --recompute
    uv run --locked python -m data_analysis_gui.cli --section s05 --param s05:alpha=0.01 --match 20261026
    uv run --locked python -m data_analysis_gui.cli --synthetic --out .local/synth      # no data needed

Results go to --output (default: parallel to --data, e.g. E:/AitBCI/data/keaiduo ->
E:/AitBCI/data_analysis_gui_output/keaiduo), in the GUI's folder layout:
figures as PNG/SVG/PDF plus a CSV with the plotted values and calculation method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback

from . import config as C
from .core.catalog import scan
from .core.filters import TrialFilter
from .sections import LOAD_ERRORS, registry, sections


def _param_filter(raw_params):
    def params_for(section):
        known = {p.name for p in section.params}
        out = {}
        for name, value in raw_params:
            target, _, name = name.rpartition(':')
            if (not target or target == section.id) and name in known:
                out[name] = value
        if section.source_sections:
            from .sections import versions
            out['_source_sections'] = {sid: dict(version=versions(sid)[-1].version,
                                                 params=params_for(versions(sid)[-1]))
                                       for sid in section.source_sections}
        return out
    return params_for


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', action='store_true', help='list sections and recordings')
    ap.add_argument('--section', action='append', default=[], help='section id, e.g. s01 (repeatable; default all)')
    ap.add_argument('--data', default=str(C.DEFAULT_DATA_ROOT))
    ap.add_argument('--match', action='append', default=[], help='substring of the recording path (repeatable)')
    ap.add_argument('--param', action='append', default=[],
                    help='name=value for every section that has it, or s05:name=value for one section')
    ap.add_argument('--filter', default='{}', help='JSON TrialFilter overrides')
    ap.add_argument('--layout', default=str(C.DEFAULT_LAYOUT_XLSX))
    ap.add_argument('--no-layout', action='store_true',
                    help='ignore the workbook; place electrodes by index on the 128 x 8 grid')
    ap.add_argument('--output', default=None,
                    help='result folder (default: parallel to --data, <...>/data/<x> -> <...>/data_analysis_gui_output/<x>)')
    ap.add_argument('--recompute', action='store_true', help='ignore stored results')
    ap.add_argument('--synthetic', action='store_true', help='synthetic sessions; writes to --out without caching')
    ap.add_argument('--out', default=str(C.LOCAL_DIR / 'synthetic'), help='folder for --synthetic results')
    args = ap.parse_args(argv)
    if args.output is None:
        args.output = str(C.default_output_root(args.data))
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass
    registry()
    for name, error in LOAD_ERRORS.items():
        print(f'[load error] {name}\n{error}', file=sys.stderr)
    if args.list:
        for mode in ('single', 'cross'):
            for s in sections(mode):
                print(f'{s.id} {s.version} [{mode}] {s.title}')
        for r in scan(args.data)[0]:
            print(('  ' if not r.reason else 'X ') + r.label, '|', r.relative, '|', r.reason)
        return 0
    criteria = TrialFilter.from_dict(json.loads(args.filter))
    params_for = _param_filter([p.split('=', 1) for p in args.param])
    chosen = [s for mode in ('single', 'cross') for s in sections(mode)
              if not args.section or 'all' in args.section or s.id in args.section]
    if args.synthetic:
        return _synthetic(chosen, criteria, params_for, args)

    recordings = [r for r in scan(args.data)[0]
                  if not r.reason and (not args.match or any(m in r.relative for m in args.match))]
    if not recordings:
        print('no matching recordings', file=sys.stderr)
        return 2
    jobs = []
    for s in chosen:
        if s.mode == 'cross':
            if len(recordings) >= s.min_recordings:
                jobs.append(dict(section=s.id, version=s.version, params=params_for(s),
                                 recordings=[str(r.path) for r in recordings]))
            continue
        target = recordings[:1]
        if 'online' in s.requires:
            target = [r for r in recordings if r.online][:1]
            if not target:
                print(f'{s.id} skipped: no online recording among matches')
                continue
        if 'channel_config' in s.requires and not target[0].channel_config:
            print(f'{s.id} skipped: no channel config INI / Neuralinsight LOG for {target[0].path}')
            continue
        jobs.append(dict(section=s.id, version=s.version, params=params_for(s), recordings=[str(r.path) for r in target]))
    request = dict(output_root=args.output, data_root=args.data, criteria=criteria.to_dict(), layout_path='' if args.no_layout else args.layout,
                   recompute=args.recompute, jobs=jobs)
    from . import worker
    return worker.run(request)


def _synthetic(chosen, criteria, params_for, args) -> int:
    from .core.synthetic import make_session
    from .runner import build_context, run_section
    from . import storage
    sessions = [make_session(seed=i, date=f'2026-01-0{i + 1}', online=(i == 0)) for i in range(3)]
    config = Path(args.out) / 'configs.ini'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('good="' + ','.join(map(str, range(300))) + '"\nuncertain="' + ','.join(map(str, range(300, 400)))
                      + '"\nbad="' + ','.join(map(str, range(400, 1024))) + '"\n')
    status, cache = 0, {}
    for s in chosen:
        use = sessions if s.mode == 'cross' else sessions[:1]
        ctx = build_context(use, criteria, params_for(s), s, '' if args.no_layout else args.layout,
                            str(config), cache)
        start = time.time()
        try:
            result = run_section(s, ctx)
        except Exception as exc:
            status = 1
            print(f'{s.id} FAILED: {exc}')
            traceback.print_exc()
            continue
        inputs = dict(format=storage.FORMAT_VERSION, section=s.id, version=s.version, code='synthetic',
                      recordings=[dict(name=x.recording.path.name, sha256='synthetic') for x in use],
                      criteria=criteria.to_dict(), params=ctx.params)
        folder = Path(args.out) / f'{s.id}_{s.version}'
        storage.save_result(result, folder, 'synthetic', inputs, s, [x.recording for x in use], time.time() - start)
        print(f'{s.id} ok in {time.time() - start:.1f}s · {len(result.figures)} figures · {folder}')
    return status


if __name__ == '__main__':
    sys.exit(main())
