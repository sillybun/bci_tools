"""Export the seventh S14 figure for the same 45 recordings, reusing verified whole-run code counts."""
from pathlib import Path
import csv
import json
import sys

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui import config as C, storage
from data_analysis_gui.core.session import Session
from data_analysis_gui.core.catalog import describe
from data_analysis_gui.sections.s14_decoder_weights_v1.depth_distribution import load_depths,distribution,figure


def read_csv(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))


def main():
    root=Path('E:/AitBCI/data_analysis_gui_output/keaiduo')
    source=json.loads((root/'s14_hot_r_0_25_redraw.json').read_text(encoding='utf-8'))
    depths=load_depths(C.DEFAULT_DEPTH_CSV)
    depth_sha=storage.file_digest(C.DEFAULT_DEPTH_CSV)
    candidates={}
    for p in root.glob('single/*/s02_v1/*/result.json'):
        entry=json.loads(p.read_text(encoding='utf-8'))
        if entry['summary'].get('scope')=='whole_recording' and entry['summary'].get('n_frames',0)>0:
            sha=entry['inputs']['recordings'][0]['sha256']
            candidates.setdefault(sha,[]).append((p,entry))
    paths={p.name:p for p in C.DEFAULT_DATA_ROOT.rglob('*.xdf')}
    records=[]
    for index,job in enumerate(source['completed'],1):
        entry=json.loads((Path(job['output'])/'result.json').read_text(encoding='utf-8'))
        rec=entry['inputs']['recordings'][0]
        sha=rec['sha256']
        provenance=None
        for path,cached in sorted(candidates.get(sha,[]),key=lambda x:x[1]['created'],reverse=True):
            if cached['inputs']['recordings'][0].get('analysis_window_policy')!=rec.get('analysis_window_policy'):
                continue
            if storage.load_cached(path.parent,cached['signature']) is None:
                continue
            rows=read_csv(path.parent/cached['tables']['channel_codes'])
            if len(rows)!=512:
                continue
            n_frames=int(cached['summary']['n_frames'])
            if any(sum(int(r[f'n_code{k}']) for k in range(4))!=n_frames for r in rows):
                continue
            ids=np.array([int(r['electrode_id']) for r in rows])
            maxima=np.array([int(r['max_code']) for r in rows])
            provenance=str(path)
            break
        if provenance is None:
            path=paths[rec['name']]
            if storage.recording_sha(path)!=sha:
                raise ValueError(f'Recording changed: {path}')
            session=Session.load(describe(path,C.DEFAULT_DATA_ROOT))
            ids=session.electrode_ids
            n_frames=len(session.counts)
            maxima=np.max(session.counts,axis=0)
            provenance=str(path)
            del session
        bins,channels=distribution(ids,maxima,depths,100.)
        name=Path(rec['name']).stem
        date=Path(job['output']).parents[1].name[:10]
        if not name.startswith(date):name=date+'_'+name
        records.append(dict(name=name,label=entry['summary']['recording'],bins=bins,channels=channels,
                            n_frames=n_frames,source=provenance,n_active=sum(r['active'] for r in channels)))
        print(f'[{index}/{len(source["completed"])}] n={records[-1]["n_active"]} {name}',flush=True)
    out=root/'single/S14/S14_07_active_channel_depth_distribution'
    out.mkdir(exist_ok=True)
    y_max=max(1,int(np.ceil(max(r['single_unit_count'] for record in records for r in record['bins'])*1.15)))
    with PdfPages(out/'All_runs_active_channel_depth_distribution.pdf') as pdf:
        for record in records:
            fig=figure(record['bins'],record['label'],y_max=y_max)
            fig.savefig(out/(record['name']+'.png'),dpi=300,facecolor='white')
            fig.savefig(out/(record['name']+'.pdf'),facecolor='white')
            pdf.savefig(fig,facecolor='white')
            fig.clear()
    for table in ('bins','channels'):
        rows=[dict(recording=record['name'],**row) for record in records for row in record[table]]
        with (out/f'active_depth_{table}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta=[dict(name=r['name'],source=r['source'],n_frames=r['n_frames'],n_active=r['n_active']) for r in records]
    (out/'manifest.json').write_text(json.dumps(dict(records=meta,depth_csv=str(C.DEFAULT_DEPTH_CSV),
        depth_csv_sha256=depth_sha,bin_um=100,shank1_depth_offset_um=106850,
        excluded='none; latest Shank1 +106850 um',
        definition='one transmitted channel counts once if any received frame has code > 0',
        scope='whole loaded recording, no trial filtering',shared_y_max=y_max,
        reused_s02=sum(r['source'].endswith('result.json') for r in records)),ensure_ascii=False,indent=2),encoding='utf-8')
    assert storage.file_digest(C.DEFAULT_DEPTH_CSV)==depth_sha
    print(f'DONE: {len(records)} PNGs, individual PDFs and combined PDF: {out}',flush=True)


if __name__=='__main__':
    main()
