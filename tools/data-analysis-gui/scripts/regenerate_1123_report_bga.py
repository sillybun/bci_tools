"""Replace only daily BGA outputs using the first usable XDF run per day."""
from pathlib import Path
from datetime import datetime,timezone
import csv,html,json,sys,zipfile
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
from data_analysis_gui import storage
from data_analysis_gui.report_bga import render_daily_bga,first_usable_run
from data_analysis_gui.report_run_pool import RunPool,atomic_json,link_file

ROOT=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-10_2026-11-23_60runs_v1')
REL=Path('02_通道稳定性分析/BGA_daily_first_run')
AUDIT=ROOT.parent.parent/'import_20261123'/('bga_first_usable_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S'))

def main():
    AUDIT.mkdir(parents=True,exist_ok=False)
    report=json.loads((ROOT/'report.json').read_text(encoding='utf-8'))
    before={p.relative_to(ROOT).as_posix():storage.file_digest(p) for p in ROOT.rglob('*')
            if p.is_file() and not p.is_relative_to(ROOT/REL) and p.suffix in ('.png','.pdf','.svg','.csv')}
    with zipfile.ZipFile(AUDIT/'previous_bga.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in [ROOT/'report.json',ROOT/'bga_daily_manifest.json',*(ROOT/REL).glob('*')]:
            if p.is_file():z.write(p,p.relative_to(ROOT).as_posix())
    stage=AUDIT/'generated';stage.mkdir()
    pool=RunPool(ROOT.parent/'individual_runs')
    result=render_daily_bga(stage,report,100,lambda *a,**k:print(k.get('message',''),flush=True),pool=pool)
    expected={'20261110':'run-001-CO8-','20261111':'run-001-CO8-','20261117':'run-001-',
              '20261118':'run-004-','20261119':'run-001-','20261120':'run-001-','20261123':'run-002-'}
    assert set(d['date'] for d in result['days'])==set(expected)
    items=[]
    for d in result['days']:
        folder=Path(d['run_folder']);assert folder.name.startswith(expected[d['date']])
        selected,values,info,_,_=first_usable_run(folder.parent,pool)
        assert selected==folder and info['status']==d['status']
        with (stage/d['files']['csv']).open(encoding='utf-8-sig') as f:
            rows=list(csv.DictReader(line for line in f if not line.startswith('#')))
        assert len(rows)==1024
        for row in rows:
            ch=int(row['channel_0based']);observed=row['observed']=='True'
            assert observed==bool(np.isfinite(values[ch]))
            if observed:assert int(row['max_code'])==values[ch]
        assert d['transmitted_channels']==(0 if d['date']=='20261117' else 512)
        for name in d['files'].values():
            assert (stage/name).is_file()
            link_file(stage/name,ROOT/name)
        details='当天无可用 XDF 神经数据' if d['status'] not in ('ok','partial_xdf_coverage') else '首个有可用 XDF 神经数据的 run'
        items.append('<article><h2>'+d['date']+' · '+html.escape(folder.name)+'</h2><p>'+details+'；传输通道：'+str(d['transmitted_channels'])+'</p><a href="'+Path(d['files']['png']).name+'"><img loading="lazy" src="'+Path(d['files']['png']).name+'"></a></article>')
    index=ROOT/REL/'index.html'
    index.write_text('<!doctype html><meta charset="utf-8"><title>每日首个有效XDF的BGA</title><style>body{font:15px Microsoft YaHei,Arial;margin:24px;color:#17212b}img{width:100%;max-width:1100px}article{margin-bottom:36px}</style><h1>每天第一个有可用 XDF 神经数据的 run：BGA 映射图</h1><p>1110–1123，排除1116。按run顺序跳过无XDF或无可读神经计数的记录，图标题为实际使用的run编号；仅当天全部run均无可用数据时显示无数据。彩色=非零code（按shank），黑色=全零，灰色=未传输/无数据。</p>'+''.join(items),encoding='utf-8')
    atomic_json(ROOT/'bga_daily_manifest.json',result)
    report['bga_daily']=dict(result,requested_backfill=True,index=str(index.relative_to(ROOT)))
    report['daily_bga_pool']=pool.manifest()
    atomic_json(ROOT/'report.json',report)
    assert all(storage.file_digest(ROOT/name)==digest for name,digest in before.items())
    verification=dict(days=[dict(date=d['date'],run=Path(d['run_folder']).name,status=d['status'],transmitted=d['transmitted_channels']) for d in result['days']],
                      other_artifacts_unchanged=len(before),index=str(index))
    atomic_json(AUDIT/'verification.json',verification)
    print(json.dumps(verification,ensure_ascii=False))

if __name__=='__main__':main()
