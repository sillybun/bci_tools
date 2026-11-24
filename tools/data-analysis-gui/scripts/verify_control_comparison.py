"""Read delivered XLSX XML independently and reconcile it to exported chart values."""
from pathlib import Path
import json,math,zipfile,xml.etree.ElementTree as ET,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data_analysis_gui.control_comparison import CATEGORY
root=Path('E:/AitBCI/data_analysis_gui_output/keaiduo/report/2026-11-10_2026-11-23_60runs_v1')/CATEGORY/'summary_all_runs'
d=json.loads((root/'comparison_data.json').read_text(encoding='utf-8'))
ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with zipfile.ZipFile(root/'behavior_config_statistics.xlsx') as z:
    strings=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        strings=[''.join(s.itertext()) for s in ET.fromstring(z.read('xl/sharedStrings.xml'))]
    def sheet(n):
        doc=ET.fromstring(z.read(f'xl/worksheets/sheet{n}.xml'));cells={}
        for c in doc.findall('.//s:c',ns):
            t=c.get('t');v=c.findtext('s:v',None,ns)
            assert t!='e',(n,c.get('r'),v)
            if t=='s':v=strings[int(v)]
            elif t=='inlineStr':v=''.join(c.find('s:is',ns).itertext())
            elif t=='b':v=v=='1'
            elif v is not None and t!='str':v=float(v)
            cells[c.get('r')]=v
        return cells
    cells=sheet(1);headers={v:k[:-1] for k,v in cells.items() if k.endswith('1') and not k[-2:-1].isdigit()}
    for i,r in enumerate(d['rows'],2):
        assert cells[f'{headers["配置对应键"]}{i}']==r['config_key']
        assert cells[f'{headers["控制方式"]}{i}']==r['control']
        for key,label in [('bps_median','Grid BPS 中位数 (bits/s)'),('rt_median_s','TTT 中位数 (s)')]:
            if isinstance(r.get(key),(int,float)):assert math.isclose(cells[f'{headers[label]}{i}'],r[key],rel_tol=1e-12)
    allcfg=sheet(3);assert 'GRID_TASK.PREVIEW_NEXT_TARGET' in allcfg.values()
    for n in (2,4):sheet(n)
assert len(d['rows'])==60 and len({r['config_key'] for r in d['rows']})==60
assert sum(r['date']=='2026-11-23' for r in d['rows'])==9
assert not any(r['date']=='2026-11-16' for r in d['rows'])
assert all(s['cache_hit'] for s in d['sources'])
(root/'behavior_config_statistics.xlsx.inspect.ndjson').unlink(missing_ok=True)
print('PASS: 60 run identities, 9 today, 1116 excluded, Excel BPS/TTT equals plotted data, next-target retained, all sources cached, no XLSX errors.')
