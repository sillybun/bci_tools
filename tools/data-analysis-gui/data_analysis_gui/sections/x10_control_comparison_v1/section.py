from pathlib import Path
import tempfile
from ..base import Section, SectionResult
from ...control_comparison import discover, collect, figure, export_workbook, WORKBOOK_NAME
from ...report_run_pool import RunPool

class ControlComparison(Section):
    id='x10'
    title='脑控与离线摇杆 BPS / TTT 配置对照'
    mode='cross'
    min_recordings=1
    description='所选日期范围的全部正式 run（包含 CSV-only 日期，排除1116），颜色区分控制方式；导出完整配置与统计 Excel。'
    cache_code_dependencies=('control_comparison.py','control_comparison_excel.mjs','report_behavior.py','sections/s13_online_behavior_v1/rolling_bps.py','core/behavior.py','sections/s13_online_behavior_v1/co8_bps.py')
    def extra_signature_inputs(self, recordings, params):
        from ...control_comparison import source_signature
        return source_signature([r.path for r in recordings])
    def run(self,ctx):
        from ... import storage
        import math
        from ...config import default_output_root
        folders=discover([s.recording.path for s in ctx.sessions])
        if not folders:return SectionResult(notes=['未找到所选日期范围的正式编号 run。'])
        pool=RunPool(default_output_root(folders[0].parent.parent)/'report/individual_runs')
        payload=collect(folders,pool,ctx.progress);result=SectionResult(summary=dict(runs=len(payload['rows'])),tables={'run_statistics':payload['rows']})
        for mode in ('grid','centerout'):
            rows=[r for r in payload['rows'] if r['mode']==mode]
            if rows:result.add_figure(mode+' BCI / Joystick',figure(rows,mode,mode+'：脑控与离线摇杆 BPS / TTT'),payload['method'],{'runs':rows})
        def clean(x):
            if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
            if isinstance(x,list):return [clean(v) for v in x]
            return None if isinstance(x,float) and not math.isfinite(x) else x
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/WORKBOOK_NAME;export_workbook(clean(storage.jsonable(payload)),p);result.attachments[p.name]=p.read_bytes()
        return result

SECTION=ControlComparison()
