"""S15: interactive task replay and a static verification overview."""
import json
from pathlib import Path
import numpy as np

from ..base import Section, SectionResult, Param, new_figure
from .replay import build_payload
from .clips import CLIP_CHOICES


class TaskReplay(Section):
    id = 's15'
    title = '任务回放'
    description = '按保存的 cursor 位置、速度、目标和时间戳回放 CO8 / Grid 任务；默认 1× 实时，支持暂停、倍速、进度和 trial 跳转。'
    params = (Param('replay_range', '回放片段', 'choice', '全程', CLIP_CHOICES),
              Param('velocity_source', '速度来源', 'choice', '自动', ('自动','保存速度','位置差分')),
              Param('tail_seconds', '尾迹时长 (s)', default=1., minimum=0, maximum=10),
              Param('max_gap', '采样缺口阈值 (s)', default=.25, minimum=.02, maximum=10))

    def available(self, sessions, channel_config_path=''):
        if not sessions or sessions[0].task_mode not in ('centerout','grid'):
            return '任务回放目前支持 CO8 和 Grid 记录'
        return ''

    def run(self, ctx):
        ctx.progress('整理任务配置与 cursor 时间轴')
        payload = build_payload(ctx.session, ctx.params, ctx.check_cancel)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        template = Path(__file__).with_name('player.html').read_text(encoding='utf-8')
        result = SectionResult(summary=dict(recording=ctx.session.label, velocity_source=payload['velocity_source'], **payload['stats']), notes=payload['notes'])
        result.summary['bps_peak'] = payload['bps_peak']
        result.summary['recorded_bps_trials'] = len(payload['recorded_bps']['trials'])
        result.tables['recorded_trial_bps'] = payload['recorded_bps']['trials']
        result.tables['replay_clips'] = payload['clips']
        result.attachments = {'task_replay.json':encoded,
                              'task_replay.html':template.replace('__REPLAY_PAYLOAD__', encoded.replace('</', '<\\/'))}
        frames = np.asarray([[np.nan if v is None else v for v in r] for r in payload['frames']])
        speed = np.hypot(frames[:,3],frames[:,4])
        breaks = np.r_[False,(np.diff(frames[:,0])>payload['max_gap']) | np.any(np.diff(frames[:,7:10],axis=0)!=0,axis=1)]
        xy = frames[:,1:3].copy();xy[breaks]=np.nan
        fig = new_figure(12,6)
        ax, velocity = fig.subplots(1,2)
        ax.plot(xy[:,0],xy[:,1],lw=.7,color='#267c9d',alpha=.7)
        targets = np.unique(frames[(frames[:,9]==1)&np.isfinite(frames[:,5:7]).all(axis=1),5:7],axis=0)
        if len(targets):ax.scatter(targets[:,0],targets[:,1],c='#c94848',s=25,label='Recorded targets')
        ax.set(xlabel='X (px)',ylabel='Y (px)',title='Task trajectory',aspect='equal')
        g=payload['geometry'];ax.set_xlim(-g['width']/2,g['width']/2);ax.set_ylim(-g['height']/2,g['height']/2)
        plotted_speed=speed.copy();plotted_speed[breaks]=np.nan
        velocity.plot(frames[:,0],plotted_speed,lw=.65,color='#267c9d')
        velocity.set(xlabel='Time (s)',ylabel='Speed (px/s)',title='Cursor velocity')
        fig.suptitle('Task Replay Overview')
        result.add_figure('Task replay overview',fig,
                          method='All loaded cursor_state frames; original timestamps. Recorded xy is replayed without reintegrating velocity. Static paths separate trials. Interactive Grid playback interpolates short position-continuous transitions; resets and long gaps are not interpolated. BPS is read by trial identity from saved rolling_metrics.csv without recomputation. Velocity source: '+payload['velocity_source']+'. The task config/log controls geometry when uniquely assigned to this recording.',
                          data={'trajectory':{'time_s':frames[:,0],'x':xy[:,0],'y':xy[:,1],'speed_px_s':plotted_speed}})
        result.tables['trial_starts']=payload['trials']
        result.tables['replay_sources']=[dict(kind=k,path=v) for k,v in payload['sources'].items()]
        return result


SECTION = TaskReplay()
