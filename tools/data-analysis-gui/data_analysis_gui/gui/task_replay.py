"""Embedded offline task player; Tk scheduling never advances by frame count."""
from bisect import bisect_right
import math
import tkinter as tk
from tkinter import ttk

from ..sections.s15_task_replay_v1.replay import PlaybackClock, sample_at
from .widgets import open_path


class ReplayView(ttk.Frame):
    def __init__(self, master, payload, html_path=None):
        super().__init__(master)
        self.data, self.clock = payload, PlaybackClock(payload['duration'])
        self._timer, self._seeking = None, False
        self.event_times = [e[0] for e in payload['events']]
        nav = ttk.Frame(self, padding=6);nav.pack(fill='x')
        self.play_button = ttk.Button(nav, text='▶ 播放', command=self.toggle)
        self.play_button.pack(side='left')
        ttk.Button(nav,text='回到开头',command=self.reset).pack(side='left',padx=4)
        ttk.Button(nav,text='上一 trial',command=lambda:self.jump(-1)).pack(side='left')
        ttk.Button(nav,text='下一 trial',command=lambda:self.jump(1)).pack(side='left',padx=4)
        self.rate=tk.StringVar(value='1')
        speed=ttk.Combobox(nav,textvariable=self.rate,values=['0.25','0.5','1','2','4'],state='readonly',width=5)
        speed.pack(side='left');speed.bind('<<ComboboxSelected>>',lambda e:self.clock.speed(self.rate.get()))
        ttk.Label(nav,text='×').pack(side='left')
        if html_path:
            ttk.Button(nav,text='独立 HTML demo',command=lambda:open_path(html_path)).pack(side='right')
        options=ttk.Frame(self,padding=(6,0));options.pack(fill='x')
        trial=ttk.Combobox(options,state='readonly',width=28,
                          values=[f'Block {t["block"]} / Trial {t["trial"]}' for t in payload['trials']])
        trial.set('跳转 trial');trial.pack(side='left')
        self.trial_choice=trial
        self.trial_times=[t['time'] for t in payload['trials']]
        trial.bind('<<ComboboxSelected>>',lambda e:self.seek(payload['trials'][trial.current()]['time']))
        self.tail=tk.BooleanVar(value=True);self.arrow=tk.BooleanVar(value=True)
        ttk.Checkbutton(options,text='轨迹尾迹',variable=self.tail,command=self.draw).pack(side='left',padx=10)
        ttk.Checkbutton(options,text='速度箭头',variable=self.arrow,command=self.draw).pack(side='left')
        self.position=tk.DoubleVar(value=0)
        self.slider=ttk.Scale(self,from_=0,to=payload['duration'],variable=self.position,command=self.scrub)
        self.slider.pack(fill='x',padx=8,pady=5)
        self.status=tk.StringVar();ttk.Label(self,textvariable=self.status,padding=6).pack(fill='x')
        self.canvas=tk.Canvas(self,background='#080b10',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda e:self.draw())
        self.canvas.bind('<Unmap>',lambda e:self.pause())
        self.bind('<Destroy>',self.destroyed,add='+')
        self._timer=self.after(33,self.tick)

    def destroyed(self,event):
        if event.widget is self and self._timer is not None:
            self.after_cancel(self._timer);self._timer=None

    def pause(self):
        self.clock.pause();self.play_button.configure(text='▶ 播放')

    def toggle(self):
        if self.clock.playing:self.pause()
        else:self.clock.play();self.play_button.configure(text='❚❚ 暂停')
        self.draw()

    def reset(self):
        self.pause();self.seek(0)

    def seek(self,value):
        self.clock.seek(value);self.draw()

    def scrub(self,value):
        if not self._seeking:self.seek(float(value))

    def jump(self,direction):
        t=self.clock.position()
        choices=[v['time'] for v in self.data['trials'] if (v['time']>t+.001 if direction>0 else v['time']<t-.001)]
        if choices:self.seek(choices[0] if direction>0 else choices[-1])

    def tick(self):
        if self.winfo_ismapped():self.draw()
        self._timer=self.after(33,self.tick)

    def draw(self):
        c=self.canvas;w,h=c.winfo_width(),c.winfo_height()
        if min(w,h)<10:return
        d=self.data;g=d['geometry'];t=self.clock.position();i,s,gap=sample_at(d,t)
        self._seeking=True;self.position.set(t);self._seeking=False
        if self.clock.playing and t>=d['duration']:self.pause()
        def number(v):return 'N/A' if gap or v is None else f'{v:.1f}'
        trial_index=bisect_right(self.trial_times,t)-1
        if s[9] and trial_index>=0:self.trial_choice.current(trial_index)
        else:self.trial_choice.set('跳转 trial')
        speed=None if s[3] is None or s[4] is None else math.hypot(s[3],s[4])
        event_index=bisect_right(self.event_times,t)-1
        event=d['events'][event_index][1] if event_index>=0 else ''
        self.status.set(f'{t:.2f} / {d["duration"]:.2f} s · {self.clock.rate:g}× · '
                        f'Block {s[7]:g} / Trial {s[8]:g} · {"任务中" if s[9] else "间隔"}'
                        f'{" · 采样缺口" if gap else ""}\n'
                        f'位置 ({number(s[1])}, {number(s[2])}) px · '
                        f'{d["velocity_source"]} ({number(s[3])}, {number(s[4])}) px/s · |v| {number(speed)} px/s · {event}')
        scale=min((w-24)/g['width'],(h-24)/g['height'])
        def xy(x,y):return w/2+x*scale,h/2-y*scale
        c.delete('all')
        c.create_rectangle(*xy(-g['width']/2,g['height']/2),*xy(g['width']/2,-g['height']/2),outline='#455269')
        if d['task_mode']=='grid' and g['grid_rows'] and g['grid_cols']:
            gw,gh=g['grid_cols']*g['cell_size'],g['grid_rows']*g['cell_size']
            for col in range(g['grid_cols']+1):
                x=-gw/2+col*g['cell_size'];c.create_line(*xy(x,-gh/2),*xy(x,gh/2),fill='#334154')
            for row in range(g['grid_rows']+1):
                y=-gh/2+row*g['cell_size'];c.create_line(*xy(-gw/2,y),*xy(gw/2,y),fill='#334154')
        if d['task_mode']=='centerout':
            r=g['target_size']/2;c.create_oval(*xy(-r,r),*xy(r,-r),outline='#64748b')
        if gap:
            c.create_text(w/2,h/2,text='此时间段没有 cursor 采样',fill='#f4c666',font=('Microsoft YaHei UI',16));return
        if s[9] and s[5] is not None and s[6] is not None:
            size=g['cell_size'] if d['task_mode']=='grid' else g['target_size']
            shape=c.create_oval if g['target_shape']=='circle' else c.create_rectangle
            shape(*xy(s[5]-size/2,s[6]+size/2),*xy(s[5]+size/2,s[6]-size/2),fill='#da4b4b',outline='')
        if self.tail.get():
            f=d['frames'];j=i
            while (j>0 and f[j][0]>t-d['tail_seconds'] and f[j-1][7:10]==s[7:10]
                   and f[j][0]-f[j-1][0]<=d['max_gap']):j-=1
            points=[coord for r in f[j:i+1] for coord in xy(r[1],r[2])]+list(xy(s[1],s[2]))
            if len(points)>=4:c.create_line(*points,fill='#44b9d1',width=2)
        r=g['cursor_radius'];c.create_oval(*xy(s[1]-r,s[2]+r),*xy(s[1]+r,s[2]-r),fill='#f5e45e',outline='')
        if self.arrow.get() and speed is not None and speed>1e-9:
            c.create_line(*xy(s[1],s[2]),*xy(s[1]+s[3]*.1,s[2]+s[4]*.1),fill='#68d9ff',width=2,arrow='last')
