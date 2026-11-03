"""Tk workbench in the style of the macaque analysis GUI.

Two fully separate workbenches share one Topbar (Preprocessing):
  单日  — recording list + section navigator on the left, one result on the right
  跨天  — recording selection table with a date timeline on top, one tab per cross section
Results are searched in storage first; only missing results are computed, in a
child process (`python -m data_analysis_gui.worker`) that reports progress line by line.
"""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .. import config as C
from .. import storage
from ..core.catalog import DATA_TYPES, day_folder, filter_type, scan
from ..core.crossday import (OTHER_SESSION_STYLE, SESSION_STYLES, CONDITION_STYLES, recording_style,
                            recording_labels, session_label_color)
from ..core.filters import REASON_LABELS, TrialFilter, apply_filter
from ..core.session import get_session
from ..core.style import DEFAULT_FONT_PERCENT
from ..sections import LOAD_ERRORS, registry, sections, versions
from . import theme
from .theme import FONT, HEADER, HEADER_CROSS, MUTED, OK, WARN, px
from ..sections.guide import load_guide
from .filter_dialog import FilterDialog, summarize
from .cross_overview import comparison_figure
from .task_queue import TaskQueue, STATE_LABELS, TERMINAL
from .widgets import ParamForm, ResultView, open_path, table_view

DEFAULT_OUTPUT = C.TOOL_ROOT / 'output'
SLOW_SINGLE_SECTIONS = ('s10', 's12', 's11', 's14')
STATUS_TEXT = {'cached': '✓ 已存储', 'missing': '· 未计算', 'running': '… 计算中', 'error': '✗ 失败',
               'unknown': '…', 'invalid': '! 参数', 'queued': '… 已排队'}

SECTION_TITLES = {
    's01': '神经活动与任务概览',
    's02': '电极活动与编码质量',
    's03': '传输质量诊断',
    's04': '群体神经活动的低维表征',
    's05': '单电极方向调谐',
    's06': '单电极时间响应（PSTH）',
    's07': '调谐汇总与基线变化',
    's08': '通道分层与子集解码',
    's09': '方向分类性能',
    's10': '运动学回归性能',
    's11': '在线解码回放与离线对照',
    's12': '单记录总览（辅助）',
    's13': '在线任务表现',
    's14': '解码权重与电极深度',
    'x00': '记录示意图（辅助）',
    'x01': '电极活动稳定性',
    'x02': '传输质量比较',
    'x03': '群体方向表征一致性',
    'x04': '方向分类泛化',
    'x05': '运动学回归泛化',
    'x06': '参考模型迁移与再校准',
    'x07': '单电极调谐与时间响应稳定性',
    'x08': '在线任务表现比较',
    'x09': '解码权重 median / average',
}


def section_display(section):
    identifier = section.display.split(' · ', 1)[0]
    return f'{identifier} · {SECTION_TITLES.get(section.id, section.title)}'


# =================================================================== app
def checked_output_root(saved) -> str:
    """Saved result folder, unless it is missing or lives in the system temp directory
    (e.g. left behind by a test run) — then the tool's own output/ folder."""
    if not saved:
        return str(DEFAULT_OUTPUT)
    path = Path(saved)
    try:
        in_temp = path.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve())
    except (OSError, ValueError):
        in_temp = False
    return str(DEFAULT_OUTPUT) if in_temp else str(path)


class App(tk.Tk):
    def __init__(self):
        theme.enable_dpi_awareness()
        super().__init__()
        self.title('BCI Data Analysis · 数据分析工作台')
        self.prefs = C.load_preferences()
        try:
            self.geometry(self.prefs.get('geometry', '1560x980'))
        except tk.TclError:
            self.geometry('1560x980')
        self.minsize(1200, 760)
        theme.apply(self)
        self.recordings, self.visible = [], []
        self.events: queue.Queue = queue.Queue()
        self.task_queue = TaskQueue(self.events)
        self.log_lines: list[str] = []
        registry()
        self._build()
        if LOAD_ERRORS:
            self.status.set('部分 section 加载失败：' + ', '.join(LOAD_ERRORS))
        self.after(50, self.refresh_recordings)
        self.after(100, self._poll)
        self.bind('<Control-r>', lambda e: self.view.compute_current())
        self.protocol('WM_DELETE_WINDOW', self.close)

    @property
    def process(self):
        return self.task_queue.process

    # ------------------------------------------------------------ layout
    def _build(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        # Task bar (progress) — same placement as the macaque GUI.
        taskbar = ttk.Frame(self, padding=(14, 7))
        taskbar.grid(row=0, column=0, sticky='ew')
        taskbar.columnconfigure(1, weight=1)
        self.progress_text = tk.StringVar(value='已结束 0/0 · 排队 0 · 失败 0 · 停止 0')
        self.task_text = tk.StringVar(value='等待任务 · 优先读取已存储结果，缺失时才计算')
        ttk.Label(taskbar, textvariable=self.progress_text, style='Status.TLabel').grid(row=0, column=0, sticky='w')
        # The label fills the available space without requesting width based on its text.
        ttk.Label(taskbar, textvariable=self.task_text, width=1, anchor='w').grid(
            row=0, column=1, sticky='ew', padx=18)
        self.cancel_button = ttk.Button(taskbar, text='停止当前', command=self.cancel, state='disabled')
        self.cancel_button.grid(row=0, column=2, sticky='e')
        ttk.Button(taskbar, text='任务队列', command=self.show_queue).grid(row=0, column=3, sticky='e', padx=(6, 0))
        ttk.Button(taskbar, text='清空待执行', command=self.clear_queue).grid(row=0, column=4, sticky='e', padx=(6, 0))
        ttk.Button(taskbar, text='日志', command=self.show_log).grid(row=0, column=5, sticky='e', padx=(6, 0))
        self.progress = ttk.Progressbar(taskbar, mode='determinate', maximum=1, value=0,
                                        style='Task.Horizontal.TProgressbar')
        self.progress.grid(row=1, column=0, columnspan=6, sticky='ew', pady=(5, 0))

        # Header with the mode switch.
        self.header = tk.Frame(self, background=HEADER, padx=18, pady=10)
        self.header.grid(row=1, column=0, sticky='ew')
        self.header_title = tk.Label(self.header, text='BCI  /  DATA ANALYSIS', foreground='white', background=HEADER,
                                     font=('Segoe UI', 17, 'bold'))
        self.header_title.pack(side='left')
        self.header_sub = tk.Label(self.header, text='', foreground='#c9d6e6', background=HEADER, font=(FONT, 10))
        self.header_sub.pack(side='left', padx=18)
        self.mode = tk.StringVar(value=self.prefs.get('mode', 'single'))
        self.mode_buttons = {}
        for key, label in (('cross', '跨记录分析  Cross-session'), ('single', '单记录分析  Single-session')):
            button = tk.Label(self.header, text=label, font=(FONT, 11, 'bold'), padx=16, pady=6, cursor='hand2')
            button.pack(side='right', padx=(6, 0))
            button.bind('<Button-1>', lambda e, k=key: self.set_mode(k))
            self.mode_buttons[key] = button

        self.all_button = tk.Label(self.header, text='⚡ 一键计算 Compute All', font=(FONT, 11, 'bold'),
                                   padx=16, pady=6, cursor='hand2', background='#2464a5', foreground='white')
        self.all_button.pack(side='right', padx=(6, 24))
        self.all_button.bind('<Button-1>', lambda e: self.compute_everything())

        self._build_topbar()
        self.body = ttk.Frame(self, padding=(12, 4, 12, 8))
        self.body.grid(row=3, column=0, sticky='nsew')
        self.body.rowconfigure(0, weight=1)
        self.body.columnconfigure(0, weight=1)
        self.single_view = SingleView(self.body, self, config_parent=self.channel_files_row)
        self.cross_view = CrossView(self.body, self)
        self.status = tk.StringVar(value='就绪')
        bar = ttk.Frame(self, padding=(14, 0, 14, 6))
        bar.grid(row=4, column=0, sticky='ew')
        ttk.Label(bar, textvariable=self.status, style='Muted.TLabel').pack(side='left')
        self.set_mode(self.mode.get(), initial=True)

    def _build_topbar(self):
        p = self.prefs
        top = ttk.LabelFrame(self, text=' Topbar · Preprocessing ', padding=(10, 6))
        top.grid(row=2, column=0, sticky='ew', padx=12, pady=(8, 2))
        top.columnconfigure(1, weight=1)
        # Row 0: data directory, data types
        ttk.Label(top, text='数据目录').grid(row=0, column=0, sticky='w')
        self.data_root = tk.StringVar(value=p.get('data_root', str(C.DEFAULT_DATA_ROOT)))
        ttk.Entry(top, textvariable=self.data_root).grid(row=0, column=1, sticky='ew', padx=6)
        row = ttk.Frame(top)
        row.grid(row=0, column=2, sticky='w')
        ttk.Button(row, text='选择…', command=self.choose_root).pack(side='left')
        ttk.Button(row, text='刷新', command=self.refresh_recordings).pack(side='left', padx=4)
        ttk.Label(row, text='   数据类型').pack(side='left')
        self.type_vars = {}
        for t in DATA_TYPES:
            var = tk.BooleanVar(value=t in p.get('data_types', ['CO8', 'CO8 online']))
            ttk.Checkbutton(row, text=t, variable=var, command=self.apply_type_filter).pack(side='left', padx=3)
            self.type_vars[t] = var
        # Row 1: trial criteria live in a pop-out dialog; the topbar shows a summary.
        self.filter = TrialFilter.from_dict(p.get('filter'))
        ttk.Label(top, text='Trial 筛选').grid(row=1, column=0, sticky='w', pady=(6, 0))
        row = ttk.Frame(top)
        row.grid(row=1, column=1, columnspan=2, sticky='ew', padx=6, pady=(6, 0))
        ttk.Button(row, text='筛选设置…', style='Accent.TButton', command=self.open_filter_dialog).pack(side='left')
        ttk.Button(row, text='筛选详情…', command=self.show_trial_table).pack(side='left', padx=6)
        self.filter_summary = tk.StringVar(value=summarize(self.filter))
        ttk.Label(row, textvariable=self.filter_summary, style='Muted.TLabel', wraplength=1300,
                  justify='left').pack(side='left', padx=(8, 0))
        self.layout_path = tk.StringVar(value=p.get('layout_path', str(C.DEFAULT_LAYOUT_XLSX)))
        # Result folder follows the data folder (<…>/data/<x> -> <…>/data_analysis_gui_output/<x>) unless the
        # user picked one explicitly.
        self.output_follow = tk.BooleanVar(value=p.get('output_follow_data', True))
        self.output_root = tk.StringVar(value=self.followed_output() if self.output_follow.get()
                                        else checked_output_root(p.get('output_root')))
        self.data_root.trace_add('write', lambda *_: self.sync_output(refresh=False))
        self.recompute = tk.BooleanVar(value=False)
        ttk.Label(top, text='通道文件').grid(row=2, column=0, sticky='w', pady=(4, 0))
        row = ttk.Frame(top)
        row.grid(row=2, column=1, columnspan=2, sticky='ew', padx=6, pady=(4, 0))
        self.channel_files_row = row
        self.use_layout = tk.BooleanVar(value=p.get('use_layout', True))
        ttk.Checkbutton(row, text='使用电极布局 xlsx', variable=self.use_layout,
                        command=self.layout_toggled).pack(side='left')
        self.layout_entry = ttk.Entry(row, textvariable=self.layout_path, width=40)
        self.layout_entry.pack(side='left', padx=4)
        self.layout_button = ttk.Button(row, text='…', width=3,
                                        command=lambda: self.choose_file(self.layout_path, [('Excel', '*.xlsx')]))
        self.layout_button.pack(side='left')
        self.layout_hint = ttk.Label(row, text='', style='Muted.TLabel')
        self.layout_hint.pack(side='left', padx=(6, 0))
        self._layout_state()
        ttk.Label(top, text='结果存储').grid(row=3, column=0, sticky='w', pady=(4, 0))
        row = ttk.Frame(top)
        row.grid(row=3, column=1, columnspan=2, sticky='ew', padx=6, pady=(4, 0))
        self.output_entry = ttk.Entry(row, textvariable=self.output_root, width=60)
        self.output_entry.pack(side='left')
        ttk.Button(row, text='…', width=3, command=self.choose_output).pack(side='left', padx=4)
        ttk.Checkbutton(row, text='跟随数据目录', variable=self.output_follow,
                        command=self.sync_output).pack(side='left', padx=(0, 6))
        self.sync_output()
        ttk.Button(row, text='打开', command=lambda: open_path(self.ensure_output())).pack(side='left')
        ttk.Checkbutton(row, text='Recompute / 强制重算（忽略已存储结果）', variable=self.recompute).pack(side='left', padx=(16, 0))
        font_options = tuple(range(80, 161, 10))
        font_percent = self.prefs.get('font_percent', DEFAULT_FONT_PERCENT)
        if not isinstance(font_percent, int) or font_percent not in font_options:
            font_percent = DEFAULT_FONT_PERCENT
        self.figure_font = tk.StringVar(value=f'{font_percent:d}%')
        ttk.Label(row, text='图表字号').pack(side='left', padx=(16, 4))
        font_box = ttk.Combobox(row, textvariable=self.figure_font, state='readonly', width=6,
                                values=[f'{size}%' for size in font_options])
        font_box.pack(side='left')
        font_box.bind('<<ComboboxSelected>>', self._figure_font_changed)
        ttk.Label(row, text='每张图保存 PNG / SVG / PDF / CSV',
                  style='Muted.TLabel').pack(side='left', padx=(12, 0))
        from ..exporting import EXPORT_LAYOUTS
        row = ttk.Frame(top)
        row.grid(row=4, column=1, columnspan=2, sticky='ew', padx=6, pady=(4, 0))
        ttk.Label(top, text='批次导出').grid(row=4, column=0, sticky='w', pady=(4, 0))
        selected_layout = p.get('export_layout', '按 run 归类')
        self.export_layout = tk.StringVar(value=selected_layout if selected_layout in EXPORT_LAYOUTS else '按 run 归类')
        ttk.Combobox(row, textvariable=self.export_layout, values=list(EXPORT_LAYOUTS),
                     state='readonly', width=18).pack(side='left')
        ttk.Button(row, text='导出当前分析（选中 runs）', command=self.export_selected).pack(side='left', padx=6)
        ttk.Button(row, text='打开本次导出', command=self.open_export_folder).pack(side='left')
        ttk.Label(row, text='计算时同步导出；每批次单独建文件夹', style='Muted.TLabel').pack(side='left', padx=10)
        self.last_export_folder = None

    def figure_font_percent(self):
        return int(self.figure_font.get().removesuffix('%'))

    def _figure_font_changed(self, _event):
        self.save_preferences()
        self.view.refresh_status()

    # ------------------------------------------------------------ mode
    def set_mode(self, mode, initial=False):
        self.mode.set(mode)
        colour = HEADER if mode == 'single' else HEADER_CROSS
        self.header.configure(background=colour)
        self.header_title.configure(background=colour)
        self.header_sub.configure(background=colour, text='单份记录的完整分析' if mode == 'single'
                                  else '按电极 ID 对齐并比较多份记录')
        for key, button in self.mode_buttons.items():
            active = key == mode
            button.configure(background='white' if active else colour, foreground=colour if active else '#c9d6e6')
        self.single_view.grid_forget()
        self.cross_view.grid_forget()
        self.view = self.single_view if mode == 'single' else self.cross_view
        self.view.grid(row=0, column=0, sticky='nsew')
        if not initial:
            self.view.refresh()

    # ------------------------------------------------------------ inputs
    def criteria(self) -> TrialFilter:
        return self.filter

    def open_filter_dialog(self):
        FilterDialog(self, self.filter, self.set_filter)

    def set_filter(self, criteria: TrialFilter):
        self.filter = criteria
        self.filter_summary.set(summarize(criteria))
        self.save_preferences()
        self.view.refresh_status()

    def compute_everything(self):
        """One click: every section of the current mode, cache-first, in the background."""
        self.view.compute_all()

    def effective_layout(self) -> str:
        """Workbook path, or '' = default index arrangement (electrode e at column (e-1)//128,
        row (e-1)%128 of the same 128 × 8 grid of 1024 sites)."""
        return self.layout_path.get() if self.use_layout.get() else ''

    def _layout_state(self):
        state = '!disabled' if self.use_layout.get() else 'disabled'
        self.layout_entry.state([state])
        self.layout_button.state([state])
        self.layout_hint.configure(text='' if self.use_layout.get() else
                                   '未选中：按电极 index 排在 128 行 × 8 列（1–128 为第 1 列…）')

    def layout_toggled(self):
        self._layout_state()
        self.save_preferences()
        self.view.refresh_status()

    def choose_root(self):
        folder = filedialog.askdirectory(initialdir=self.data_root.get() or None)
        if folder:
            self.data_root.set(folder)
            self.refresh_recordings()

    def choose_file(self, var, types):
        path = filedialog.askopenfilename(filetypes=types)
        if path:
            var.set(path)
            self.view.refresh_status()

    def choose_output(self):
        folder = filedialog.askdirectory(initialdir=self.output_root.get() or None)
        if folder:
            self.output_follow.set(False)           # an explicit choice stops following the data folder
            self.output_root.set(folder)
            self.sync_output()

    def followed_output(self) -> str:
        try:
            return str(C.default_output_root(self.data_root.get() or C.DEFAULT_DATA_ROOT))
        except (OSError, ValueError):
            return str(DEFAULT_OUTPUT)

    def sync_output(self, refresh=True):
        """Apply the follow-data-folder setting to the result folder field and refresh cache status."""
        following = self.output_follow.get()
        if following:
            self.output_root.set(self.followed_output())
        if hasattr(self, 'output_entry'):
            self.output_entry.configure(state='readonly' if following else 'normal')
        if refresh and hasattr(self, 'view'):
            self.view.refresh_status()

    def output_path(self) -> Path:
        """Current result folder, not created (status lookups; the path may still be changing)."""
        return Path(self.output_root.get() or DEFAULT_OUTPUT)

    def ensure_output(self):
        path = Path(self.output_root.get() or DEFAULT_OUTPUT)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def refresh_recordings(self):
        self.recordings, notice = scan(self.data_root.get())
        bad = sum(1 for r in self.recordings if r.reason)
        self.status.set(f'{len(self.recordings) - bad} 份可用记录' + (f' · {bad} 份无法分析' if bad else '') +
                        (f' · {notice}' if notice else ''))
        self.apply_type_filter()

    def apply_type_filter(self):
        types = [t for t, v in self.type_vars.items() if v.get()]
        self.visible = filter_type(self.recordings, types)
        self.single_view.set_recordings(self.visible)
        self.cross_view.set_recordings(self.visible)

    # ------------------------------------------------------------ jobs
    def job_key(self, section, recordings, params):
        """Signature + folder for a job; computed off the Tk thread when hashing is needed."""
        inputs = storage.signature_inputs(section, recordings, self.criteria(), params,
                                          self.effective_layout(),
                                          channel_config_path=self._job_channel_config(section, recordings),
                                          font_percent=self.figure_font_percent())
        signature = storage.fingerprint(inputs)
        return signature, storage.result_folder(self.output_path(), section, recordings, signature)

    def _job_channel_config(self, section, recordings):
        if recordings and storage.uses_channel_config(section):
            return self.single_view.channel_config_for(recordings[0])
        return ''

    def export_selected(self):
        """Export the selected analysis for every selected recording from cache."""
        try:
            recs = self.view.selected_recordings()
            if not recs:
                raise ValueError('请先选择需要导出的 runs。')
            if self.mode.get() == 'single':
                section = self.single_view.current_section()
                if section is None:
                    raise ValueError('请选择一个分析模块。')
                jobs = [self.single_view._job(section, r) for r in recs]
            elif self.cross_view.overview_selected():
                section = self.cross_view.single_overview.section()
                jobs = [self.single_view._job(section, r) for r in recs]
            else:
                identifier = self.cross_view.current_id()
                if identifier is None:
                    raise ValueError('请选择一个分析模块。')
                tab = self.cross_view.tabs[identifier]
                jobs = [(tab['section'], recs, self.cross_view._raw_params(tab))]
            self.run_jobs(jobs, None, export_only=True)
        except (TypeError, ValueError) as exc:
            messagebox.showwarning('无法导出', str(exc))

    def open_export_folder(self):
        folder = self.last_export_folder
        if folder is not None and folder.is_dir():
            open_path(folder)
        else:
            messagebox.showinfo('尚未导出', '请先计算或导出所选记录的分析结果。')

    def run_jobs(self, jobs, on_event, *, report=None, export_only=False):
        """Snapshot settings now and enqueue one independent worker per analysis."""
        if not jobs:
            return False
        try:
            criteria = self.criteria()
        except (TypeError, ValueError) as exc:
            messagebox.showerror('筛选参数无效', str(exc))
            return False
        self.save_preferences()
        request = dict(output_root=str(self.ensure_output()), data_root=self.data_root.get(),
                       criteria=criteria.to_dict(), layout_path=self.effective_layout(),
                       font_percent=self.figure_font_percent(),
                       recompute=self.recompute.get(),
                       jobs=[dict(section=s.id, version=s.version, params=s.resolve_params(p),
                                  channel_config_path=self._job_channel_config(s, recs),
                                  recordings=[str(r.path) for r in recs]) for s, recs, p in jobs])
        if report is not None:
            request['report'] = report
        else:
            from ..exporting import EXPORT_LAYOUTS, export_batch
            request['export'] = export_batch(request['output_root'], EXPORT_LAYOUTS[self.export_layout.get()])
            request['export_only'] = export_only
            self.last_export_folder = Path(request['export']['root'])
        added = self.task_queue.submit(request,on_event)
        self.status.set(f'已加入 {len(added)} 项任务；相同设置的等待/运行任务不重复提交')
        self.task_queue.start_next()
        self.update_progress()
        return True

    def _poll(self):
        try:
            while True:
                event = self.events.get_nowait()
                if event['event']=='_status':
                    callback,*args = event['call']
                    callback(*args)
                    continue
                if event['event']=='log':
                    self.log_lines.append(event['line'])
                task = self.task_queue.handle(event)
                if task and event['event'] == 'exit' and self.view is self.cross_view:
                    self.cross_view.single_overview.source_finished(task.job['section'])
                if task and event['event']=='error':
                    self.log_lines.append(event.get('traceback',event.get('message','')))
                if task and event['event']=='exit' and event.get('cancelled'):
                    self.status.set(f'已停止 {task.label}；其余等待任务继续')
        except queue.Empty:
            pass
        self.task_queue.start_next()
        self.update_progress()
        self.after(80,self._poll)

    def update_progress(self, text=None):
        tasks = self.task_queue.items
        total = len(tasks)
        ended = sum(t.state in TERMINAL for t in tasks)
        waiting = len(self.task_queue.waiting)
        failed = sum(t.state=='error' for t in tasks)
        stopped = sum(t.state in ('stopped','cleared') for t in tasks)
        self.progress.configure(maximum=max(total,1),value=ended)
        self.progress_text.set(f'已结束 {ended}/{total} · 排队 {waiting} · 失败 {failed} · 停止 {stopped}')
        current = self.task_queue.current
        if current:
            paths = current.job['recordings']
            selection = f'{len(paths)} 份记录' if len(paths) > 1 else Path(paths[0]).name if paths else ''
            title = (current.label if current.request.get('report')
                     else ' · '.join(part for part in (current.job['section'].upper(), selection) if part))
            detail = current.detail or '核验已存储结果…'
            self.task_text.set(' '.join(f'{STATE_LABELS[current.state]} · {title} — {detail}'.split()))
        else:
            message = text or ('全部任务完成' if tasks else '等待任务 · 可多次提交，按顺序执行')
            self.task_text.set(' '.join(message.split()))
        self.cancel_button.state(['!disabled' if current and current.state=='running' else 'disabled'])
        self._refresh_queue_window()

    def cancel(self):
        if self.task_queue.stop_current():
            self.update_progress('正在停止当前任务…')

    def clear_queue(self):
        removed = self.task_queue.clear_waiting()
        self.status.set(f'已移出 {len(removed)} 项等待任务；当前任务继续运行')
        self.update_progress()

    def queued_status(self, section, recordings, raw):
        try:
            params=section.resolve_params(raw)
        except ValueError:
            return None
        paths=[str(r.path) for r in recordings]
        for task in self.task_queue.items:
            if task.request.get('report'):
                continue
            if task.state not in ('running','queued','stopping'):
                continue
            request,job=task.request,task.job
            if (job['section']==section.id and job['version']==section.version and job['recordings']==paths
                    and job['params']==params and request['criteria']==self.criteria().to_dict()
                    and job.get('channel_config_path', '') == self._job_channel_config(section, recordings)
                    and request.get('font_percent', DEFAULT_FONT_PERCENT) == self.figure_font_percent()
                    and request['output_root']==str(self.output_path()) and request['layout_path']==self.effective_layout()):
                return 'queued' if task.state=='queued' else 'running'
        return None

    def show_queue(self):
        window = getattr(self,'queue_window',None)
        if window is not None and window.winfo_exists():
            window.lift()
            return
        window = self.queue_window = tk.Toplevel(self)
        window.title('任务队列')
        window.geometry('1100x540')
        bar = ttk.Frame(window,padding=8)
        bar.pack(fill='x')
        ttk.Button(bar,text='停止当前',command=self.cancel).pack(side='left')
        ttk.Button(bar,text='清空待执行',command=self.clear_queue).pack(side='left',padx=8)
        self.queue_prioritize_button = ttk.Button(bar, text='移到下一项',
                                                  command=self._prioritize_selected_task, state='disabled')
        self.queue_prioritize_button.pack(side='left', padx=(0, 8))
        ttk.Label(bar,text='等待任务按列表顺序执行。双击查看任务设置。').pack(side='left')
        frame,self.queue_tree = recording_tree(window,[('state','状态',90),('section','Section',75),
                          ('recordings','记录',430),('detail','进度 / 结果',400)],selectmode='browse')
        frame.pack(fill='both',expand=True,padx=8,pady=8)
        self.queue_tree.bind('<Double-1>',lambda e:self._show_queue_details())
        self.queue_tree.bind('<<TreeviewSelect>>', lambda e: self._update_queue_prioritize_button())
        self._queue_render_key = None
        self._refresh_queue_window()

    def _refresh_queue_window(self):
        tree = getattr(self,'queue_tree',None)
        if tree is None or not tree.winfo_exists():
            return
        waiting = list(self.task_queue.waiting)
        waiting_ids = {task.uid for task in waiting}
        tasks = [task for task in self.task_queue.items if task.uid not in waiting_ids] + waiting
        key = tuple((t.uid,t.state,t.detail) for t in tasks)
        if key==getattr(self,'_queue_render_key',None):
            return
        for index, task in enumerate(tasks):
            values=(STATE_LABELS[task.state],task.job['section'].upper(),
                    ', '.join(Path(p).name for p in task.job['recordings']),task.detail)
            if tree.exists(task.uid):
                tree.item(task.uid,values=values)
            else:
                tree.insert('','end',iid=task.uid,values=values)
            if tree.index(task.uid) != index:
                tree.move(task.uid, '', index)
        self._queue_render_key=key
        self._update_queue_prioritize_button()

    def _update_queue_prioritize_button(self):
        selected = self.queue_tree.selection()
        can_prioritize = bool(selected) and any(task.uid == selected[0] for task in self.task_queue.waiting)
        self.queue_prioritize_button.configure(state='normal' if can_prioritize else 'disabled')

    def _prioritize_selected_task(self):
        selected = self.queue_tree.selection()
        if selected and self.task_queue.prioritize(selected[0]):
            self.task_queue.start_next()
            self.update_progress('已将选中任务移到等待队列首位')
            self.queue_tree.see(selected[0])

    def _show_queue_details(self):
        selected=self.queue_tree.selection()
        if not selected:
            return
        task=next(t for t in self.task_queue.items if t.uid==selected[0])
        from .widgets import text_view
        window=tk.Toplevel(self)
        window.title('任务设置 · '+task.uid)
        window.geometry('900x640')
        text_view(window,json.dumps(task.request,ensure_ascii=False,indent=2),mono=True).pack(fill='both',expand=True)

    def show_log(self):
        window = tk.Toplevel(self)
        window.title('计算日志')
        window.geometry('1000x600')
        from .widgets import text_view
        text_view(window, '\n'.join(self.log_lines) or '（暂无日志）', mono=True).pack(fill='both', expand=True)

    # ------------------------------------------------------------ dialogs
    def show_trial_table(self):
        recordings = self.view.selected_recordings()
        if not recordings:
            messagebox.showinfo('未选择记录', '请先选择记录。')
            return
        try:
            criteria = self.criteria()
        except (TypeError, ValueError) as exc:
            messagebox.showerror('筛选参数无效', str(exc))
            return
        window = tk.Toplevel(self)
        window.title('Trial 筛选详情')
        window.geometry('1200x700')
        notebook = ttk.Notebook(window)
        notebook.pack(fill='both', expand=True, padx=8, pady=8)
        for recording in recordings:
            cohort = apply_filter(get_session(recording), criteria)
            s = cohort.summary()
            frame = ttk.Frame(notebook, padding=8)
            lines = [f"保留 {s['included_trials']} / {s['total_trial_rows']} 行 trial · 成功 {s['success_trials']} · "
                     f"block {s['blocks']} · 八方向 {s['direction_counts']}"]
            lines += [f'  {REASON_LABELS.get(k, k)}：{v}' for k, v in s['exclusion_counts'].items()]
            ttk.Label(frame, text='\n'.join(lines), justify='left').pack(anchor='w', pady=(0, 6))
            table_view(frame, cohort.exclusion_table()).pack(fill='both', expand=True)
            notebook.add(frame, text=recording.label)

    # ------------------------------------------------------------ prefs
    def save_preferences(self):
        p = self.prefs
        p.update(data_root=self.data_root.get(), mode=self.mode.get(), geometry=self.geometry(),
                 data_types=[t for t, v in self.type_vars.items() if v.get()], layout_path=self.layout_path.get(),
                 use_layout=self.use_layout.get(), output_root=self.output_root.get(),
                 font_percent=self.figure_font_percent(),
                 export_layout=self.export_layout.get(),
                 output_follow_data=self.output_follow.get())
        p['filter'] = self.filter.to_dict()
        self.single_view.save_preferences(p)
        self.cross_view.save_preferences(p)
        C.save_preferences(p)

    def close(self):
        self.save_preferences()
        self.task_queue.close()
        self.destroy()


# =================================================================== shared logic
class StatusMixin:
    """Background signature/status computation for (section, recordings, params)."""

    def check_status(self, items, callback, *, with_summary=False):
        """items: [(token, section, recordings, raw_params)] → callback(token, status, folder, record)."""
        app = self.app
        try:
            criteria = app.criteria()
        except (TypeError, ValueError):
            return
        layout, root = app.effective_layout(), app.output_path()      # lookup only, nothing is created
        font_percent = app.figure_font_percent()
        items = [(token, section, recordings, raw, app._job_channel_config(section, recordings))
                 for token, section, recordings, raw in items]
        generation = self._status_generation = getattr(self, '_status_generation', 0) + 1

        def apply(*args):
            if generation == self._status_generation:
                callback(*args)

        def report(token, status, folder=None, signature='', summary=None):
            args = (token, status, folder, signature)
            if with_summary:
                args += (summary,)
            app.events.put(dict(event='_status', call=(apply, *args)))

        def work():
            for token, section, recordings, raw, config in items:
                summary = None
                try:
                    params = section.resolve_params(raw)
                except Exception:
                    report(token, 'invalid')
                    continue
                try:
                    inputs = storage.signature_inputs(section, recordings, criteria, params, layout,
                                                      channel_config_path=config, font_percent=font_percent)
                    signature = storage.fingerprint(inputs)
                    folder = storage.result_folder(root, section, recordings, signature)
                    if with_summary:
                        summary = storage.load_cached_summary(folder, signature)
                        status = 'cached' if summary is not None else 'missing'
                    else:
                        status = 'cached' if storage.exists_cached(folder, signature) else 'missing'
                except Exception:
                    status, folder, signature = 'missing', None, ''
                report(token, status, folder, signature, summary)
        threading.Thread(target=work, daemon=True).start()


def recording_tree(master, columns, heights=12, selectmode='browse', *, stretch_columns=False):
    frame = ttk.Frame(master)
    tree = ttk.Treeview(frame, columns=[c[0] for c in columns], show='headings', height=heights,
                        selectmode=selectmode)
    for key, label, width in columns:
        tree.heading(key, text=label)
        tree.column(key, width=px(tree, width), anchor='w', stretch=stretch_columns or key in ('label', 'file'))
    bar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
    tree.configure(yscrollcommand=bar.set)
    tree.pack(side='left', fill='both', expand=True)
    bar.pack(side='right', fill='y')
    tree.tag_configure('cached', foreground=OK)
    tree.tag_configure('missing', foreground=MUTED)
    tree.tag_configure('error', foreground='#b3261e')
    return frame, tree


# =================================================================== single-day
def split_panes(master, orient):
    return tk.PanedWindow(master, orient=orient, background=theme.BACKGROUND,
                          sashwidth=px(master, 8), sashrelief='raised', borderwidth=0)


class SingleView(ttk.Frame, StatusMixin):
    def __init__(self, master, app: App, config_parent=None):
        super().__init__(master)
        self.app = app
        self.recordings = []
        self.forms: dict[str, ParamForm] = {}
        self.status_of: dict[str, tuple] = {}
        saved = app.prefs.get('single', {})
        self.saved_params = saved.get('params', {})
        self.saved_recording = saved.get('recording')
        self.saved_recordings = saved.get('recordings') or ([self.saved_recording] if self.saved_recording else [])
        self.saved_section = saved.get('section', 's01')
        self.include_slow_sections = tk.BooleanVar(master=self, value=saved.get('include_slow_sections', False))
        workspace = split_panes(self, 'horizontal')
        workspace.pack(fill='both', expand=True)
        sidebar = split_panes(workspace, 'vertical')
        workspace.add(sidebar, width=px(self, 300), minsize=px(self, 210), stretch='never')

        left = ttk.Frame(sidebar)
        sidebar.add(left, minsize=px(self, 125), stretch='always')
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        recording_header = ttk.Frame(left)
        recording_header.grid(row=0, column=0, sticky='ew')
        ttk.Label(recording_header, text='记录', style='Title.TLabel').pack(side='left')
        ttk.Button(recording_header, text='全选', style='Small.TButton',
                   command=self.select_all_recordings).pack(side='right')
        frame, self.rec_tree = recording_tree(left, [('date', '日期', 80), ('type', '类型', 72), ('run', 'Run', 56),
                                                     ('condition', '实验条件', 210),
                                                     ('fpga', 'FPGA', 46)], heights=9, selectmode='extended',
                                              stretch_columns=True)
        frame.grid(row=1, column=0, sticky='nsew')
        self.rec_tree.bind('<<TreeviewSelect>>', lambda e: self.on_recording())
        self.card = ttk.Label(left, text='', style='Muted.TLabel', justify='left', wraplength=px(master, 290))
        self.card.grid(row=2, column=0, sticky='ew', pady=(4, 8))
        left.bind('<Configure>', lambda e: self.card.configure(wraplength=max(1, e.width - px(left, 10))))
        self.config_display = None
        if config_parent is not None:
            ttk.Label(config_parent, text='通道配置 INI / LOG').pack(side='left', padx=(10, 4))
            self.config_display = tk.StringVar(value='未选择记录')
            ttk.Entry(config_parent, textvariable=self.config_display, state='readonly', width=52).pack(side='left')
            self.config_choose = ttk.Button(config_parent, text='…', width=3, state='disabled',
                                            command=self.choose_channel_config)
            self.config_choose.pack(side='left', padx=(4, 0))
            self.config_reset = ttk.Button(config_parent, text='自动', state='disabled',
                                           command=self.reset_channel_config)
            self.config_reset.pack(side='left', padx=(4, 0))
        section_panel = ttk.Frame(sidebar)
        sidebar.add(section_panel, minsize=px(self, 85), stretch='always')
        ttk.Label(section_panel, text='单记录分析项目', style='Title.TLabel').pack(anchor='w')
        frame = ttk.Frame(section_panel)
        frame.pack(fill='both', expand=True)
        self.sec_tree = ttk.Treeview(frame, columns=('status',), show='tree headings', selectmode='browse')
        self.sec_tree.heading('#0', text='Section')
        self.sec_tree.heading('status', text='状态')
        self.sec_tree.column('#0', width=px(frame, 222))
        self.sec_tree.column('status', width=px(frame, 66), anchor='w', stretch=False)
        self.sec_tree.tag_configure('cached', foreground=OK)
        self.sec_tree.tag_configure('missing', foreground=MUTED)
        self.sec_tree.tag_configure('error', foreground='#b3261e')
        self.sec_tree.tag_configure('running', foreground=WARN)
        self.sec_tree.tag_configure('queued', foreground=WARN)
        bar = ttk.Scrollbar(frame, orient='vertical', command=self.sec_tree.yview)
        self.sec_tree.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        self.sec_tree.pack(side='left', fill='both', expand=True)
        self.sec_tree.bind('<<TreeviewSelect>>', lambda e: self.on_section())
        for s in sorted(sections('single'), key=lambda section: section.id == 's12'):
            self.sec_tree.insert('', 'end', iid=s.id, text=section_display(s), values=('…',))

        result_panes = split_panes(workspace, 'vertical')
        workspace.add(result_panes, minsize=px(self, 650), stretch='always')
        right = ttk.Frame(result_panes)
        result_panes.add(right, minsize=px(self, 105), stretch='never')
        right.columnconfigure(0, weight=1)
        head = ttk.Frame(right)
        head.grid(row=0, column=0, sticky='ew')
        self.title_var = tk.StringVar()
        ttk.Label(head, textvariable=self.title_var, style='Title.TLabel').pack(side='left')
        ttk.Label(head, text='   版本').pack(side='left')
        self.version = tk.StringVar()
        self.version_box = ttk.Combobox(head, textvariable=self.version, width=5, state='readonly')
        self.version_box.pack(side='left', padx=4)
        self.version_box.bind('<<ComboboxSelected>>', lambda e: self.on_section(keep_version=True))
        ttk.Button(head, text='加载 / 计算', style='Accent.TButton', command=self.compute_current
                   ).pack(side='right', padx=6)
        ttk.Button(head, text='结果文件夹', command=self.open_folder).pack(side='right')
        ttk.Checkbutton(head, text='批量包含耗时项 S10 / S12 / S11',
                        variable=self.include_slow_sections).pack(side='right', padx=6)
        self.desc = ttk.Label(right, text='', style='Muted.TLabel', wraplength=1000, justify='left')
        self.desc.grid(row=1, column=0, sticky='ew', pady=(2, 4))
        self.form_holder = ttk.LabelFrame(right, text=' 参数 ', padding=(8, 4))
        self.form_holder.grid(row=2, column=0, sticky='ew', pady=(0, 6))
        self.result = ResultView(result_panes)
        result_panes.add(self.result, minsize=px(self, 125), stretch='always')

    # ---------------------------------------------------------- selection
    def select_all_recordings(self):
        rows = self.rec_tree.get_children()
        if rows:
            if not self.rec_tree.focus():
                self.rec_tree.focus(rows[0])
            self.rec_tree.selection_set(*rows)

    def set_recordings(self, recordings):
        previous = self.selected_recordings()
        keep = [r.key for r in previous] if previous else self.saved_recordings
        focus = self.rec_tree.focus() or self.saved_recording
        self.recordings = recordings
        self.rec_tree.delete(*self.rec_tree.get_children())
        for r, label in zip(recordings, recording_labels(recordings)):
            tag = label if getattr(r, 'condition', '') else r.data_type
            self.rec_tree.tag_configure(tag, foreground=session_label_color(label))
            self.rec_tree.insert('', 'end', iid=r.key, values=(r.date, r.data_type, r.run, getattr(r, 'condition', '') or '—',
                                                               '有' if r.has_fpga_clock else '—'), tags=(tag,))
        targets = [key for key in keep if key in self.rec_tree.get_children()]
        if not targets and recordings:
            targets = [recordings[0].key]
        if targets:
            self.rec_tree.selection_set(targets)
            focus = focus if focus in targets else targets[0]
            self.rec_tree.focus(focus)
            self.rec_tree.see(focus)
        if self.saved_section in self.sec_tree.get_children() and not self.sec_tree.selection():
            self.sec_tree.selection_set(self.saved_section)
        self.on_recording()

    def selected_recordings(self):
        sel = set(self.rec_tree.selection())
        return [r for r in self.recordings if r.key in sel]

    def preview_recordings(self):
        """Multi-selection controls computation; focus chooses the one result shown."""
        selected = self.selected_recordings()
        focus = self.rec_tree.focus()
        return next(([r] for r in selected if r.key==focus),selected[:1])

    def current_section(self):
        sel = self.sec_tree.selection()
        if not sel:
            return None
        options = versions(sel[0])
        return next((s for s in options if s.version == self.version.get()), options[-1])

    def on_recording(self):
        recs = self.preview_recordings()
        if recs:
            r = recs[0]
            self.card.configure(text=f'已选 {len(self.selected_recordings())} 份 · 当前预览\n{r.path.name}\n任务 {r.task_mode} · '
                                     f'{"在线 decoder 流" if r.online else "离线"} · '
                                     f'{"含 FPGA 时钟" if r.has_fpga_clock else "无 FPGA 时钟"}'
                                     + (f'\n{r.condition}' if getattr(r, 'condition', '') else ''))
        else:
            self.card.configure(text='')
        if self.config_display is not None:
            self.config_display.set((self.channel_config_for(recs[0]) or '未找到通道配置') if recs else '未选择记录')
            self.config_choose.configure(state='normal' if recs else 'disabled')
            self.config_reset.configure(state='normal' if recs and self._config_key(recs[0]) in
                                        self.app.prefs.get('channel_config_overrides', {}) else 'disabled')
        self.refresh()

    @staticmethod
    def _config_key(recording):
        return str((day_folder(recording.path) or recording.path.parent).resolve())

    def channel_config_for(self, recording):
        override = self.app.prefs.get('channel_config_overrides', {}).get(self._config_key(recording))
        return override if override is not None else recording.channel_config

    def choose_channel_config(self):
        recs = self.preview_recordings()
        if not recs:
            return
        recording = recs[0]
        current = self.channel_config_for(recording)
        folder = Path(current).parent if current else day_folder(recording.path) or recording.path.parent
        selected = filedialog.askopenfilename(parent=self, title=f'选择 {recording.date} 的通道配置',
                                              initialdir=folder, filetypes=[('通道配置 INI / LOG', ('*.ini', '*.log')),
                                                                           ('INI', '*.ini'), ('Neuralinsight 日志', '*.log')])
        if selected:
            self.app.prefs.setdefault('channel_config_overrides', {})[self._config_key(recording)] = selected
            self._channel_config_changed()

    def reset_channel_config(self):
        recs = self.preview_recordings()
        if recs:
            self.app.prefs.get('channel_config_overrides', {}).pop(self._config_key(recs[0]), None)
            self._channel_config_changed()

    def _channel_config_changed(self):
        for section in sections('single'):
            if storage.uses_channel_config(section):
                self.status_of.pop(section.id, None)
        self.app.save_preferences()
        self.on_recording()

    def on_section(self, keep_version=False):
        section_id = (self.sec_tree.selection() or [None])[0]
        if section_id is None:
            return
        options = [s.version for s in versions(section_id)]
        self.version_box.configure(values=options)
        if not keep_version or self.version.get() not in options:
            self.version.set(options[-1])
        section = self.current_section()
        # Preserve edited source settings before replacing their form widgets.
        for key, existing in self.forms.items():
            self.saved_params[key] = existing.raw()
        self.title_var.set(f'{section_display(section)}  ({section.version})')
        self.desc.configure(text=section.description)
        for child in self.form_holder.winfo_children():
            child.destroy()
        key = f'{section.id}_{section.version}'
        form = ParamForm(self.form_holder, section, self.saved_params.get(key))
        form.pack(fill='x')
        self.forms[key] = form
        controls = self.form_holder.master
        controls.update_idletasks()
        controls.master.paneconfigure(controls, minsize=controls.winfo_reqheight())
        self.refresh_status()
        self.show_current()

    def refresh(self):
        self.refresh_status()
        self.show_current()

    # ---------------------------------------------------------- status/display
    def refresh_status(self):
        recs = self.preview_recordings()
        if not recs:
            return
        items = []
        for s in sections('single'):
            raw = self._raw_params(s)
            items.append(((s.id,recs[0].key), s, recs, raw))
        self.check_status(items, self._set_status)

    def _set_status(self, section_id, status, folder, signature):
        recs = self.preview_recordings()
        if isinstance(section_id,tuple):
            section_id, recording_key = section_id
            if not recs or recording_key!=recs[0].key:
                return
        if section_id not in self.sec_tree.get_children():
            return
        self.status_of[section_id] = (status, folder, signature, recs[0].key if recs else None)
        section=next(s for s in sections('single') if s.id==section_id)
        selected=self.current_section()
        if selected and selected.id==section_id:
            section=selected
        queued=self.app.queued_status(section,recs,self._raw_params(section))
        display=queued or status
        self.sec_tree.item(section_id, values=(STATUS_TEXT[display],), tags=(display,))
        if section_id == (self.sec_tree.selection() or [None])[0]:
            self.show_current()

    def show_current(self):
        section, recs = self.current_section(), self.preview_recordings()
        if section is None or not recs:
            self.result.message('请选择一份记录和一个 section。')
            return
        state = self.status_of.get(section.id)
        if state and state[0] == 'cached' and state[3] == recs[0].key:
            record = storage.load_cached(state[1], state[2])
            if record:
                self.result.show(record, state[1], load_guide(section))
                self.app.status.set(f'{section.id.upper()} · 已读取存储结果 · {state[1]}')
                return
        self.result.message(guide=load_guide(section), text=f'{section_display(section)}\n\n记录：{recs[0].label}\n\n尚无与当前参数、筛选条件和代码版本一致的存储结果。'
                            '\n点击「加载 / 计算」：先搜索已存储结果，缺失时才计算并保存 PNG / SVG / PDF / CSV。')

    # ---------------------------------------------------------- compute
    def _raw_params(self, section):
        form = self.forms.get(f'{section.id}_{section.version}')
        raw = dict(form.raw() if form else self.saved_params.get(f'{section.id}_{section.version}', {}))
        if section.source_sections:
            raw['_source_sections'] = {sid: dict(version=versions(sid)[-1].version,
                                                 params=self._raw_params(versions(sid)[-1]))
                                       for sid in section.source_sections}
        return raw

    def _job(self, section, recording=None):
        raw = self._raw_params(section)
        section.resolve_params(raw)  # validate this analysis and inherited settings before starting
        return section, [recording] if recording else self.preview_recordings(), raw

    def compute_current(self):
        section = self.current_section()
        if section is None or not self.selected_recordings():
            messagebox.showwarning('未选择', '请选择一份记录和一个 section。')
            return
        try:
            jobs = [self._job(section,r) for r in self.selected_recordings()
                    if not ('online' in section.requires and not r.online)
                    and not ('channel_config' in section.requires and not self.channel_config_for(r))]
        except ValueError as exc:
            messagebox.showerror('参数无效', str(exc))
            return
        if jobs:
            self._start(jobs)
        else:
            messagebox.showwarning('无适用记录','所选记录不满足当前 section 的在线流 / 通道配置 INI 或 LOG 要求。')

    def compute_all(self):
        """Queue all applicable single-day sections for the selected recordings."""
        recordings = self.selected_recordings()
        if not recordings:
            messagebox.showwarning('未选择', '请选择一份记录。')
            return
        jobs, skipped, invalid = [], [], []
        included_sections = [s for s in sections('single')
                             if self.include_slow_sections.get() or s.id not in SLOW_SINGLE_SECTIONS]
        for recording in recordings:
            for s in included_sections:
                if 'channel_config' in s.requires and not self.channel_config_for(recording):
                    skipped.append(f'{s.id.upper()}（{recording.date} 无通道配置 INI / LOG）')
                    continue
                if 'online' in s.requires and not recording.online:
                    continue
                try:
                    jobs.append(self._job(s, recording))
                except ValueError as exc:
                    invalid.append(f'{s.id.upper()}（{recording.date} 参数无效：{exc}）')
        if skipped:
            self.app.status.set('已跳过：' + '、'.join(sorted(set(skipped))))
        if invalid:
            messagebox.showwarning('已跳过参数无效的任务', '\n'.join(sorted(set(invalid))))
        if jobs:
            self._start(jobs)

    def _start(self, jobs):
        current = self.preview_recordings()
        for s, recs, _ in jobs:
            if recs == current:
                self.sec_tree.item(s.id, values=(STATUS_TEXT['queued'],), tags=('queued',))
        if not self.app.run_jobs(jobs, self._on_event):
            self.refresh_status()

    def _on_event(self, event):
        kind = event['event']
        if kind == 'start':
            self.refresh_status()
        elif kind in ('complete', 'error'):
            section_id, _, names = event['task'].partition(':')
            recs = self.preview_recordings()
            paths = event.get('recording_paths')
            mine = bool(recs) and (paths==[str(recs[0].path)] if paths else names == recs[0].path.name)
            if kind == 'error' and mine:
                self.sec_tree.item(section_id, values=(STATUS_TEXT['error'],), tags=('error',))
                if section_id == (self.sec_tree.selection() or [None])[0]:
                    self.result.error(f'{event["message"]}\n\n{event.get("traceback", "")}')
            elif mine:
                self.refresh_status()
        elif kind in ('exit','cancelled'):
            self.refresh_status()

    def open_folder(self):
        section = self.current_section()
        state = self.status_of.get(section.id) if section else None
        open_path(state[1] if state and state[0] == 'cached' else self.app.ensure_output())

    def save_preferences(self, prefs):
        params = dict(self.saved_params)
        for key, form in self.forms.items():
            params[key] = form.raw()
        recs = self.selected_recordings()
        preview = self.preview_recordings()
        section = self.current_section()
        prefs['single'] = dict(params=params, recording=preview[0].key if preview else self.saved_recording,
                               recordings=[r.key for r in recs],
                               include_slow_sections=self.include_slow_sections.get(),
                               section=section.id if section else self.saved_section)


# =================================================================== cross-day
class CrossSingleOverview(ttk.Frame, StatusMixin):
    """Compare each checked recording's independently computed single-day metrics."""

    def __init__(self, master, cross):
        super().__init__(master, padding=(8, 6))
        self.cross = cross
        self.app = cross.app
        self.sections = [section for section in sections('single') if section.id not in ('s13', 's14')]
        bar = ttk.Frame(self)
        bar.pack(fill='x')
        ttk.Label(bar, text='来源 Section').pack(side='left')
        self.choice = ttk.Combobox(bar, state='readonly', width=36,
                                   values=[section_display(section) for section in self.sections])
        self.choice.current(0)
        self.choice.pack(side='left', padx=6)
        self.choice.bind('<<ComboboxSelected>>', lambda event: self.refresh())
        self.compute_button = ttk.Button(bar, text='加载 / 计算所选记录', style='Accent.TButton',
                                         command=self.compute)
        self.compute_button.pack(side='left', padx=6)
        self.status = tk.StringVar()
        ttk.Label(self, text='逐记录独立计算 · 使用单记录分析参数和顶栏筛选 · 同日不同 run 分开显示',
                  style='Muted.TLabel').pack(anchor='w', pady=(6, 0))
        ttk.Label(self, textvariable=self.status, style='Muted.TLabel').pack(anchor='w', pady=(0, 4))
        self.plot = ttk.Frame(self)
        self.plot.pack(fill='both', expand=True)

    def section(self):
        return self.sections[self.choice.current()]

    def unavailable(self, section, recording):
        if 'online' in section.requires and not recording.online:
            return '仅在线'
        if 'channel_config' in section.requires and not self.app.single_view.channel_config_for(recording):
            return '缺少通道配置 INI / LOG'
        return ''

    def refresh(self):
        self._status_generation = getattr(self, '_status_generation', 0) + 1
        for widget in self.plot.winfo_children():
            widget.destroy()
        recordings = self.cross.selected_recordings()
        section = self.section()
        available = [(i, r) for i, r in enumerate(recordings) if not self.unavailable(section, r)]
        self.compute_button.configure(state='normal' if available else 'disabled')
        if not recordings:
            self.status.set('在上方勾选记录后查看对比。')
            return
        if not available:
            self.status.set(f'所选记录均不适用（{self.unavailable(section, recordings[0])}）。')
            return
        self.status.set('正在查找匹配的单记录结果…')
        raw = self.app.single_view._raw_params(section)
        summaries = [None] * len(recordings)
        statuses = {}

        def receive(index, status, folder, signature, summary):
            summaries[index] = summary
            statuses[index] = status
            if len(statuses) != len(available):
                return
            cached = sum(s == 'cached' for s in statuses.values())
            queued = sum(s == 'missing' and bool(self.app.queued_status(section, [recordings[i]], raw))
                         for i, s in statuses.items())
            invalid = sum(s == 'invalid' for s in statuses.values())
            missing = len(available) - cached - queued - invalid
            self.status.set(f'已读取 {cached}/{len(recordings)} 份 · 待计算 {missing} · 排队/计算中 {queued}'
                            f' · 参数无效 {invalid} · 不适用 {len(recordings) - len(available)}')
            if cached:
                self.draw(comparison_figure(section.id, recordings, summaries,
                                            font_percent=self.app.figure_font_percent()))

        items = [(i, section, [recording], raw) for i, recording in available]
        self.check_status(items, receive, with_summary=True)

    def draw(self, figure):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        canvas = FigureCanvasTkAgg(figure, master=self.plot)
        toolbar = NavigationToolbar2Tk(canvas, self.plot, pack_toolbar=False)
        toolbar.pack(side='bottom', fill='x')
        canvas.get_tk_widget().pack(fill='both', expand=True)
        canvas.draw()

    def compute(self):
        section = self.section()
        try:
            jobs = [self.app.single_view._job(section, r) for r in self.cross.selected_recordings()
                    if not self.unavailable(section, r)]
        except ValueError as exc:
            messagebox.showerror('参数无效', str(exc))
            return
        if self.app.run_jobs(jobs, None):
            self.refresh()

    def source_finished(self, section_id):
        if self.cross.overview_selected() and self.section().id == section_id:
            self.refresh()


class CrossView(ttk.Frame, StatusMixin):
    SECTION_LAYOUT = 2  # X02 was inserted before TDR in layout 2.

    def __init__(self, master, app: App):
        super().__init__(master)
        self.app = app
        self.recordings = []
        saved = app.prefs.get('cross', {})
        self.saved_params = dict(saved.get('params', {}))
        if saved.get('section_layout', 1) == 1:
            for old in ('x05', 'x04', 'x03', 'x02'):
                new = f'x{int(old[1:]) + 1:02}'
                for key in tuple(self.saved_params):
                    if key.startswith(f'{old}_'):
                        self.saved_params[new + key[3:]] = self.saved_params.pop(key)
        self.checked: set[str] = set(saved.get('recordings', []))
        self.tabs: dict[str, dict] = {}
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.LabelFrame(self, text=' 记录选择 · 勾选至少两份 ', padding=(8, 6))
        top.grid(row=0, column=0, sticky='ew')
        top.columnconfigure(0, weight=3)
        top.columnconfigure(1, weight=2)
        frame, self.rec_tree = recording_tree(top, [('check', '✓', 28), ('date', '日期', 80), ('type', '类型', 76),
                                                    ('run', 'Run', 60), ('condition', '实验条件', 210),
                                                    ('fpga', 'FPGA', 46), ('file', '文件', 260)],
                                              heights=7, selectmode='none')
        legend = ttk.Frame(top)
        legend.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 6))
        for index, (kind, (name, color, marker)) in enumerate({**SESSION_STYLES, **CONDITION_STYLES}.items()):
            item = ttk.Frame(legend)
            item.grid(row=index % 2, column=index // 2, sticky='w', padx=(0, 18))
            ttk.Label(item, text='■' if marker == 's' else '●', foreground=color).pack(side='left')
            ttk.Label(item, text=name, foreground='black' if kind == 'CO8' else color).pack(side='left', padx=5)
        frame.grid(row=1, column=0, sticky='nsew')
        self.rec_tree.bind('<Button-1>', self.toggle)
        side = ttk.Frame(top)
        side.grid(row=1, column=1, sticky='nsew', padx=(12, 0))
        buttons = ttk.Frame(side)
        buttons.pack(fill='x')
        for label, command in (('全选', self.check_all), ('全不选', self.check_none),
                               ('每天第一份', self.check_first_per_day)):
            ttk.Button(buttons, text=label, style='Small.TButton', command=command).pack(side='left', padx=(0, 4))
        offline_buttons = ttk.Frame(side)
        offline_buttons.pack(fill='x', pady=(4, 0))
        for label, modes in (('离线 CO8', ('centerout',)), ('离线 Grid', ('grid',)),
                             ('离线 CO8 + Grid', ('centerout', 'grid'))):
            ttk.Button(offline_buttons, text=label, style='Small.TButton',
                       command=lambda modes=modes: self.check_offline(modes)).pack(side='left', padx=(0, 4))
        self.timeline = tk.Canvas(side, height=96, background='white', highlightthickness=1,
                                  highlightbackground='#c9d3df')
        self.timeline.pack(fill='x', pady=(8, 4))
        self.timeline.bind('<Configure>', lambda e: self.draw_timeline())
        self.selection_text = tk.StringVar()
        ttk.Label(side, textvariable=self.selection_text, style='Muted.TLabel', wraplength=520,
                  justify='left').pack(anchor='w')
        row = ttk.Frame(side)
        row.pack(anchor='w', pady=(6, 0))
        ttk.Button(row, text='计算全部跨记录项目', command=self.compute_all).pack(side='left')
        ttk.Button(row, text='一键报告', style='Accent.TButton',
                   command=self.compute_report).pack(side='left', padx=6)
        ttk.Button(row, text='报告文件夹', command=self.open_report_folder).pack(side='left')
        self.report_status = tk.StringVar()
        self.report_folder = None
        ttk.Label(side, textvariable=self.report_status, style='Muted.TLabel',
                  wraplength=520, justify='left').pack(anchor='w', pady=(4, 0))

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, sticky='nsew', pady=(8, 0))
        for s in sections('cross'):
            self._build_tab(s)
        self.single_overview = CrossSingleOverview(self.notebook, self)
        self.notebook.add(self.single_overview, text=' 单记录指标对比 ')
        self.notebook.bind('<<NotebookTabChanged>>', lambda e: self.show_selected_tab())

    def _build_tab(self, section):
        frame = ttk.Frame(self.notebook, padding=(8, 6))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        head = ttk.Frame(frame)
        head.grid(row=0, column=0, sticky='ew')
        ttk.Label(head, text=section_display(section), style='Title.TLabel').pack(side='left')
        version = tk.StringVar(value=section.version)
        box = ttk.Combobox(head, textvariable=version, values=[s.version for s in versions(section.id)],
                           state='readonly', width=5)
        box.pack(side='left', padx=8)
        status = tk.StringVar(value='…')
        ttk.Label(head, textvariable=status, style='Muted.TLabel').pack(side='left', padx=8)
        ttk.Button(head, text='加载 / 计算', style='Accent.TButton',
                   command=lambda i=section.id: self.compute([i])).pack(side='right')
        ttk.Button(head, text='结果文件夹', command=lambda i=section.id: self.open_folder(i)).pack(side='right', padx=6)
        ttk.Label(frame, text=section.description, style='Muted.TLabel', wraplength=1300, justify='left'
                  ).grid(row=1, column=0, sticky='nw')
        holder = ttk.Frame(frame)
        holder.grid(row=2, column=0, sticky='nsew', pady=(4, 0))
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(1, weight=1)
        form = ParamForm(holder, section, self.saved_params.get(f'{section.id}_{section.version}'), columns=5)
        form.grid(row=0, column=0, sticky='ew', pady=(0, 6))
        result = ResultView(holder, placeholder='勾选 ≥2 份记录后点击「加载 / 计算」')
        result.grid(row=1, column=0, sticky='nsew')
        self.notebook.add(frame, text=f' {section.display.split(" · ", 1)[0]} ')
        self.tabs[section.id] = dict(section=section, frame=frame, form=form, result=result, status=status,
                                     version=version, state=None)
        box.bind('<<ComboboxSelected>>', lambda e, i=section.id: self.change_version(i))

    def change_version(self, section_id):
        tab = self.tabs[section_id]
        tab['section'] = next(s for s in versions(section_id) if s.version == tab['version'].get())
        tab['form'].destroy()
        tab['form'] = ParamForm(tab['result'].master, tab['section'],
                                self.saved_params.get(f'{section_id}_{tab["version"].get()}'), columns=5)
        tab['form'].grid(row=0, column=0, sticky='ew', pady=(0, 6))
        self.refresh()

    # ---------------------------------------------------------- selection
    def set_recordings(self, recordings):
        self.recordings = recordings
        keys = {r.key for r in recordings}
        self.checked &= keys
        self.rec_tree.delete(*self.rec_tree.get_children())
        for r, label in zip(recordings, recording_labels(recordings)):
            color = session_label_color(label)
            tag = label if getattr(r, 'condition', '') else r.data_type
            self.rec_tree.tag_configure(tag, foreground=color)
            run = r.run
            self.rec_tree.insert('', 'end', iid=r.key, values=('☑' if r.key in self.checked else '☐', r.date,
                                                               r.data_type, run, getattr(r, 'condition', '') or '—',
                                                               '有' if r.has_fpga_clock else '—',
                                                               r.path.name), tags=(tag,))
        self.selection_changed()

    def toggle(self, event):
        row = self.rec_tree.identify_row(event.y)
        if not row:
            return
        self.checked.symmetric_difference_update({row})
        self.rec_tree.set(row, 'check', '☑' if row in self.checked else '☐')
        self.selection_changed()

    def _set_checked(self, keys):
        self.checked = set(keys)
        for r in self.recordings:
            self.rec_tree.set(r.key, 'check', '☑' if r.key in self.checked else '☐')
        self.selection_changed()

    def check_all(self):
        self._set_checked(r.key for r in self.recordings)

    def check_none(self):
        self._set_checked([])

    def check_where(self, predicate):
        self._set_checked(r.key for r in self.recordings if predicate(r))

    def check_offline(self, modes):
        self.check_where(lambda r: not r.online and r.task_mode in modes)

    def check_first_per_day(self):
        seen, keys = set(), []
        for r in self.recordings:
            if r.date not in seen and not r.online:
                seen.add(r.date)
                keys.append(r.key)
        self._set_checked(keys)

    def selected_recordings(self):
        return [r for r in self.recordings if r.key in self.checked]

    def selection_changed(self):
        recs = self.selected_recordings()
        days = sorted({r.date for r in recs})
        if recs:
            span = (date.fromisoformat(days[-1]) - date.fromisoformat(days[0])).days if days[0] else 0
            self.selection_text.set(f'已选 {len(recs)} 份 · {len(days)} 个日期 · 跨度 {span} 天（{days[0]} → {days[-1]}）'
                                    + (' · 含在线记录' if any(r.online for r in recs) else ''))
        else:
            self.selection_text.set('尚未勾选记录')
        self.draw_timeline()
        self.refresh()

    def draw_timeline(self):
        c = self.timeline
        c.delete('all')
        if not self.recordings:
            return
        width, height = max(c.winfo_width(), 200), 96
        dates = sorted({r.date for r in self.recordings if r.date})
        if not dates:
            return
        d0, d1 = date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])
        span = max((d1 - d0).days, 1)
        x = lambda d: 30 + (width - 60) * (date.fromisoformat(d) - d0).days / span
        c.create_line(30, 58, width - 30, 58, fill='#9fb3c8', width=2)
        per_day: dict[str, int] = {}
        for r in self.recordings:
            if not r.date:
                continue
            k = per_day.get(r.date, 0)
            per_day[r.date] = k + 1
            cx, cy = x(r.date), 50 - 13 * k
            on = r.key in self.checked
            _, colour, marker = recording_style(r)
            draw_marker = c.create_rectangle if marker == 's' else c.create_oval
            draw_marker(cx - 6, cy - 6, cx + 6, cy + 6, fill=colour if on else 'white', outline=colour)
        for d in dates:
            c.create_text(x(d), 74, text=d[5:], fill=MUTED, font=(FONT, 8))
        c.create_text(width - 30, 90, text='实心 = 已选 · 方形 = CO8 · 圆形 = Grid', anchor='e', fill=MUTED,
                      font=(FONT, 8))

    # ---------------------------------------------------------- status/display
    def overview_selected(self):
        return self.notebook.select() == str(self.single_overview)

    def show_selected_tab(self):
        if self.overview_selected():
            self.single_overview.refresh()
        else:
            self.show_current()

    def show_session_overview(self, result, recordings):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from ..core.style import finish_figure
        from ..sections.session_overview import PAGES, session_overview

        font_percent = self.app.figure_font_percent()
        key = ('session-preview', tuple(recordings), font_percent)
        if result._display_key == key:
            return
        notebook = result.notebook
        previous = notebook.select()
        previous_title = notebook.tab(previous, 'text') if previous else ''
        result.clear()
        result._display_key = key

        def draw(frame, kind):
            if frame.winfo_children():
                return
            figure, _ = session_overview(recordings, kind)
            finish_figure(figure, font_percent)
            canvas = FigureCanvasTkAgg(figure, master=frame)
            toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
            toolbar.pack(side='bottom', fill='x')
            canvas.get_tk_widget().pack(fill='both', expand=True)
            canvas.draw()

        for kind, title in PAGES:
            frame = ttk.Frame(notebook)
            frame.bind('<Map>', lambda event, frame=frame, kind=kind: draw(frame, kind))
            notebook.add(frame, text=title)
            if title == previous_title:
                notebook.select(frame)

    def current_id(self):
        selected = self.notebook.select()
        return next((i for i, tab in self.tabs.items() if str(tab['frame']) == selected), None)

    def refresh(self):
        self.refresh_status()

    def refresh_status(self):
        recs = self.selected_recordings()
        for tab in self.tabs.values():
            tab['form'].set_recording_choices(recs, lambda e: self.refresh_status())
            tab['state'] = None
            minimum = tab['section'].min_recordings
            tab['status'].set(f'需勾选 ≥{minimum} 份记录' if len(recs) < minimum else '…')
        self.show_current()
        if self.overview_selected():
            self.single_overview.refresh()
        items = [((i,tuple(r.key for r in recs)), tab['section'], recs, self._raw_params(tab))
                 for i, tab in self.tabs.items() if len(recs) >= tab['section'].min_recordings]
        if items:
            self.check_status(items, self._set_status)

    def _set_status(self, section_id, status, folder, signature):
        if isinstance(section_id,tuple):
            section_id,keys=section_id
            if keys!=tuple(r.key for r in self.selected_recordings()):
                return
        tab = self.tabs[section_id]
        tab['state'] = (status, folder, signature, tuple(r.key for r in self.selected_recordings()))
        queued=self.app.queued_status(tab['section'],self.selected_recordings(),self._raw_params(tab))
        tab['status'].set(STATUS_TEXT[queued or status])
        mark = '✓ ' if status == 'cached' else ''
        name = tab['section'].display.split(' · ', 1)[0]
        self.notebook.tab(tab['frame'], text=f' {mark}{name} ')
        if section_id == self.current_id():
            self.show_current()

    def show_current(self):
        section_id = self.current_id()
        if section_id is None:
            return
        tab = self.tabs[section_id]
        recs = self.selected_recordings()
        state = tab['state']
        if state and state[0] == 'cached' and state[3] == tuple(r.key for r in recs):
            record = storage.load_cached(state[1], state[2])
            if record:
                tab['result'].show(record, state[1], load_guide(tab['section']))
                return
        if section_id == 'x00':
            self.show_session_overview(tab['result'], recs)
            return
        if len(recs) < tab['section'].min_recordings:
            tab['result'].message(f'请在上方勾选至少 {tab["section"].min_recordings} 份记录。')
            return
        names = '\n'.join(f'  · {label}' for label in recording_labels(recs))
        tab['result'].message(guide=load_guide(tab['section']), text=f'{section_display(tab["section"])}\n\n已选记录：\n{names}\n\n尚无一致的存储结果；'
                              '点击「加载 / 计算」先搜索存储，缺失时才计算。')

    # ---------------------------------------------------------- compute
    def _raw_params(self, tab):
        raw = tab['form'].raw()
        sources = [versions(sid)[-1] for sid in tab['section'].source_sections]
        if sources:
            raw['_source_sections'] = {
                source.id: dict(version=source.version, params=self.app.single_view._raw_params(source))
                for source in sources
            }
        return raw

    def compute_current(self):
        if self.overview_selected():
            self.single_overview.compute()
        elif (section_id := self.current_id()) is not None:
            self.compute([section_id])

    def compute(self, section_ids):
        recs = self.selected_recordings()
        section_ids = [i for i in section_ids if len(recs) >= self.tabs[i]['section'].min_recordings]
        if not section_ids:
            messagebox.showwarning('记录不足', '请勾选足够的记录。')
            return
        jobs, invalid = [], []
        for i in section_ids:
            tab = self.tabs[i]
            raw = self._raw_params(tab)
            try:
                tab['section'].resolve_params(raw)
            except ValueError as exc:
                invalid.append(f'{section_display(tab["section"])}: {exc}')
                continue
            jobs.append((tab['section'], recs, raw))
            tab['status'].set(STATUS_TEXT['running'])
        if invalid:
            messagebox.showwarning('参数无效', '\n'.join(invalid))
        if not self.app.run_jobs(jobs, self._on_event):
            self.refresh_status()

    def compute_all(self):
        self.compute(list(self.tabs))

    def compute_report(self):
        from ..reporting import report_plan

        def params_for(section):
            if section.mode == 'single':
                return self.app.single_view._raw_params(section)
            return self._raw_params(self.tabs[section.id])

        try:
            jobs, report = report_plan(self.selected_recordings(), params_for)
        except ValueError as exc:
            messagebox.showwarning('无法生成报告', str(exc))
            return
        if self.app.run_jobs(jobs, self._report_event, report=report):
            self.report_status.set(f'报告已排队 · {report["name"]}')

    def _report_event(self, event):
        kind = event['event']
        if kind == 'message':
            self.report_status.set(event['message'])
        elif kind in ('complete', 'error'):
            if event.get('folder'):
                self.report_folder = Path(event['folder'])
            if kind == 'complete':
                skipped = f'（跳过 {event["skipped"]} 类缺少记录的内容）' if event.get('skipped') else ''
                self.report_status.set(f'报告已完成{skipped} · {self.report_folder}')
            else:
                self.report_status.set(event['message'])
            self.app.status.set(self.report_status.get())
        elif kind == 'cancelled' or (kind == 'exit' and event.get('cancelled')):
            self.report_status.set('一键报告已停止，已完成的计算结果仍保存在原目录')
        if kind == 'exit':
            self.refresh_status()

    def open_report_folder(self):
        from ..reporting import REPORT_ROOT
        open_path(self.report_folder or REPORT_ROOT)

    def _on_event(self, event):
        paths=event.get('recording_paths')
        if paths is not None and event['event']=='error' and paths!=[str(r.path) for r in self.selected_recordings()]:
            return
        if event['event'] == 'error':
            section_id = event['task'].split(':')[0]
            if section_id in self.tabs:
                self.tabs[section_id]['status'].set(STATUS_TEXT['error'])
                self.tabs[section_id]['result'].error(f'{event["message"]}\n\n{event.get("traceback", "")}')
        elif event['event'] in ('complete', 'exit','cancelled'):
            self.refresh_status()

    def open_folder(self, section_id):
        state = self.tabs[section_id]['state']
        open_path(state[1] if state and state[0] == 'cached' else self.app.ensure_output())

    def save_preferences(self, prefs):
        params = dict(self.saved_params)
        for i, tab in self.tabs.items():
            params[f'{i}_{tab["version"].get()}'] = tab['form'].raw()
        prefs['cross'] = dict(params=params, recordings=sorted(self.checked), section_layout=self.SECTION_LAYOUT)


def main():
    App().mainloop()
