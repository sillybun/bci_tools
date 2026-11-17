"""Reusable GUI pieces: parameter form, stored-result viewer, table view."""
from __future__ import annotations

import os
import json
import re
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import ttk

from .. import storage
from .canvas import ZoomCanvas
from .theme import CARD, FONT, MUTED, px


def open_path(path):
    path = str(path)
    if sys.platform == 'win32':
        os.startfile(path)  # noqa: S606 - user-requested local file
    elif sys.platform == 'darwin':
        subprocess.Popen(['open', path])
    else:
        subprocess.Popen(['xdg-open', path])


class FileButton(ttk.Button):
    """Ignore stray mouse releases after the result toolbar has been rebuilt.

    TButton's class binding can set ``pressed`` on Button1-Enter alone.
    Require the press to originate on this button before allowing its release
    binding to invoke an external application. Keyboard activation is unchanged.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._mouse_pressed = False
        self.bind('<ButtonPress-1>', self._press)
        self.bind('<ButtonRelease-1>', self._release)
        self.bind('<Leave>', self._leave)

    def _press(self, event):
        self._mouse_pressed = not self.instate(['disabled'])

    def _leave(self, event):
        self._mouse_pressed = False

    def _release(self, event):
        pressed, self._mouse_pressed = self._mouse_pressed, False
        if not pressed:
            self.state(['!pressed'])
            return 'break'


def table_view(master, rows, columns=None):
    frame = ttk.Frame(master)
    columns = columns or list(dict.fromkeys(k for r in rows for k in r))
    tree = ttk.Treeview(frame, columns=columns, show='headings')
    for c in columns:
        tree.heading(c, text=c)
        tree.column(c, width=px(tree, max(50, min(220, 7 * len(c)))), anchor='w', stretch=False)
    for r in rows[:5000]:
        tree.insert('', 'end', values=[r.get(c, '') for c in columns])
    ybar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
    xbar = ttk.Scrollbar(frame, orient='horizontal', command=tree.xview)
    tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
    tree.grid(row=0, column=0, sticky='nsew')
    ybar.grid(row=0, column=1, sticky='ns')
    xbar.grid(row=1, column=0, sticky='ew')
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return frame


def text_view(master, text, mono=False):
    frame = ttk.Frame(master)
    widget = tk.Text(frame, wrap='word', font=('Consolas', 10) if mono else (FONT, 10), relief='flat',
                     background=CARD, padx=12, pady=10)
    widget.insert('1.0', text)
    widget.configure(state='disabled')
    bar = ttk.Scrollbar(frame, orient='vertical', command=widget.yview)
    widget.configure(yscrollcommand=bar.set)
    widget.pack(side='left', fill='both', expand=True)
    bar.pack(side='right', fill='y')
    return frame


SERIES = re.compile(r'^(?P<name>.*\D)\s*(?P<number>\d+)$')


def series_name(title: str) -> str:
    match = SERIES.match(title.strip())
    return match['name'].strip() if match else title


def group_series(figures):
    """Consecutive figures titled '<name> <number>' with the same name form one group."""
    groups = []
    for figure in figures:
        match = SERIES.match(figure['title'].strip())
        if match and groups and len(groups[-1]) and SERIES.match(groups[-1][-1]['title'].strip()) and \
                series_name(groups[-1][-1]['title']) == match['name'].strip():
            groups[-1].append(figure)
        else:
            groups.append([figure])
    return groups


def markdown_view(master, text):
    """Read-only view of a guide.md: headings, bullets and `code` styled; formulas stay plain text."""
    frame = ttk.Frame(master)
    widget = tk.Text(frame, wrap='word', font=(FONT, 10), relief='flat', background=CARD, padx=18, pady=12,
                     spacing1=2, spacing3=2)
    widget.tag_configure('h1', font=(FONT, 15, 'bold'), foreground='#152c47', spacing1=8, spacing3=6)
    widget.tag_configure('h2', font=(FONT, 12, 'bold'), foreground='#2464a5', spacing1=14, spacing3=4)
    widget.tag_configure('h3', font=(FONT, 10, 'bold'), foreground='#152c47', spacing1=8, spacing3=2)
    widget.tag_configure('h4', font=(FONT, 10, 'bold'), foreground='#53657c', spacing1=6, spacing3=2)
    widget.tag_configure('bullet', lmargin1=14, lmargin2=30)
    widget.tag_configure('code', font=('Consolas', 10), background='#eef2f7')
    widget.tag_configure('quote', foreground=MUTED, lmargin1=14, lmargin2=14)
    for line in (text or '（该 section 暂无说明）').splitlines():
        stripped = line.lstrip()
        if stripped.startswith('#### '):
            widget.insert('end', stripped[5:] + '\n', 'h4')
            continue
        if stripped.startswith('### '):
            widget.insert('end', stripped[4:] + '\n', 'h3')
            continue
        if stripped.startswith('## '):
            widget.insert('end', stripped[3:] + '\n', 'h2')
            continue
        if stripped.startswith('# '):
            widget.insert('end', stripped[2:] + '\n', 'h1')
            continue
        tags = ()
        if stripped.startswith(('- ', '* ')):
            line, tags = '• ' + stripped[2:], ('bullet',)
        elif stripped.startswith('> '):
            line, tags = stripped[2:], ('quote',)
        parts = line.split('`')
        for i, part in enumerate(parts):
            widget.insert('end', part.replace('**', ''), tags + (('code',) if i % 2 else ()))
        widget.insert('end', '\n', tags)
    widget.configure(state='disabled')
    bar = ttk.Scrollbar(frame, orient='vertical', command=widget.yview)
    widget.configure(yscrollcommand=bar.set)
    widget.pack(side='left', fill='both', expand=True)
    bar.pack(side='right', fill='y')
    return frame


class ParamForm(ttk.Frame):
    """Grid of entries generated from Section.params; values() returns raw strings/bools."""

    def __init__(self, master, section, saved=None, columns=4):
        super().__init__(master)
        self.section = section
        self.vars = {}
        self.recording_boxes = {}
        saved = saved or {}
        for i, p in enumerate(section.params):
            cell = ttk.Frame(self)
            cell.grid(row=i // columns, column=i % columns, sticky='w', padx=(0, 18), pady=2)
            value = saved.get(p.name, p.default)
            if p.kind == 'bool':
                var = tk.BooleanVar(value=bool(value) if not isinstance(value, str) else value == 'True')
                ttk.Checkbutton(cell, text=p.label, variable=var).pack(side='left')
            else:
                ttk.Label(cell, text=p.label).pack(side='left')
                var = tk.StringVar(value='' if value is None else str(value))
                if p.kind == 'choice':
                    width = max(8, max(len(str(c)) for c in p.choices) + 2)
                    ttk.Combobox(cell, textvariable=var, values=p.choices, state='readonly', width=width
                                 ).pack(side='left', padx=4)
                elif p.kind == 'recording':
                    box = ttk.Combobox(cell, textvariable=var, values=p.choices, state='readonly', width=28)
                    box.pack(side='left', padx=4)
                    self.recording_boxes[p.name] = box
                else:
                    ttk.Entry(cell, textvariable=var, width=28 if p.kind in ('text', 'file') else 9
                              ).pack(side='left', padx=4)
            if p.help:
                ttk.Label(cell, text='ⓘ', style='Muted.TLabel').pack(side='left')
            self.vars[p.name] = var
        if not section.params:
            ttk.Label(self, text='此 section 无额外参数', style='Muted.TLabel').grid(row=0, column=0, sticky='w')

    def set_recording_choices(self, recordings, on_change):
        keys = tuple(r.key for r in recordings)
        for p in self.section.params:
            if p.kind == 'recording':
                box = self.recording_boxes[p.name]
                choices = p.choices + keys
                box.configure(values=choices)
                if self.vars[p.name].get() not in choices:
                    self.vars[p.name].set(p.default)
                box.bind('<<ComboboxSelected>>', on_change)

    def raw(self) -> dict:
        return {k: v.get() for k, v in self.vars.items()}

    def parsed(self) -> dict:
        """Validated values (raises ValueError with a readable message)."""
        values = self.section.defaults()
        raw = self.raw()
        for p in self.section.params:
            values[p.name] = p.parse(raw[p.name])
        return values


class ResultView(ttk.Frame):
    """Shows a stored result: one tab per figure (PNG preview), summary, method, tables."""

    def __init__(self, master, placeholder='选择后点击「加载 / 计算」：已存储的结果直接读取，缺失时才计算'):
        super().__init__(master)
        self.summary_text = tk.StringVar()
        self.summary_label = ttk.Label(self, textvariable=self.summary_text, wraplength=1000,
                                       justify='left', padding=(10, 8))
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)
        self.bind('<Configure>', lambda e: self.summary_label.configure(wraplength=max(120, e.width - 24)))
        self.placeholder = placeholder
        self._display_key = None
        self.message(placeholder)

    def clear(self):
        self._display_key = None
        self.summary_label.pack_forget()
        self.summary_text.set('')
        for tab in self.notebook.tabs():
            self.notebook.forget(tab)
            self.nametowidget(tab).destroy()

    def message(self, text, title='状态', guide=''):
        key = ('message', text, title, guide)
        if self._display_key == key:
            return
        self.clear()
        frame = ttk.Frame(self.notebook, style='Card.TFrame')
        ttk.Label(frame, text=text, style='CardMuted.TLabel', justify='left', wraplength=900,
                  font=(FONT, 11)).pack(anchor='nw', padx=24, pady=24)
        self.notebook.add(frame, text=title)
        if guide:
            self.notebook.add(markdown_view(self.notebook, guide), text='说明')
        self._display_key = key

    def error(self, text):
        self.clear()
        self.notebook.add(text_view(self.notebook, text, mono=True), text='错误')

    def show(self, record: dict, folder: Path, guide: str = ''):
        folder = Path(folder)
        # Status checks may deliver the same result repeatedly. Preserve the
        # current page/focus and avoid replacing controls during mouse events.
        # Include metadata/digests so a recompute at the same path still reloads.
        key = ('result', str(folder.resolve()), json.dumps(record, sort_keys=True), guide)
        if self._display_key == key:
            return
        self.clear()
        self.guide = guide
        from ..sections.result_summaries import result_sentence
        sentence = result_sentence((record.get('section') or {}).get('id'), record.get('summary') or {})
        self.summary_text.set('本次结果：' + sentence)
        self.summary_label.pack(side='top', fill='x', before=self.notebook)
        replay_file = record.get('attachments', {}).get('task_replay.json')
        if replay_file:
            from .task_replay import ReplayView
            try:
                payload = json.loads((folder / replay_file).read_text(encoding='utf-8'))
                html = record.get('attachments', {}).get('task_replay.html')
                self.notebook.add(ReplayView(self.notebook, payload, folder / html if html else None),
                                  text='任务回放')
            except (OSError, ValueError, KeyError) as exc:
                self.notebook.add(text_view(self.notebook, f'回放数据无法读取：{exc}'), text='任务回放')
        for group in group_series(record['figures']):
            if len(group) == 1:
                text = group[0]['title'][:30]
            else:
                text = f'{series_name(group[0]["title"])[:24]} ({len(group)})'
            self.notebook.add(self._figure_tab(group, folder), text=text)
        lines = []
        if record.get('notes'):
            lines += ['注：' + n for n in record['notes']] + ['']
        lines += ['一句话总结：' + sentence, '']
        lines += [f'{k}: {v}' for k, v in record.get('summary', {}).items() if k != 'one_sentence_summary']
        lines += ['', f'记录：{", ".join(record.get("recordings", []))}',
                  f'计算耗时：{record.get("seconds", 0)} s · 生成于 {record.get("created", "")}',
                  f'结果目录：{folder}', f'签名：{record["signature"][:16]}']
        self.notebook.add(text_view(self.notebook, '\n'.join(lines), mono=True), text='Summary')
        if record.get('tables'):
            self.notebook.add(self._table_tab(record['tables'], folder), text='Table View')
        if guide:   # the guide is always the very last tab
            self.notebook.add(markdown_view(self.notebook, guide), text='说明')
        self._display_key = key

    def _table_tab(self, tables, folder):
        """All section tables in one tab; only the selected CSV is loaded."""
        frame = ttk.Frame(self.notebook)
        entries = list(tables.items())
        state = dict(index=0)
        nav = ttk.Frame(frame, padding=(4, 4))
        nav.pack(side='top', fill='x')
        previous = ttk.Button(nav, text='◀ Prev', style='Small.TButton', command=lambda: step(-1))
        previous.pack(side='left')
        position = tk.StringVar()
        ttk.Label(nav, textvariable=position, width=9, anchor='center').pack(side='left', padx=6)
        following = ttk.Button(nav, text='Next ▶', style='Small.TButton', command=lambda: step(1))
        following.pack(side='left')
        choice = ttk.Combobox(nav, state='readonly', values=[name for name, _ in entries], width=50)
        choice.pack(side='left', fill='x', expand=True, padx=12)
        choice.bind('<<ComboboxSelected>>', lambda _: go(choice.current()))
        open_button = FileButton(nav, text='打开 CSV', style='Small.TButton',
                                 command=lambda: open_path(folder / entries[state['index']][1]))
        open_button.pack(side='left')
        ttk.Label(frame, text='表格预览最多显示 5000 行；完整数据可通过「打开 CSV」查看。',
                  style='Muted.TLabel').pack(side='top', anchor='w', padx=8, pady=(0, 4))
        content = ttk.Frame(frame)
        content.pack(fill='both', expand=True)

        def go(index):
            state['index'] = max(0, min(len(entries) - 1, index))
            name, rel = entries[state['index']]
            position.set(f'{state["index"] + 1} / {len(entries)}')
            choice.current(state['index'])
            previous.state(['disabled' if state['index'] == 0 else '!disabled'])
            following.state(['disabled' if state['index'] == len(entries) - 1 else '!disabled'])
            for child in content.winfo_children():
                child.destroy()
            try:
                rows = storage.read_table(folder / rel)
            except (OSError, ValueError) as exc:
                text_view(content, f'无法读取表格 {name}\n{exc}').pack(fill='both', expand=True)
                open_button.state(['disabled'])
            else:
                table_view(content, rows).pack(fill='both', expand=True)
                open_button.state(['!disabled'])

        def step(delta):
            go(state['index'] + delta)

        frame.series_step = step
        frame.series_state = state
        go(0)
        return frame

    def _figure_tab(self, figures, folder):
        """One tab for a single figure, or for a numbered series with Prev / Next navigation."""
        figures = figures if isinstance(figures, list) else [figures]
        frame = ttk.Frame(self.notebook)
        state = dict(index=0)
        current = lambda: figures[state['index']]          # noqa: E731
        zoom_text = tk.StringVar(value='整页')
        canvas = ZoomCanvas(frame, on_zoom=zoom_text.set)

        if len(figures) > 1:
            nav = ttk.Frame(frame, padding=(4, 4, 4, 0))
            nav.pack(side='top', fill='x')
            position = tk.StringVar()
            choice = tk.StringVar()
            prev_button = ttk.Button(nav, text='◀ Prev', style='Small.TButton', command=lambda: step(-1))
            prev_button.pack(side='left')
            ttk.Label(nav, textvariable=position, style='Status.TLabel', width=9, anchor='center').pack(side='left', padx=6)
            next_button = ttk.Button(nav, text='Next ▶', style='Small.TButton', command=lambda: step(1))
            next_button.pack(side='left')
            box = ttk.Combobox(nav, textvariable=choice, state='readonly', width=36,
                               values=[f['title'] for f in figures])
            box.pack(side='left', padx=(12, 0))
            box.bind('<<ComboboxSelected>>', lambda e: go(box.current()))
            ttk.Label(nav, text='← / → 翻页', style='Muted.TLabel').pack(side='left', padx=10)

        bar = ttk.Frame(frame, padding=(4, 4))
        bar.pack(side='top', fill='x')
        ttk.Button(bar, text='整页', style='Small.TButton',
                   command=lambda: canvas.fit(whole=True)).pack(side='left')
        ttk.Button(bar, text='适应宽度', style='Small.TButton', command=canvas.fit).pack(side='left', padx=(4, 0))
        ttk.Button(bar, text='100%', style='Small.TButton', command=canvas.actual).pack(side='left', padx=4)
        ttk.Label(bar, textvariable=zoom_text, style='Muted.TLabel', width=7).pack(side='left')
        ttk.Separator(bar, orient='vertical').pack(side='left', fill='y', padx=8)
        for ext in ('png', 'svg', 'pdf', 'csv'):
            FileButton(bar, text=ext.upper(), style='Small.TButton',
                       command=lambda e=ext: open_path(folder / current()['files'][e])).pack(side='left', padx=1)
        if figures[0]['files'].get('series_pdf'):
            FileButton(bar, text='整组 PDF', style='Small.TButton',
                       command=lambda: open_path(folder / current()['files']['series_pdf'])).pack(side='left', padx=4)
        menu_button = ttk.Menubutton(bar, text='数据表')
        menu = tk.Menu(menu_button, tearoff=False)
        menu_button['menu'] = menu
        menu_button.pack(side='left', padx=4)
        ttk.Button(bar, text='说明', style='Accent.TButton',
                   command=lambda: self._guide_window(current())).pack(side='left', padx=(8, 1))
        ttk.Button(bar, text='计算方式', style='Small.TButton',
                   command=lambda: self._method_window(current())).pack(side='left', padx=(4, 1))
        FileButton(bar, text='打开文件夹', style='Small.TButton',
                   command=lambda: open_path(folder / 'figures')).pack(side='left', padx=1)
        ttk.Label(bar, text='Ctrl+滚轮缩放 · 拖动平移', style='Muted.TLabel').pack(side='right')
        canvas.pack(side='top', fill='both', expand=True)

        def show_current():
            figure = current()
            data = figure.get('data_files') or {}
            menu.delete(0, 'end')
            for name, rel in data.items():
                menu.add_command(label=name, command=lambda p=folder / rel: open_path(p))
            menu_button.configure(text=f'数据表 ({len(data)})')
            menu_button.state(['!disabled' if data else 'disabled'])
            if len(figures) > 1:
                position.set(f'{state["index"] + 1} / {len(figures)}')
                choice.set(figure['title'])
                prev_button.state(['disabled' if state['index'] == 0 else '!disabled'])
                next_button.state(['disabled' if state['index'] == len(figures) - 1 else '!disabled'])
            canvas.load(folder / figure['files']['png'])

        def go(index):
            state['index'] = max(0, min(len(figures) - 1, index))
            show_current()

        def step(delta):
            go(state['index'] + delta)

        if len(figures) > 1:
            notebook = self.notebook

            def on_key(event, delta):
                # Only when this series tab is showing and the user is not typing in a field.
                if not frame.winfo_exists() or notebook.select() != str(frame):
                    return None
                if isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox, tk.Text, tk.Spinbox)):
                    return None
                step(delta)
                return 'break'
            top = frame.winfo_toplevel()
            top.bind('<Left>', lambda e: on_key(e, -1), add='+')
            top.bind('<Right>', lambda e: on_key(e, 1), add='+')
        frame.series_step = step            # used by tests
        frame.series_state = state
        show_current()
        return frame

    def _guide_window(self, figure):
        from ..sections.guide import figure_guide
        part = figure_guide(getattr(self, 'guide', ''), figure['title'])
        window = tk.Toplevel(self)
        window.title(f'说明 · {figure["title"]}')
        window.geometry(f'{px(self, 900)}x{px(self, 640)}')
        markdown_view(window, part or getattr(self, 'guide', '') or '（该图暂无说明）').pack(fill='both', expand=True)

    def _method_window(self, figure):
        window = tk.Toplevel(self)
        window.title(f'计算方式 · {figure["title"]}')
        window.geometry('900x560')
        files = '\n'.join(f'  {k}: {v}' for k, v in figure['files'].items())
        data = '\n'.join(f'  {k}: {v}' for k, v in figure.get('data_files', {}).items()) or '  （无，见主 CSV）'
        text = f'{figure["title"]}\n\n计算方式\n{figure["method"]}\n\n文件\n{files}\n\n绘图数据表\n{data}\n\n' \
               '主 CSV 以 # 开头的行记录计算方式、记录 SHA-256、筛选条件与参数；其余为图中实际绘制的数值。'
        text_view(window, text).pack(fill='both', expand=True)
