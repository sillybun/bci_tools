"""Pop-out dialog with every trial-selection setting (basic / advanced / session level)."""
from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ..core.filters import (CRITERIA_VERSION, FIELD_SPEC, GROUP_LABELS, GROUP_OF, PRIORITY, REASON_LABELS,
                            TrialFilter, apply_filter)
from ..core.session import get_session
from .theme import FONT, MUTED, px

TABS = (('basic', '基础'), ('advanced', 'Advanced'), ('session', 'Session 级'))
RULES = (
    '每个 trial 在每条标准下记录 pass / fail / flag 和原因，按固定优先级给出最终状态与主要原因：'
    + ' → '.join(GROUP_LABELS[g] for g in PRIORITY) + '。',
    '方向队列使用标准 1–7；速度队列另外要求标准 8（速度 bin 质量），只用于速度 / KF 分析。',
    'Flag（过快、命中早于窗口结束）默认不排除，仅在敏感性分析中排除。',
    '阈值写入配置文件并记录版本（' + CRITERIA_VERSION + '），在查看分析结果之前确定，不根据结果调整。',
    'Trial 级问题一律记录原因、不中止；只有 session 级数据错误（重复流不一致、时间戳不递增）才中止加载。',
)


def summarize(criteria: TrialFilter) -> str:
    c = criteria
    parts = []
    if c.exclude_incomplete:
        parts.append('排除 Incomplete')
    if c.success_only:
        parts.append('仅 Success')
    if c.overtime_enabled:
        basis = 'start→命中' if c.overtime_basis == 'start_to_acquire' else 'start→END'
        limit = f'> {c.overtime_seconds:g} s' if c.overtime_mode == 'fixed' else f'> median + {c.overtime_mad_k:g}·MAD'
        parts.append(f'Overtime（{basis} {limit}）')
    parts.append(f'窗口 {c.window_start:g}–{c.window_end:g} s')
    if c.neural_check:
        parts.append(f'神经覆盖 ≥ {c.min_frame_fraction:.0%} · 空档 ≤ {c.max_gap_seconds * 1000:.0f} ms')
    if c.direction_check:
        parts.append('方向标签检查')
    parts.append('速度 bin：' + ('整 trial 排除' if c.velocity_bin_policy == 'drop_trial' else '只去掉坏 bin'))
    if c.exclude_flagged_acquired:
        parts.append('敏感性：排除命中早于窗口结束')
    return ' · '.join(parts)


class FilterDialog(tk.Toplevel):
    def __init__(self, app, criteria: TrialFilter, on_apply):
        super().__init__(app)
        self.app = app
        self.on_apply = on_apply
        self.title(f'Trial 筛选设置 · {CRITERIA_VERSION}')
        self.geometry(f'{px(app, 1180)}x{px(app, 760)}')
        self.transient(app)
        self.vars = {}
        body = ttk.Frame(self, padding=10)
        body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=7)
        body.columnconfigure(1, weight=4)
        body.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(body)
        notebook.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        for key, label in TABS:
            frame = ttk.Frame(notebook, padding=10)
            notebook.add(frame, text=f'  {label}  ')
            row = 0
            for name, text, kind, group, help_text in FIELD_SPEC:
                if group != key:
                    continue
                value = getattr(criteria, name)
                if kind == 'bool':
                    var = tk.BooleanVar(value=bool(value))
                    ttk.Checkbutton(frame, text=text, variable=var).grid(row=row, column=0, columnspan=2,
                                                                       sticky='w', pady=3)
                else:
                    ttk.Label(frame, text=text).grid(row=row, column=0, sticky='w', pady=3)
                    var = tk.StringVar(value='' if value is None else str(value))
                    if kind.startswith('choice:'):
                        widget = ttk.Combobox(frame, textvariable=var, values=kind.split(':', 1)[1].split(','),
                                              state='readonly', width=18)
                    else:
                        widget = ttk.Entry(frame, textvariable=var, width=12)
                    widget.grid(row=row, column=1, sticky='w', padx=8, pady=3)
                self.vars[name] = (var, kind)
                if help_text:
                    row += 1
                    ttk.Label(frame, text=help_text, style='Muted.TLabel', wraplength=px(app, 620), justify='left'
                              ).grid(row=row, column=0, columnspan=2, sticky='w', padx=(22, 0))
                row += 1
        rules = ttk.Frame(notebook, padding=12)
        notebook.add(rules, text='  规范  ')
        for i, text in enumerate(RULES):
            ttk.Label(rules, text=f'{i + 1}. {text}', wraplength=px(app, 640), justify='left').pack(anchor='w', pady=4)

        side = ttk.LabelFrame(body, text=' 预览（当前选中的记录） ', padding=8)
        side.grid(row=0, column=1, sticky='nsew')
        ttk.Button(side, text='预览', command=self.preview).pack(anchor='w')
        self.preview_text = tk.Text(side, wrap='word', font=(FONT, 9), relief='flat', height=30, width=40)
        self.preview_text.pack(fill='both', expand=True, pady=(6, 0))
        self.preview_text.insert('1.0', '点击「预览」查看当前设置下被排除 / 标记的 trial 数（按主要原因）。')

        buttons = ttk.Frame(self, padding=(10, 0, 10, 10))
        buttons.pack(fill='x')
        ttk.Button(buttons, text='恢复默认', command=lambda: self.load_values(TrialFilter())).pack(side='left')
        ttk.Button(buttons, text='载入配置…', command=self.load_file).pack(side='left', padx=6)
        ttk.Button(buttons, text='保存配置…', command=self.save_file).pack(side='left')
        ttk.Label(buttons, text='应用后：结果签名包含全部阈值，已存储的旧结果不会被误用',
                  foreground=MUTED).pack(side='left', padx=16)
        ttk.Button(buttons, text='取消', command=self.destroy).pack(side='right')
        ttk.Button(buttons, text='应用', style='Accent.TButton', command=self.apply).pack(side='right', padx=6)

    # ------------------------------------------------------------ values
    def collect(self) -> TrialFilter:
        values = {}
        for name, (var, kind) in self.vars.items():
            raw = var.get()
            if kind == 'bool':
                values[name] = bool(raw)
            elif kind == 'int':
                values[name] = int(float(raw))
            elif kind == 'optint':
                values[name] = int(float(raw)) if str(raw).strip() else None
            elif kind == 'float':
                values[name] = float(raw)
            else:
                values[name] = raw
        return TrialFilter(**values)

    def load_values(self, criteria: TrialFilter):
        for name, (var, kind) in self.vars.items():
            value = getattr(criteria, name)
            var.set(bool(value) if kind == 'bool' else '' if value is None else str(value))

    def apply(self):
        try:
            criteria = self.collect()
        except (TypeError, ValueError) as exc:
            messagebox.showerror('参数无效', str(exc), parent=self)
            return
        self.on_apply(criteria)
        self.destroy()

    def load_file(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[('JSON', '*.json')])
        if path:
            try:
                self.load_values(TrialFilter.load(path))
            except (OSError, ValueError, TypeError) as exc:
                messagebox.showerror('无法载入', str(exc), parent=self)

    def save_file(self):
        try:
            criteria = self.collect()
        except (TypeError, ValueError) as exc:
            messagebox.showerror('参数无效', str(exc), parent=self)
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension='.json',
                                            initialfile=f'trial_filter_{CRITERIA_VERSION}.json',
                                            filetypes=[('JSON', '*.json')])
        if path:
            criteria.save(path)

    # ------------------------------------------------------------ preview
    def preview(self):
        try:
            criteria = self.collect()
        except (TypeError, ValueError) as exc:
            messagebox.showerror('参数无效', str(exc), parent=self)
            return
        recordings = self.app.view.selected_recordings()
        self.preview_text.delete('1.0', 'end')
        self.preview_text.insert('1.0', '计算中…' if recordings else '没有选中的记录')

        def work():
            lines = []
            for r in recordings:
                try:
                    cohort = apply_filter(get_session(r), criteria)
                    s, st = cohort.summary(), cohort.session_status()
                except Exception as exc:  # session-level errors are shown, not raised
                    lines.append(f'■ {r.label}\n  无法加载：{exc}\n')
                    continue
                lines.append(f'■ {r.label}  [{st["session_type"].split("（")[0]}] '
                             f'{"合格" if st["qualified"] else "不合格：" + "；".join(st["issues"])}')
                lines.append(f'  trial 行 {s["total_trial_rows"]} → 方向队列 {s["included_trials"]} → 速度队列 '
                             f'{s["velocity_trials"]}')
                by_group = {}
                for reason, n in s['primary_reason_counts'].items():
                    by_group.setdefault(GROUP_OF.get(reason, 'incomplete'), []).append((reason, n))
                for group in PRIORITY:
                    for reason, n in by_group.get(group, []):
                        lines.append(f'  ✗ {n:4d}  {REASON_LABELS.get(reason, reason)}')
                vel_reasons = {}
                for reasons in cohort.velocity_audit['reasons'] if cohort.n else []:
                    for x in reasons:
                        vel_reasons[x] = vel_reasons.get(x, 0) + 1
                for reason, n in vel_reasons.items():
                    lines.append(f'  △ {n:4d}  {REASON_LABELS.get(reason, reason)}（trial 数，仅速度队列）')
                for reason, n in s['flag_counts'].items():
                    lines.append(f'  ⚑ {n:4d}  {REASON_LABELS.get(reason, reason)}')
                for o in st['outages']:
                    lines.append(f'  ⚠ 神经断流 {o["duration_s"]:.1f} s（{o["start_s"]:.0f}–{o["end_s"]:.0f} s，'
                                 f'影响 {o["trials_overlapping"]} 个 trial）')
                lines.append('')
            text = '\n'.join(lines)
            self.app.events.put(dict(event='_status', call=(self._show_preview, text)))
        threading.Thread(target=work, daemon=True).start()

    def _show_preview(self, text):
        if self.winfo_exists():
            self.preview_text.delete('1.0', 'end')
            self.preview_text.insert('1.0', text)
