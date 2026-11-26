"""Review and apply a sibling-folder rename from the GUI."""
from pathlib import Path
import re
import tkinter as tk
from tkinter import messagebox, ttk

from ..core.run_config import suggest_run_name


def rename_target(source, name):
    name = name.strip()
    if (not name or name in ('.', '..') or name.endswith('.') or
            re.search(r'[<>:"/\\|?*\x00-\x1f]', name) or
            re.fullmatch(r'(?:CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?', name, re.I)):
        raise ValueError('请输入有效的文件夹名称。')
    if name == source.name:
        raise ValueError('新名称与原名称相同。')
    target = source.with_name(name)
    if target.is_symlink() or (target.exists() and not
            (name.casefold() == source.name.casefold() and target.samefile(source))):
        raise ValueError('同一目录中已存在这个名称。')
    return target


class RenameFolderDialog(tk.Toplevel):
    def __init__(self, app, source):
        super().__init__(app)
        self.source = source
        self.app = app
        self.title('重命名文件夹')
        self.transient(app)
        self.minsize(850, 210)
        body = ttk.Frame(self, padding=14)
        body.pack(fill='both', expand=True)
        body.columnconfigure((0, 1), weight=1)
        ttk.Label(body, text=str(source.parent)).grid(row=0, column=0, columnspan=2, sticky='w')
        ttk.Label(body, text='更名前').grid(row=1, column=0, sticky='w', pady=(8, 2))
        ttk.Label(body, text='更名后（可编辑）').grid(row=1, column=1, sticky='w', pady=(8, 2))
        self.original = tk.StringVar(value=source.name)
        self.new_name = tk.StringVar(value=suggest_run_name(source))
        self.note = tk.StringVar()
        self.preview = tk.StringVar()
        ttk.Entry(body, textvariable=self.original, state='readonly').grid(row=2, column=0, sticky='ew', padx=(0, 8))
        editor = ttk.Entry(body, textvariable=self.new_name)
        editor.grid(row=2, column=1, sticky='ew')
        ttk.Label(body, text='追加备注（可选）').grid(row=3, column=0, sticky='w', pady=(8, 2))
        ttk.Entry(body, textvariable=self.note).grid(row=4, column=0, columnspan=2, sticky='ew')
        ttk.Label(body, textvariable=self.preview, wraplength=810).grid(
            row=5, column=0, columnspan=2, sticky='w', pady=8)
        buttons = ttk.Frame(body)
        buttons.grid(row=6, column=0, columnspan=2, sticky='e')
        ttk.Button(buttons, text='取消', command=self.destroy).pack(side='left', padx=(0, 8))
        ttk.Button(buttons, text='确认重命名', style='Accent.TButton', command=self.apply).pack(side='left')
        for variable in (self.new_name, self.note):
            variable.trace_add('write', self.update_preview)
        self.update_preview()
        self.bind('<Escape>', lambda _: self.destroy())
        self.grab_set()
        editor.focus_set()

    def final_name(self):
        name, note = self.new_name.get().strip(), self.note.get().strip()
        return f'{name}-{note}' if name and note else name

    def update_preview(self, *_):
        self.preview.set('最终名称：' + self.final_name())

    def apply(self):
        if self.app.task_queue.current or self.app.task_queue.waiting:
            messagebox.showwarning('有分析任务', '请在任务队列结束后重命名。', parent=self)
            return
        try:
            target = rename_target(self.source, self.final_name())
            self.source.rename(target)
        except (OSError, ValueError) as exc:
            messagebox.showerror('重命名失败', str(exc), parent=self)
            return
        self.destroy()
        self.app.refresh_recordings()
        self.app.status.set(f'文件夹已重命名：{target.name}')
