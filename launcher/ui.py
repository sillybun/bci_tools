"""A small searchable tool shelf, using only Python's bundled Tk."""

import math
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import messagebox, ttk

from .catalog import Tool, discover_tools
from .runner import ToolRunner


ROOT = Path(__file__).resolve().parents[1]
BG = "#f4f6fa"
INK = "#19283e"
MUTED = "#65758b"
BLUE = "#2563eb"


class Launcher(tk.Tk):
    def __init__(self, root_directory: Path = ROOT):
        super().__init__()
        self.root_directory = root_directory
        self.title("BCI Tools · 工具箱")
        self.geometry("980x660")
        self.minsize(760, 500)
        self.configure(bg=BG)
        self.runner = ToolRunner(root_directory / ".local" / "logs")
        self.tools: list[Tool] = []
        self.errors: list[str] = []
        self.category: str | None = None
        self.query = tk.StringVar()
        self.images: dict[str, tk.PhotoImage] = {}
        self.card_controls = {}
        self.columns = 0
        self._resize_job = None
        self._style()
        self._build()
        self.query.trace_add("write", lambda *_: self._render_cards())
        self.bind("<Escape>", lambda _event: self.query.set(""))
        self.bind("<Control-f>", self._focus_search)
        self.bind("<Command-f>", self._focus_search)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.refresh()
        self.after(400, self._poll)

    def _style(self):
        self.option_add("*Font", "TkDefaultFont")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TButton", padding=(12, 8), relief="flat", borderwidth=0,
                        background="#e9eef6", foreground=INK)
        style.map("TButton", background=[("active", "#dce6f5")],
                  foreground=[("disabled", "#8b98aa")])
        style.configure("Launch.TButton", background=BLUE, foreground="white")
        style.map("Launch.TButton", background=[("disabled", "#e6edf8"), ("active", "#1d4ed8")],
                  foreground=[("disabled", "#63748d")])
        style.configure("Search.TEntry", padding=10, fieldbackground="white",
                        bordercolor="#dce3ed", lightcolor="#dce3ed", darkcolor="#dce3ed")

    def _label(self, parent, text, *, size=12, color=INK, bold=False, bg=BG, **kwargs):
        return tk.Label(parent, text=text, bg=bg, fg=color,
                        font=("TkDefaultFont", size, "bold" if bold else "normal"), **kwargs)

    def _build(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill="x", padx=28, pady=(24, 18))
        self._label(header, "BCI Tools", size=26, bold=True).pack(anchor="w")
        self._label(header, "团队的 BCI 工具箱", color=MUTED).pack(anchor="w", pady=(4, 16))
        search_row = tk.Frame(header, bg=BG)
        search_row.pack(fill="x")
        self._label(search_row, "搜索工具", color=MUTED).pack(side="left", padx=(0, 12))
        self.search = ttk.Entry(search_row, textvariable=self.query, style="Search.TEntry")
        self.search.pack(side="left", fill="x", expand=True)
        ttk.Button(search_row, text="刷新", command=self.refresh).pack(side="left", padx=(12, 0))

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=28)
        self.sidebar = tk.Frame(body, bg=BG, width=164)
        self.sidebar.pack(side="left", fill="y", padx=(0, 22))
        self.sidebar.pack_propagate(False)
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)
        self.heading = self._label(right, "", size=16, bold=True)
        self.heading.pack(anchor="w", pady=(0, 14))
        shelf = tk.Frame(right, bg=BG)
        shelf.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(shelf, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(shelf, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.grid_frame = tk.Frame(self.canvas, bg=BG)
        self.grid_id = self.canvas.create_window((0, 0), window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", lambda _: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", self._resize)
        self.bind_all("<MouseWheel>", self._scroll)
        self.bind_all("<Button-4>", lambda _: self.canvas.yview_scroll(-1, "units"))
        self.bind_all("<Button-5>", lambda _: self.canvas.yview_scroll(1, "units"))

        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", padx=28, pady=16)
        self.status = self._label(footer, "", size=11, color=MUTED, anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        self.problems = ttk.Button(footer, text="查看配置问题", command=self._show_problems)

    def _focus_search(self, _event=None):
        self.search.focus_set()
        self.search.selection_range(0, "end")
        return "break"

    def _scroll(self, event):
        delta = -event.delta if sys.platform == "darwin" else -int(event.delta / 120)
        self.canvas.yview_scroll(int(delta), "units")

    def _resize(self, event):
        self.canvas.itemconfigure(self.grid_id, width=event.width)
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(100, self._layout)

    def _layout(self):
        self._resize_job = None
        columns = max(1, self.canvas.winfo_width() // 300)
        if columns != self.columns:
            self._render_cards()

    def refresh(self):
        self.tools, self.errors = discover_tools(self.root_directory / "tools")
        if self.category not in {tool.category for tool in self.tools}:
            self.category = None
        self.images.clear()
        self._render_categories()
        self._render_cards()
        self.status.configure(
            text=f"发现 {len(self.tools)} 个工具" + (f" · {len(self.errors)} 个配置需要检查" if self.errors else "")
        )
        if self.errors:
            self.problems.pack(side="right")
        else:
            self.problems.pack_forget()

    def _render_categories(self):
        for child in self.sidebar.winfo_children():
            child.destroy()
        self._label(self.sidebar, "分类", size=11, color=MUTED).pack(anchor="w", pady=(2, 12))
        categories = [None, *sorted({tool.category for tool in self.tools})]
        for category in categories:
            count = sum(category is None or tool.category == category for tool in self.tools)
            selected = self.category == category
            button = tk.Label(
                self.sidebar, text=f"{category or '全部工具'}   {count}",
                anchor="w", padx=12, pady=11, cursor="hand2",
                bg="#e5edff" if selected else BG, fg=BLUE if selected else MUTED,
                font=("TkDefaultFont", 12, "bold" if selected else "normal"),
            )
            button.pack(fill="x", pady=2)
            button.bind("<Button-1>", lambda _, value=category: self._select_category(value))

    def _select_category(self, category):
        self.category = category
        self._render_categories()
        self._render_cards()

    def _render_cards(self):
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.card_controls.clear()
        for column in range(self.columns):
            self.grid_frame.columnconfigure(column, weight=0, uniform="")
        self.columns = max(1, self.canvas.winfo_width() // 300)
        for column in range(self.columns):
            self.grid_frame.columnconfigure(column, weight=1, uniform="cards")
        visible = [tool for tool in self.tools if tool.matches(self.query.get(), self.category)]
        self.heading.configure(text=f"{self.category or '全部工具'}  ·  {len(visible)}")
        if not visible:
            message = "没有找到匹配的工具\n试试其他关键词或分类" if self.tools else "还没有工具\n添加工具目录和 tool.toml 后，点击刷新"
            self._label(self.grid_frame, message, color=MUTED, justify="left").grid(
                row=0, column=0, sticky="w", pady=32)
        for index, tool in enumerate(visible):
            self._card(tool, index)
        self.canvas.yview_moveto(0)

    def _card(self, tool, index):
        card = tk.Frame(self.grid_frame, bg="white", highlightthickness=1,
                        highlightbackground="#e0e6ef", padx=18, pady=18)
        card.grid(row=index // self.columns, column=index % self.columns,
                  sticky="nsew", padx=(0, 12), pady=(0, 12))
        if tool.icon and tool.id not in self.images:
            try:
                image = tk.PhotoImage(file=str(tool.icon))
                scale = max(1, math.ceil(max(image.width(), image.height()) / 54))
                self.images[tool.id] = image.subsample(scale)
            except tk.TclError:
                pass
        icon = self.images.get(tool.id)
        if icon:
            tk.Label(card, image=icon, bg="white").pack(anchor="w", pady=(0, 12))
        else:
            self._label(card, tool.name[:1], size=24, bold=True, color=BLUE,
                        bg="#edf3ff", padx=14, pady=7).pack(anchor="w", pady=(0, 12))
        title = self._label(card, tool.name, size=15, bold=True, bg="white", anchor="w",
                            justify="left", wraplength=240)
        title.pack(anchor="w", fill="x")
        description = self._label(card, tool.description, size=12, color=MUTED, bg="white",
                                  wraplength=240, justify="left", anchor="nw", height=3)
        description.pack(fill="x", pady=(8, 14))
        card.bind("<Configure>", lambda event: (
            title.configure(wraplength=max(180, event.width - 40)),
            description.configure(wraplength=max(180, event.width - 40))))
        self._label(card, tool.category, size=10, color=MUTED, bg="white").pack(anchor="w")
        actions = tk.Frame(card, bg="white")
        actions.pack(fill="x", pady=(14, 0))
        launch = ttk.Button(actions, text="打开工具", style="Launch.TButton",
                            command=lambda: self._launch(tool))
        launch.pack(side="left")
        ttk.Button(actions, text="使用说明", command=lambda: self._open_file(tool.readme)).pack(
            side="right")
        state_row = tk.Frame(card, bg="white")
        state_row.pack(fill="x", pady=(12, 0))
        state = self._label(state_row, "", size=10, color=MUTED, bg="white")
        state.pack(side="left")
        log = self._label(state_row, "查看日志", size=10, color=BLUE, bg="white", cursor="hand2")
        log.bind("<Button-1>", lambda _: self._open_log(tool))
        self.card_controls[tool.id] = (launch, state, log)
        self._update_card(tool.id)

    def _launch(self, tool):
        try:
            self.runner.start(tool)
        except OSError as exc:
            self.status.configure(text=f"{tool.name} 启动失败")
            messagebox.showerror("无法打开工具", f"{exc}\n\n请检查工具环境和启动命令。", parent=self)
        else:
            self.status.configure(text=f"已启动 {tool.name} · 首次运行可能需要准备环境")
        self._update_card(tool.id)

    def _update_card(self, tool_id):
        controls = self.card_controls.get(tool_id)
        if not controls:
            return
        launch, state, log = controls
        run = self.runner.runs.get(tool_id)
        if run is None:
            launch.configure(text="打开工具", state="normal")
            state.configure(text="就绪", fg=MUTED)
            return
        code = run.process.poll()
        launch.configure(text="运行中" if code is None else "打开工具",
                         state="disabled" if code is None else "normal")
        state.configure(text="已启动" if code is None else "已退出" if code == 0 else f"运行失败 · {code}",
                        fg="#14805e" if code is None else MUTED if code == 0 else "#c0393d")
        log.pack(side="right")

    def _poll(self):
        for tool_id in self.card_controls:
            self._update_card(tool_id)
        self.after(400, self._poll)

    def _open_file(self, path):
        try:
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("无法打开文件", f"{path}\n\n{exc}", parent=self)

    def _open_log(self, tool):
        run = self.runner.runs.get(tool.id)
        if not run:
            return
        window = tk.Toplevel(self)
        window.title(f"{tool.name} · 启动日志")
        window.geometry("760x440")
        text = tk.Text(window, wrap="word", padx=14, pady=14)
        scrollbar = ttk.Scrollbar(window, command=text.yview)
        scrollbar.pack(side="right", fill="y")
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(fill="both", expand=True)

        def reload():
            try:
                with run.log_path.open("rb") as handle:
                    handle.seek(0, 2)
                    handle.seek(max(0, handle.tell() - 64_000))
                    content = handle.read().decode("utf-8", errors="replace")
            except OSError as exc:
                content = str(exc)
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.insert("1.0", content)
            text.configure(state="disabled")
            text.see("end")

        ttk.Button(window, text="刷新日志", command=reload).pack(pady=10)
        reload()

    def _show_problems(self):
        messagebox.showwarning("工具配置需要检查", "\n\n".join(self.errors), parent=self)
