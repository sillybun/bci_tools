"""A quiet, searchable macOS-style tool shelf, using Python's bundled Tk."""

import math
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from tkinter import font, messagebox, ttk

from .appearance import (
    BG, SIDEBAR, TOOLBAR, INK, MUTED, BLUE, BORDER, SELECTED,
    rounded_rectangle, surface_image, symbol_image,
)
from .catalog import Tool, discover_tools
from .runner import ToolRunner


ROOT = Path(__file__).resolve().parents[1]
CARD_SIZE = 230
CARD_GAP = 20
SHELF_PADDING = 26


class Launcher(tk.Tk):
    def __init__(self, root_directory: Path = ROOT):
        super().__init__()
        self.root_directory = root_directory
        self.title("BCI Tools")
        self.geometry("980x640")
        self.minsize(640, 440)
        self.configure(bg=BG)
        self.runner = ToolRunner(root_directory / ".local" / "logs")
        self.tools: list[Tool] = []
        self.errors: list[str] = []
        self.category: str | None = None
        self.query = tk.StringVar()
        self.images: dict[str, tk.PhotoImage] = {}
        self.card_controls = {}
        self.cards = {}
        self.category_buttons = {}
        self.failed_launches = {}
        self.columns = 0
        self._resize_job = None
        self._poll_job = None
        self._menu = None
        self._style()
        self._build()
        self.query.trace_add("write", self._query_changed)
        self.bind("<Escape>", self._clear_search)
        self.bind("<Control-f>", self._focus_search)
        self.bind("<Command-f>", self._focus_search)
        self.bind("<MouseWheel>", self._scroll)
        self.bind("<Button-4>", self._scroll)
        self.bind("<Button-5>", self._scroll)
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.refresh()
        self._poll_job = self.after(400, self._poll)

    def _style(self):
        self.font_family = font.nametofont("TkDefaultFont").actual("family")
        self.option_add("*Font", "TkDefaultFont")
        style = ttk.Style(self)
        style.theme_use("clam")
        # Nine-slice surfaces keep ttk keyboard, focus and disabled semantics.
        self._theme_images = []
        surfaces = {
            "Primary": (BLUE, "#006be0", "#005ec4", "#e6edf7", BLUE),
            "Quiet": (TOOLBAR, "#e9edf3", "#dde3ed", TOOLBAR, "#e1e9f6"),
            "More": ("#ffffff", "#edf1f6", "#e3e9f2", "#ffffff", "#e1e9f6"),
            "Category": (SIDEBAR, "#e4e9f0", "#dbe3ef", SIDEBAR, "#e1e7f0"),
            "Selected": (SELECTED, "#d5e4fc", "#cadcf9", SELECTED, "#cfdef6"),
        }
        for name, colors in surfaces.items():
            images = [surface_image(self, color) for color in colors]
            self._theme_images.extend(images)
            element = f"{name}.surface"
            style.element_create(element, "image", images[0],
                                 ("disabled", images[3]), ("pressed", images[2]),
                                 ("active", images[1]), ("focus", images[4]),
                                 border=8, padding=0, sticky="nsew")
            style.layout(f"{name}.TButton", [(element, {"sticky": "nsew", "children": [
                ("Button.padding", {"sticky": "nsew", "children": [
                    ("Button.label", {"sticky": "nsew"})]})]})])
            style.configure(f"{name}.TButton", padding=(10, 6), borderwidth=0,
                            anchor="center", font=(self.font_family, 11),
                            foreground="white" if name == "Primary" else INK,
                            background=SIDEBAR if name in ("Category", "Selected")
                            else TOOLBAR if name == "Quiet" else "white")
            style.map(f"{name}.TButton", foreground=[("disabled", "#9ba6b5")])
        for name in ("Category", "Selected"):
            style.configure(f"{name}.TButton", anchor="w", padding=(10, 9),
                            foreground=BLUE if name == "Selected" else "#647185")
        style.configure("More.TButton", font=(self.font_family, 17), padding=0)
        style.configure("Search.TEntry", borderwidth=0, padding=0, relief="flat",
                        fieldbackground="white", background="white", foreground=INK, insertcolor=BLUE,
                        font=(self.font_family, 11))
        style.layout("Search.TEntry", [("Entry.textarea", {"sticky": "nsew"})])
        style.layout("Shelf.Vertical.TScrollbar", [("Vertical.Scrollbar.trough", {
            "sticky": "ns", "children": [("Vertical.Scrollbar.thumb", {"sticky": "nsew"})]})])
        style.configure("Shelf.Vertical.TScrollbar", width=7, arrowsize=0, borderwidth=0,
                        background="#c7cdd6", troughcolor=BG, bordercolor=BG,
                        lightcolor="#c7cdd6", darkcolor="#c7cdd6", gripcount=0)
        style.map("Shelf.Vertical.TScrollbar", background=[("active", "#aeb7c5")])
        self.symbols = {name: symbol_image(self, name, MUTED)
                        for name in ("grid", "terminal", "signal", "folder", "search", "refresh")}
        self.selected_symbols = {name: symbol_image(self, name, BLUE)
                                 for name in ("grid", "terminal", "signal", "folder")}

    def _label(self, parent, text, *, size=11, color=INK, bold=False, bg=BG, **kwargs):
        return tk.Label(parent, text=text, bg=bg, fg=color, borderwidth=0,
                        font=(self.font_family, size, "bold" if bold else "normal"), **kwargs)

    def _build(self):
        header = tk.Frame(self, bg=TOOLBAR, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)
        self._label(header, "BCI Tools", size=17, bold=True, bg=TOOLBAR).pack(
            side="left", padx=24)
        self.refresh_button = ttk.Button(header, image=self.symbols["refresh"],
                                         command=self.refresh, style="Quiet.TButton", takefocus=True)
        self.refresh_button.pack(side="right", padx=(10, 20), ipady=1)
        self.problems = ttk.Button(header, text="!", width=2, style="Quiet.TButton",
                                   command=self._show_problems)
        self.search_box = tk.Canvas(header, width=254, height=34, bg=TOOLBAR,
                                    highlightthickness=0, borderwidth=0)
        self.search_box.pack(side="right")
        self.search_outline = rounded_rectangle(self.search_box, 1, 1, 253, 33, radius=8,
                                                  fill="white", outline=BORDER, width=1)
        self.search_box.create_image(17, 17, image=self.symbols["search"])
        self.search = ttk.Entry(self.search_box, textvariable=self.query, style="Search.TEntry")
        self.search.place(x=34, y=7, width=208, height=20)
        self.placeholder = self._label(self.search_box, "搜索工具…", color=MUTED, bg="white")
        self.placeholder.place(x=34, y=7, height=20)
        self.placeholder.bind("<Button-1>", self._focus_search)
        self.search_box.bind("<Button-1>", self._focus_search)
        self.search.bind("<FocusIn>", lambda _: self._search_focus(True))
        self.search.bind("<FocusOut>", lambda _: self._search_focus(False))
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)
        self.sidebar = tk.Frame(body, bg=SIDEBAR, width=176)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")
        shelf = tk.Frame(body, bg=BG)
        shelf.pack(side="left", fill="both", expand=True)
        shelf.columnconfigure(0, weight=1)
        shelf.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(shelf, bg=BG, highlightthickness=0, borderwidth=0,
                                yscrollincrement=12)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar = ttk.Scrollbar(shelf, orient="vertical", command=self.canvas.yview,
                                       style="Shelf.Vertical.TScrollbar")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.grid_frame = tk.Frame(self.canvas, bg=BG)
        self.grid_id = self.canvas.create_window((SHELF_PADDING, SHELF_PADDING),
                                                 window=self.grid_frame, anchor="nw")
        self.grid_frame.bind("<Configure>", self._sync_scrollbar)
        self.canvas.bind("<Configure>", self._resize)

    def _search_focus(self, focused):
        self.search_box.itemconfigure(self.search_outline, outline=BLUE if focused else BORDER)
        self._sync_placeholder()

    def _sync_placeholder(self):
        if self.query.get() or self.focus_get() == self.search:
            self.placeholder.place_forget()
        else:
            self.placeholder.place(x=34, y=7, height=20)

    def _query_changed(self, *_):
        self._sync_placeholder()
        self._render_cards()

    def _focus_search(self, _event=None):
        self.search.focus_set()
        self.search.selection_range(0, "end")
        return "break"

    def _clear_search(self, _event=None):
        self.query.set("")
        return "break"

    def _scroll(self, event):
        if not self.scrollbar.winfo_ismapped():
            return
        x, y = event.x_root, event.y_root
        if not (self.canvas.winfo_rootx() <= x < self.canvas.winfo_rootx() + self.canvas.winfo_width()
                and self.canvas.winfo_rooty() <= y < self.canvas.winfo_rooty() + self.canvas.winfo_height()):
            return
        if event.num in (4, 5):
            delta = -1 if event.num == 4 else 1
        else:
            delta = -event.delta if sys.platform == "darwin" else -int(event.delta / 120)
        self.canvas.yview_scroll(int(delta), "units")
        return "break"

    def _resize(self, event):
        self.canvas.itemconfigure(self.grid_id, width=max(1, event.width - SHELF_PADDING * 2))
        if self._resize_job:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(60, self._layout)
        self._sync_scrollbar()

    def _column_count(self):
        available = self.canvas.winfo_width() - SHELF_PADDING * 2
        return max(1, (available + CARD_GAP) // (CARD_SIZE + CARD_GAP))

    def _layout(self):
        self._resize_job = None
        if self._column_count() != self.columns:
            self._render_cards(reset_scroll=False)
        self._sync_scrollbar()

    def _sync_scrollbar(self, _event=None):
        content_height = self.grid_frame.winfo_reqheight() + SHELF_PADDING * 2
        if self.cards:
            content_height -= CARD_GAP
        height = self.canvas.winfo_height()
        self.canvas.configure(scrollregion=(0, 0, self.canvas.winfo_width(), max(height, content_height)))
        if content_height > height and height > 1:
            self.scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 5), pady=8)
        else:
            self.scrollbar.grid_remove()
            self.canvas.yview_moveto(0)

    def refresh(self):
        self.tools, self.errors = discover_tools(self.root_directory / "tools")
        if self.category not in {tool.category for tool in self.tools}:
            self.category = None
        self.images.clear()
        self._render_categories()
        self._render_cards()
        if self.errors:
            self.problems.pack(side="right", padx=(8, 0), before=self.search_box)
        else:
            self.problems.pack_forget()

    @staticmethod
    def _category_symbol(category):
        if category is None:
            return "grid"
        if any(word in category for word in ("进程", "环境", "终端")):
            return "terminal"
        if any(word in category for word in ("采集", "信号", "连接")):
            return "signal"
        return "folder"

    def _render_categories(self):
        for child in self.sidebar.winfo_children():
            child.destroy()
        self.category_buttons.clear()
        self._label(self.sidebar, "分类", size=10, color=MUTED, bg=SIDEBAR).pack(
            anchor="w", padx=20, pady=(23, 12))
        for category in [None, *sorted({tool.category for tool in self.tools})]:
            selected = self.category == category
            symbol = self._category_symbol(category)
            button = ttk.Button(
                self.sidebar, text=f"  {category or '全部工具'}", compound="left",
                image=(self.selected_symbols if selected else self.symbols)[symbol],
                style="Selected.TButton" if selected else "Category.TButton",
                command=lambda value=category: self._select_category(value), takefocus=True,
            )
            button.pack(fill="x", padx=12, pady=3)
            self.category_buttons[category] = button

    def _select_category(self, category):
        self.category = category
        self._render_categories()
        self._render_cards()

    def _render_cards(self, *, reset_scroll=True):
        if self._menu:
            self._menu.destroy()
            self._menu = None
        for child in self.grid_frame.winfo_children():
            child.destroy()
        self.card_controls.clear()
        self.cards.clear()
        self.columns = self._column_count()
        visible = [tool for tool in self.tools if tool.matches(self.query.get(), self.category)]
        if not visible:
            message = "没有找到匹配的工具" if self.tools else "还没有工具"
            detail = "试试其他关键词或分类" if self.tools else "添加工具目录和 tool.toml 后刷新"
            empty = tk.Frame(self.grid_frame, bg=BG, height=190)
            empty.grid(row=0, column=0, sticky="ew")
            self._label(empty, message, size=13).pack(pady=(60, 8))
            self._label(empty, detail, color=MUTED).pack()
        for index, tool in enumerate(visible):
            self._card(tool, index)
        if reset_scroll:
            self.canvas.yview_moveto(0)

    def _card(self, tool, index):
        card = tk.Canvas(self.grid_frame, width=CARD_SIZE, height=CARD_SIZE, bg=BG,
                         highlightthickness=0, borderwidth=0)
        card.grid(row=index // self.columns, column=index % self.columns,
                  sticky="nw", padx=(0, CARD_GAP), pady=(0, CARD_GAP))
        rounded_rectangle(card, 2, 4, 228, 230, radius=13, fill="#e9ecf1", outline="")
        rounded_rectangle(card, 1, 1, 229, 228, radius=13, fill="white", outline=BORDER, width=1)
        # The built-in terminal uses the compact glyph from the launcher design.
        # Other tools keep their manifest icon, or a category glyph when absent.
        terminal = tool.id == "integrated-term"
        if tool.icon and not terminal and tool.id not in self.images:
            try:
                icon = tk.PhotoImage(file=str(tool.icon))
                scale = max(1, math.ceil(max(icon.width(), icon.height()) / 78))
                self.images[tool.id] = icon.subsample(scale)
            except tk.TclError:
                pass
        icon = self.images.get(tool.id)
        if terminal:
            rounded_rectangle(card, 83, 55, 147, 119, radius=13,
                              fill="#303b48", outline="#293440", width=1)
            card.create_line(99, 75, 109, 85, 99, 95, fill="white", width=3.5,
                             capstyle="round", joinstyle="round")
            card.create_line(118, 96, 132, 96, fill="white", width=3.5, capstyle="round")
        elif icon:
            card.create_image(115, 87, image=icon)
        else:
            rounded_rectangle(card, 84, 56, 146, 118, radius=15,
                              fill="#edf4ff", outline="#e5edfb")
            symbol = self._category_symbol(tool.category)
            key = f"fallback:{symbol}"
            if key not in self.images:
                self.images[key] = symbol_image(self, symbol, BLUE, size=34)
            card.create_image(115, 87, image=self.images[key])
        self._label(card, tool.name, size=13, bold=True, bg="white", wraplength=196,
                    justify="center").place(x=17, y=132, width=196, height=42)
        more = ttk.Button(card, text="⋯", style="More.TButton", takefocus=True)
        more.configure(command=lambda: self._show_tool_menu(tool, more))
        more.place(x=188, y=11, width=30, height=28)
        launch = ttk.Button(card, text="打开", style="Primary.TButton",
                            command=lambda: self._launch(tool), takefocus=True)
        launch.place(x=146, y=185, width=66, height=30)
        state = self._label(card, "未运行", size=10, color=MUTED, bg="white", anchor="w")
        state.place(x=32, y=191, width=108, height=19)
        card.create_oval(19, 197, 26, 204, fill="#9aa4b3", outline="", tags="state_dot")
        self.cards[tool.id] = card
        self.card_controls[tool.id] = (launch, state, more)
        self._update_card(tool.id)

    def _launch(self, tool):
        try:
            self.runner.start(tool)
        except OSError as exc:
            self.failed_launches[tool.id] = str(exc)
            self._update_card(tool.id)
            messagebox.showerror("无法打开工具", f"{exc}\n\n请检查工具环境和启动命令。", parent=self)
        else:
            self.failed_launches.pop(tool.id, None)
            self._update_card(tool.id)

    def _update_card(self, tool_id):
        controls = self.card_controls.get(tool_id)
        if not controls:
            return
        launch, state, _more = controls
        run = self.runner.runs.get(tool_id)
        code = run.process.poll() if run else None
        running = run is not None and code is None
        launch.state(["disabled"] if running else ["!disabled"])
        if running:
            label, color, dot = "运行中", "#348365", "#36a376"
        elif tool_id in self.failed_launches:
            label, color, dot = "启动失败", "#bf5158", "#d5676c"
        elif run is None:
            label, color, dot = "未运行", MUTED, "#9aa4b3"
        elif code == 0:
            label, color, dot = "已退出", MUTED, "#9aa4b3"
        else:
            label, color, dot = f"运行失败 · {code}", "#bf5158", "#d5676c"
        state.configure(text=label, fg=color)
        self.cards[tool_id].itemconfigure("state_dot", fill=dot)

    def _poll(self):
        for tool_id in self.card_controls:
            self._update_card(tool_id)
        self._poll_job = self.after(400, self._poll)

    def _show_tool_menu(self, tool, button):
        if self._menu:
            self._menu.destroy()
        self._menu = tk.Menu(self, tearoff=False)
        self._menu.add_command(label="工具说明", command=lambda: self._show_details(tool))
        self._menu.add_command(label="使用帮助", command=lambda: self._open_file(tool.readme))
        self._menu.add_separator()
        self._menu.add_command(label="查看日志", command=lambda: self._open_log(tool),
                               state="normal" if self._log_path(tool).is_file() else "disabled")
        self._menu.tk_popup(button.winfo_rootx(), button.winfo_rooty() + button.winfo_height() + 3)

    def _show_details(self, tool):
        window = tk.Toplevel(self, bg="white")
        window.title(f"{tool.name} · 工具说明")
        window.geometry("400x240")
        window.resizable(False, False)
        window.transient(self)
        self._label(window, tool.name, size=16, bold=True, bg="white").pack(
            anchor="w", padx=24, pady=(25, 8))
        self._label(window, tool.category, size=10, color=MUTED, bg="white").pack(
            anchor="w", padx=24)
        self._label(window, tool.description, bg="white", wraplength=350, justify="left",
                    anchor="w").pack(fill="x", padx=24, pady=16)
        ttk.Button(window, text="完成", style="Primary.TButton", command=window.destroy).pack(
            side="bottom", anchor="e", padx=24, pady=16)
        window.bind("<Escape>", lambda _: window.destroy())
        return window

    def _open_file(self, path):
        try:
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("无法打开文件", f"{path}\n\n{exc}", parent=self)

    def _log_path(self, tool):
        run = self.runner.runs.get(tool.id)
        return run.log_path if run else self.runner.log_directory / f"{tool.id}.log"

    def _open_log(self, tool):
        log_path = self._log_path(tool)
        window = tk.Toplevel(self, bg=BG)
        window.title(f"{tool.name} · 启动日志")
        window.geometry("760x440")
        window.transient(self)
        viewer = tk.Frame(window, bg="white")
        viewer.pack(fill="both", expand=True, padx=16, pady=(16, 0))
        viewer.columnconfigure(0, weight=1)
        viewer.rowconfigure(0, weight=1)
        text = tk.Text(viewer, wrap="word", padx=14, pady=14, bg="white", fg=INK,
                       borderwidth=0, highlightthickness=1, highlightbackground=BORDER,
                       font="TkFixedFont")
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(viewer, command=text.yview, style="Shelf.Vertical.TScrollbar")

        def scrolled(first, last):
            scrollbar.set(first, last)
            if float(first) > 0 or float(last) < 1:
                scrollbar.grid(row=0, column=1, sticky="ns")
            else:
                scrollbar.grid_remove()

        text.configure(yscrollcommand=scrolled)

        def reload():
            try:
                with log_path.open("rb") as handle:
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

        ttk.Button(window, text="刷新日志", style="Quiet.TButton", command=reload).pack(
            anchor="e", padx=16, pady=12)
        window.bind("<Escape>", lambda _: window.destroy())
        reload()
        return window

    def _show_problems(self):
        messagebox.showwarning("工具配置需要检查", "\n\n".join(self.errors), parent=self)

    def destroy(self):
        for job in (self._resize_job, self._poll_job):
            if job:
                self.after_cancel(job)
        super().destroy()
