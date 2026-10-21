"""Run explicitly on a desktop: uv run --locked python tests/ui_smoke.py."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import time
import tkinter as tk
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from launcher.ui import CARD_GAP, CARD_SIZE, Launcher, ROOT


def settle(window, duration=0.15):
    """Let resize debouncing, Tk geometry and process polling actually run."""
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        window.update()
        time.sleep(0.01)


def check_responsive_shelf(window, tool):
    # Multiple rows expose clipping and scrollbar regressions hidden by one tool.
    window.tools = [replace(tool, id=f"layout-{index}", name=f"工具 {index}")
                    for index in range(12)]
    window._render_cards()
    for geometry, columns in [("640x440", 1), ("760x500", 2), ("1280x640", 4)]:
        window.geometry(geometry)
        settle(window)
        assert window.columns == columns, (geometry, window.columns)
        assert window.sidebar.winfo_width() == 176
        assert window.scrollbar.winfo_ismapped()
        cards = list(window.cards.values())
        for card in cards:
            assert (card.winfo_width(), card.winfo_height()) == (CARD_SIZE, CARD_SIZE)
            assert card.winfo_rootx() + CARD_SIZE <= (
                window.canvas.winfo_rootx() + window.canvas.winfo_width())
        if columns > 1:
            assert cards[1].winfo_x() - cards[0].winfo_x() == CARD_SIZE + CARD_GAP
        assert cards[columns].winfo_y() - cards[0].winfo_y() == CARD_SIZE + CARD_GAP
    window.canvas.yview_moveto(1)
    window.query.set("工具 11")
    settle(window)
    assert len(window.cards) == 1
    assert not window.scrollbar.winfo_ismapped()
    assert window.canvas.yview()[0] == 0
    window.query.set("")
    window.tools = [tool]
    window._render_cards()
    window.geometry("980x640")
    settle(window)


def check_more_menu(window, tool, *, has_log):
    # Native popups are exercised manually; here invoke the actual menu commands
    # without leaving a popup waiting for a human during the smoke run.
    with patch.object(tk.Menu, "tk_popup"):
        window.card_controls[tool.id][2].invoke()
    menu = window._menu
    assert [menu.entrycget(index, "label") for index in (0, 1, 3)] == [
        "工具说明", "使用帮助", "查看日志"]
    assert menu.entrycget(3, "state") == ("normal" if has_log else "disabled")
    menu.invoke(0)
    window.update()
    details = next(child for child in window.winfo_children() if isinstance(child, tk.Toplevel))
    assert any(isinstance(child, tk.Label) and child.cget("text") == tool.description
               for child in details.winfo_children())
    details.destroy()
    with patch.object(window, "_open_file") as open_file:
        menu.invoke(1)
        open_file.assert_called_once_with(tool.readme)
    if has_log:
        menu.invoke(3)
        window.update()
        log_window = next(child for child in window.winfo_children() if isinstance(child, tk.Toplevel))
        viewer = next(child for child in log_window.winfo_children() if isinstance(child, tk.Frame))
        log_text = next(child for child in viewer.winfo_children() if isinstance(child, tk.Text))
        assert "smoke failure" in log_text.get("1.0", "end")
        log_window.destroy()


def main():
    with tempfile.TemporaryDirectory(prefix="bci-launcher-smoke-") as directory:
        temporary = Path(directory)
        window = Launcher()
        window.runner.log_directory = temporary / "logs"
        try:
            window.update()
            tool = next(tool for tool in window.tools if tool.id == "integrated-term")
            check_responsive_shelf(window, tool)
            window.tools = [tool]
            window._render_categories()
            window._render_cards()
            window.query.set("不存在的工具")
            window.update()
            assert not window.card_controls
            window.query.set("lsl")
            window.update()
            assert list(window.card_controls) == [tool.id]
            window.query.set("")
            window._select_category(tool.category)
            window.update()
            assert list(window.card_controls) == [tool.id]
            window.geometry("760x500")
            window.update()
            window._layout()
            window.update()
            launch, state, _ = window.card_controls[tool.id]
            assert launch.winfo_viewable()
            assert launch.winfo_rootx() >= window.winfo_rootx()
            assert launch.winfo_rootx() + launch.winfo_width() <= window.winfo_rootx() + window.winfo_width()
            check_more_menu(window, tool, has_log=False)

            # Run the real tool in its uv environment, with disposable user state.
            child_code = (
                "from pathlib import Path; import sys; import app; import lsl_registry; "
                f"app.CONFIG_PATH = Path({str(temporary / 'processes.json')!r}); "
                f"lsl_registry.registry_path = lambda: Path({str(temporary / 'registry.json')!r}); "
                "window = app.App(); "
                "print('TOOL_ENV=' + sys.prefix, flush=True); "
                "print('TOOL_UI=' + window.title(), flush=True); "
                "window.after(1500, window._close); window.mainloop()"
            )
            test_tool = replace(tool, command=("uv", "run", "--locked", "python", "-c", child_code))
            window.tools = [test_tool]
            window._render_cards()
            window.card_controls[tool.id][0].invoke()
            run = window.runner.runs[tool.id]
            window._update_card(tool.id)
            assert "disabled" in window.card_controls[tool.id][0].state()
            assert window.card_controls[tool.id][1].cget("text") == "运行中"
            assert window.runner.start(test_tool) is run
            # Filtering and refresh must not lose the actual process state.
            window.query.set("不存在")
            window.query.set("")
            window.refresh()
            assert "disabled" in window.card_controls[tool.id][0].state()
            deadline = time.monotonic() + 15
            while run.running and time.monotonic() < deadline:
                window.update()
                time.sleep(0.03)
            assert run.process.poll() == 0, run.log_path.read_text()
            settle(window, 0.5)
            assert window.card_controls[tool.id][1].cget("text") == "已退出"
            log = run.log_path.read_text()
            assert f"TOOL_ENV={ROOT / 'tools/integrated-term/.venv'}" in log, log
            assert "TOOL_UI=集成终端控制台" in log, log
            assert (temporary / "processes.json").is_file()

            # A failed tool remains actionable and its error is visible in the log viewer.
            failed_tool = replace(tool, command=(sys.executable, "-c", "raise RuntimeError('smoke failure')"))
            window.tools = [failed_tool]
            window._render_cards()
            window.card_controls[tool.id][0].invoke()
            failed_run = window.runner.runs[tool.id]
            failed_run.process.wait(timeout=5)
            window._update_card(tool.id)
            assert "运行失败" in window.card_controls[tool.id][1].cget("text")
            assert "disabled" not in window.card_controls[tool.id][0].state()
            check_more_menu(window, failed_tool, has_log=True)
            assert "smoke failure" in failed_run.log_path.read_text()

            missing_tool = replace(tool, command=(str(temporary / "missing-executable"),))
            window.tools = [missing_tool]
            window._render_cards()
            with patch("launcher.ui.messagebox.showerror") as showerror:
                window.card_controls[tool.id][0].invoke()
                showerror.assert_called_once()
            assert window.card_controls[tool.id][1].cget("text") == "启动失败"
            assert "disabled" not in window.card_controls[tool.id][0].state()

            window.refresh()
            assert next(item for item in window.tools if item.id == tool.id).command == tool.command
            print("Native UI smoke passed: discovery, search, category, 1/2/4-column layout,")
            print("fixed square cards, overflow-only scrollbar, description/help/log menus,")
            print("real tool launch, independent uv environment, duplicate prevention, polling,")
            print("state across filtering/refresh, normal exit, failure/retry states, and log viewer.")
        finally:
            for run in window.runner.runs.values():
                if run.running:
                    run.process.terminate()
                run.process.wait(timeout=5)
            window.destroy()


if __name__ == "__main__":
    main()
