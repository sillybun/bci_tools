"""Run explicitly on a desktop: uv run --locked python tests/ui_smoke.py."""

from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from launcher.ui import Launcher, ROOT


def main():
    with tempfile.TemporaryDirectory(prefix="bci-launcher-smoke-") as directory:
        temporary = Path(directory)
        window = Launcher()
        window.runner.log_directory = temporary / "logs"
        try:
            window.update()
            tool = next(tool for tool in window.tools if tool.id == "integrated-term")
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
            deadline = time.monotonic() + 15
            while run.running and time.monotonic() < deadline:
                window.update()
                time.sleep(0.03)
            assert run.process.poll() == 0, run.log_path.read_text()
            window._update_card(tool.id)
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
            window._open_log(failed_tool)
            assert "smoke failure" in failed_run.log_path.read_text()
            window.refresh()
            assert next(item for item in window.tools if item.id == tool.id).command == tool.command
            print("Native UI smoke passed: discovery, search, category, narrow layout, real tool launch,")
            print("independent uv environment, normal exit, failure status, log viewer, and refresh.")
        finally:
            for run in window.runner.runs.values():
                if run.running:
                    run.process.terminate()
                run.process.wait(timeout=5)
            window.destroy()


if __name__ == "__main__":
    main()
