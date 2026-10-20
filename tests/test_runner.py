import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from launcher.catalog import Tool
from launcher.runner import ToolRunner, tool_environment


class RunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tool_directory = self.root / "tool with spaces"
        self.tool_directory.mkdir()
        self.runner = ToolRunner(self.root / "logs")
        self.addCleanup(self.stop_children)

    def stop_children(self):
        for run in self.runner.runs.values():
            if run.running:
                run.process.terminate()
            run.process.wait(timeout=5)

    def tool(self, code):
        return Tool("demo", self.tool_directory, "Demo", "Test tool", "Test", (),
                    (sys.executable, "-c", code), self.tool_directory / "README.md")

    def test_launch_uses_tool_cwd_and_captures_output(self):
        tool = self.tool("import os,sys; print(os.getcwd()); print(sys.argv[1])")
        from dataclasses import replace
        tool = replace(tool, command=(*tool.command, 'literal ; $HOME "quoted"'))
        run = self.runner.start(tool)
        self.assertEqual(run.process.wait(timeout=5), 0)
        log = run.log_path.read_text(encoding="utf-8")
        self.assertIn(str(self.tool_directory.resolve()), log)
        self.assertIn('literal ; $HOME "quoted"', log)
        self.assertFalse(run.running)

    def test_duplicate_click_reuses_running_process_then_allows_relaunch(self):
        tool = self.tool("import time; time.sleep(30)")
        first = self.runner.start(tool)
        self.assertIs(self.runner.start(tool), first)
        first.process.terminate()
        first.process.wait(timeout=5)
        second = self.runner.start(self.tool("print('restarted')"))
        self.assertIsNot(first, second)
        self.assertEqual(second.process.wait(timeout=5), 0)

    def test_failure_keeps_stderr_and_exit_code(self):
        run = self.runner.start(self.tool("raise RuntimeError('example launch failure')"))
        self.assertNotEqual(run.process.wait(timeout=5), 0)
        self.assertIn("example launch failure", run.log_path.read_text())

    def test_missing_executable_reports_error_without_running_record(self):
        from dataclasses import replace
        tool = replace(self.tool(""), command=(str(self.root / "does-not-exist"),))
        with self.assertRaises(OSError):
            self.runner.start(tool)
        self.assertNotIn(tool.id, self.runner.runs)

    def test_environment_does_not_leak_launcher_venv_to_tools(self):
        prefix = self.root / "launcher-venv"
        values = {
            "VIRTUAL_ENV": str(prefix),
            "UV_PROJECT_ENVIRONMENT": str(prefix),
            "PYTHONHOME": "wrong-home",
            "PYTHONPATH": "wrong-path",
            "PATH": os.pathsep.join([str(prefix / "bin"), str(prefix / "Scripts"), "/usr/bin"]),
            "BCI_DEVICE": "keep-me",
        }
        with patch.dict(os.environ, values), patch("launcher.runner.sys.prefix", str(prefix)), patch(
            "launcher.runner.sys.base_prefix", str(self.root / "python")
        ):
            env = tool_environment()
        for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONHOME", "PYTHONPATH"):
            self.assertNotIn(key, env)
        self.assertEqual(env["PATH"], "/usr/bin")
        self.assertEqual(env["BCI_DEVICE"], "keep-me")

    def test_child_receives_clean_environment(self):
        with patch.dict(os.environ, {"VIRTUAL_ENV": "launcher", "UV_PROJECT_ENVIRONMENT": "launcher"}):
            run = self.runner.start(self.tool(
                "import os,json; print(json.dumps({key: os.getenv(key) for key in "
                "['VIRTUAL_ENV', 'UV_PROJECT_ENVIRONMENT']}))"
            ))
        self.assertEqual(run.process.wait(timeout=5), 0)
        data = json.loads(run.log_path.read_text().splitlines()[-1])
        self.assertEqual(data, {"VIRTUAL_ENV": None, "UV_PROJECT_ENVIRONMENT": None})

    def test_tool_survives_launcher_process_exit(self):
        marker = self.root / "still-running.txt"
        child_code = (
            "from pathlib import Path; import time; time.sleep(0.3); "
            f"Path({str(marker)!r}).write_text('finished')"
        )
        supervisor = (
            "from pathlib import Path; import sys; "
            "from launcher.catalog import Tool; from launcher.runner import ToolRunner; "
            f"directory = Path({str(self.tool_directory)!r}); "
            f"tool = Tool('detached', directory, 'Test', 'Test', 'Test', (), "
            f"(sys.executable, '-c', {child_code!r}), directory / 'README.md'); "
            f"ToolRunner(Path({str(self.root / 'logs')!r})).start(tool)"
        )
        result = subprocess.run(
            [sys.executable, "-c", supervisor],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertTrue(marker.exists(), "Tool stopped when the launcher process exited")
        self.assertEqual(marker.read_text(), "finished")
