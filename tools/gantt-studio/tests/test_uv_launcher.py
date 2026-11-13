import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("gantt_launcher", Path(__file__).parents[1] / "app.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_rejects_missing_runtime(self):
        with patch.object(launcher.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Node.js"):
                launcher.check_runtime()

    def test_rejects_old_node(self):
        with patch.object(launcher.shutil, "which", return_value="runtime"), patch.object(
            launcher.subprocess, "check_output", return_value="v22.11.0\n"
        ):
            with self.assertRaisesRegex(RuntimeError, "22.12"):
                launcher.check_runtime()

    def test_installs_once_and_reinstalls_on_lock_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "package-lock.json"
            lock.write_text("one")
            def install(*args, **kwargs):
                (root / "node_modules").mkdir(exist_ok=True)
                (root / "node_modules/.package-lock.json").write_text("installed")
            with patch.object(launcher, "ROOT", root), patch.object(launcher.subprocess, "run", side_effect=install) as run:
                launcher.prepare("npm")
                launcher.prepare("npm")
                self.assertEqual(run.call_count, 1)
                lock.write_text("two")
                launcher.prepare("npm")
                self.assertEqual(run.call_count, 2)
                self.assertEqual(run.call_args.args[0], ["npm", "ci"])
                self.assertEqual(run.call_args.kwargs["cwd"], root)

    def test_failed_install_does_not_write_success_stamp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "package-lock.json").write_text("lock")
            with patch.object(launcher, "ROOT", root), patch.object(
                launcher.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "npm")
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    launcher.prepare("npm")
            self.assertFalse((root / "node_modules/.uv-launch-lock").exists())
