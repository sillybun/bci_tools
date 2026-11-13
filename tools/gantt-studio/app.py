"""Run the existing Electron application from an isolated uv project."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def check_runtime() -> str:
    node = shutil.which("node")
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    if not node or not npm:
        raise RuntimeError("需要 Node.js >=22.12 和 npm；安装后重新运行此命令。")
    version = subprocess.check_output([node, "--version"], text=True).strip()
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (22, 12, 0):
        raise RuntimeError(f"需要 Node.js >=22.12，当前为 {version}")
    return npm


def prepare(npm: str) -> None:
    fingerprint = hashlib.sha256((ROOT / "package-lock.json").read_bytes()).hexdigest()
    stamp = ROOT / "node_modules" / ".uv-launch-lock"
    installed = ROOT / "node_modules" / ".package-lock.json"
    if not installed.exists() or not stamp.exists() or stamp.read_text() != fingerprint:
        print("首次启动或依赖锁更新：使用 npm ci 准备 Electron 环境…", flush=True)
        # Keep the tool cache writable and independent of a system npm cache.
        env = os.environ.copy()
        env.setdefault("npm_config_cache", str(ROOT / ".cache" / "npm"))
        subprocess.run([npm, "ci"], cwd=ROOT, check=True, env=env)
        stamp.write_text(fingerprint)


def main() -> int:
    parser = argparse.ArgumentParser(description="Start Gantt Studio through uv")
    parser.add_argument("--check", action="store_true", help="check Node/npm without installing or launching")
    parser.add_argument("--setup", action="store_true", help="install locked npm dependencies only")
    args = parser.parse_args()
    try:
        npm = check_runtime()
        if args.check:
            print("Node.js / npm ready")
            return 0
        prepare(npm)
        if args.setup:
            return 0
        return subprocess.run([npm, "run", "desktop"], cwd=ROOT).returncode
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Gantt Studio 启动失败：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
