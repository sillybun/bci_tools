"""Start tools without coupling their lifetime or environment to the launcher."""

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys

from .catalog import Tool


def tool_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT"):
        env.pop(key, None)
    if sys.prefix != sys.base_prefix:
        # Remove the launcher's venv from PATH; each tool chooses its own runtime.
        bin_dirs = {(Path(sys.prefix) / name).resolve() for name in ("bin", "Scripts")}
        env["PATH"] = os.pathsep.join(
            entry for entry in env.get("PATH", "").split(os.pathsep)
            if entry and Path(entry).resolve() not in bin_dirs
        )
    return env


@dataclass
class Run:
    process: subprocess.Popen
    log_path: Path

    @property
    def running(self) -> bool:
        return self.process.poll() is None


class ToolRunner:
    def __init__(self, log_directory: Path):
        self.log_directory = log_directory
        self.runs: dict[str, Run] = {}

    def start(self, tool: Tool) -> Run:
        existing = self.runs.get(tool.id)
        if existing and existing.running:
            return existing
        self.log_directory.mkdir(parents=True, exist_ok=True)
        log_path = self.log_directory / f"{tool.id}.log"
        options = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
            if os.name == "nt" else {"start_new_session": True}
        )
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"{tool.name}\n工作目录：{tool.directory}\n\n")
            log.flush()
            process = subprocess.Popen(
                list(tool.command),
                cwd=tool.directory,
                env=tool_environment(),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **options,
            )
        run = Run(process, log_path)
        self.runs[tool.id] = run
        return run
