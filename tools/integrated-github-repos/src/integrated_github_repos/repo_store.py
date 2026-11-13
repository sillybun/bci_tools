from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QStandardPaths


class RepoStore:
    def __init__(self, path: Path | None = None) -> None:
        base = Path(QStandardPaths.writableLocation(QStandardPaths.AppDataLocation))
        self.path = path or base / "repositories.json"

    def load(self) -> list[Path]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        paths = []
        for value in payload.get("repositories", []):
            path = Path(value).expanduser()
            if path.is_dir():
                paths.append(path.resolve())
        return paths

    def save(self, paths: list[Path]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        unique = list(dict.fromkeys(str(path.resolve()) for path in paths))
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps({"schema_version": 1, "repositories": unique}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
