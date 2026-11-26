"""Locate archived logs by their original filename or run-prefixed name."""
from fnmatch import fnmatch
from pathlib import Path


def archived_logs(folder, pattern, *, recursive=False):
    folder = Path(folder)
    candidates = folder.rglob('*') if recursive else folder.glob('*')
    return sorted(p for p in candidates if p.is_file() and
                  fnmatch(p.name.split('--', 1)[-1] if p.name.startswith('run-') and '--' in p.name else p.name,
                          pattern))
