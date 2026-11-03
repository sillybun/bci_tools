"""Versioned section registry: discovers `<id>_<name>_v<N>/section.py` folders."""
from __future__ import annotations

import importlib
import pkgutil
import re
import traceback
from pathlib import Path

from .base import Section

FOLDER = re.compile(r'^(?P<id>[sx]\d{2})_(?P<name>[a-z0-9_]+?)_v(?P<version>\d+)$')
_REGISTRY: dict[str, dict[int, Section]] | None = None
LOAD_ERRORS: dict[str, str] = {}


def _discover() -> dict[str, dict[int, Section]]:
    registry: dict[str, dict[int, Section]] = {}
    here = Path(__file__).parent
    for info in sorted(pkgutil.iter_modules([str(here)]), key=lambda m: m.name):
        match = FOLDER.match(info.name)
        if not match or not info.ispkg:
            continue
        try:
            module = importlib.import_module(f'{__name__}.{info.name}.section')
            section: Section = module.SECTION
        except Exception:  # keep the GUI usable if one section is broken
            LOAD_ERRORS[info.name] = traceback.format_exc()
            continue
        section.version = f"v{match['version']}"
        section.id = section.id or match['id']
        registry.setdefault(match['id'], {})[int(match['version'])] = section
    return registry


def registry(reload: bool = False) -> dict[str, dict[int, Section]]:
    global _REGISTRY
    if _REGISTRY is None or reload:
        LOAD_ERRORS.clear()
        _REGISTRY = _discover()
    return _REGISTRY


def sections(mode: str) -> list[Section]:
    """Latest version of every section for the given mode, ordered by id."""
    prefix = 's' if mode == 'single' else 'x'
    out = []
    for key in sorted(registry()):
        if key.startswith(prefix):
            versions = registry()[key]
            out.append(versions[max(versions)])
    return out


def versions(section_id: str) -> list[Section]:
    versions = registry().get(section_id, {})
    return [versions[v] for v in sorted(versions)]
