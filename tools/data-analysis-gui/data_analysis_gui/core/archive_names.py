"""Short XDF archive names and local recording aliases."""
from functools import lru_cache
import json
import re


SHORT_XDF = re.compile(r'rec-\d{3,}(?:-(?:early|startup|aux))?-(?:spike|lfp)\.xdf', re.I)
ALIAS_FILE = '.xdf_name_map.json'


@lru_cache(maxsize=8)
def _aliases(path, mtime, size):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    return data.get('recordings', {})


def archived_recording_alias(path):
    if not SHORT_XDF.fullmatch(path.name):
        return {}
    for folder in path.parents:
        sidecar = folder / ALIAS_FILE
        try:
            stat = sidecar.stat()
        except OSError:
            continue
        try:
            return _aliases(sidecar, stat.st_mtime_ns, stat.st_size).get(
                path.relative_to(folder).as_posix(), {})
        except (OSError, ValueError, AttributeError):
            return {}
    return {}
