"""Create and verify the vendored bci-decoder-sdk snapshots (maintenance tool).

Each snapshot `sdk_<short sha>/` holds verbatim copies of `src/bci_decoder_api/`
and selected `src/decoders/<name>/` packages from one commit of
github.com/sillybun/bci-decoder-sdk. The only change to a file body is a
mechanical import rewrite so two `bci_decoder_api` versions can coexist:

    from bci_decoder_api ...   ->   from ...bci_decoder_api ...

applied only to lines that start (after indentation) with `from bci_decoder_api`
inside `decoders/<name>/` files (three dots = the snapshot package). Every file
gets a header comment naming repo, commit, original path and original git blob
SHA. `VENDOR.json` records the same plus the pip spec.

Re-create (reads the local checkout with `git show`; never touches its worktree):

    uv run --locked python -m data_analysis_gui.decoders.vendor.vendor_tool E:/AitBCI/bci-decoder-sdk
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = 'github.com/sillybun/bci-decoder-sdk'
PIP_SPEC = 'bci-decoder-api @ git+ssh://git@github.com/sillybun/bci-decoder-sdk.git@{sha}'
HERE = Path(__file__).resolve().parent
END = '--- end vendor header ---'
SNAPSHOTS = {
    'b70297c092157a0627fa8d5f97f1694d6043e2bc': ('src/bci_decoder_api', 'src/decoders/__init__.py',
                                                 'src/decoders/standalone_refit_kf',
                                                 'src/decoders/madduri_coadaptive_decoder'),
    '8d79df0475b85ca859a02a8268f5262358a22afa': ('src/bci_decoder_api', 'src/decoders/__init__.py',
                                                 'src/decoders/ecog_linear_velocity_decoder'),
}
IMPORT_RE = re.compile(rb'^(\s*)from bci_decoder_api(\b)')
VENDORED_RE = re.compile(rb'^(\s*)from \.\.\.bci_decoder_api(\b)')
REWRITES = [{'applies_to': 'decoders/**/*.py',
             'pattern': r'^(\s*)from bci_decoder_api\b',
             'replacement': r'\1from ...bci_decoder_api'}]


def snapshot_dir(sha: str) -> Path:
    return HERE / f'sdk_{sha[:7]}'


def blob_sha(data: bytes) -> str:
    return hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest()


def _eol(data: bytes) -> bytes:
    return b'\r\n' if b'\r\n' in data else b'\n'


def rewrite(relative: str, data: bytes) -> bytes:
    """Original file bytes -> vendored body (import lines only)."""
    if not (relative.startswith('decoders/') and relative.endswith('.py')):
        return data
    lines = data.split(b'\n')
    return b'\n'.join(IMPORT_RE.sub(rb'\1from ...bci_decoder_api\2', line) for line in lines)


def unrewrite(relative: str, data: bytes) -> bytes:
    """Vendored body -> original bytes (inverse of `rewrite`)."""
    if not (relative.startswith('decoders/') and relative.endswith('.py')):
        return data
    lines = data.split(b'\n')
    return b'\n'.join(VENDORED_RE.sub(rb'\1from bci_decoder_api\2', line) for line in lines)


def is_import_line(line: bytes) -> bool:
    return bool(re.match(rb'^\s*(from|import)\s', line))


def header(sha: str, original: str, blob: str, relative: str, eol: bytes) -> bytes:
    lines = [f'Vendored from {REPO}',
             f'commit: {sha}',
             f'original path: {original}',
             f'original git blob: {blob}',
             'Verbatim copy; only import lines were rewritten (`from bci_decoder_api` -> '
             '`from ...bci_decoder_api`) so two SDK versions can coexist. See VENDOR.json.',
             END]
    if relative.endswith('.md'):
        text = ['<!--'] + lines + ['-->']
    else:
        text = ['# ' + line for line in lines]
    return eol.join(t.encode('utf-8') for t in text) + eol


def split_header(data: bytes) -> tuple[bytes, bytes]:
    """Return (header, body) of a vendored file."""
    marker = END.encode()
    i = data.find(marker)
    if i < 0:
        raise ValueError('missing vendor header')
    j = data.find(b'\n', i)
    if data[:i].lstrip().startswith(b'<!--'):
        j = data.find(b'\n', j + 1)              # also consume the closing -->
    return data[:j + 1], data[j + 1:]


def body_of(path: Path) -> bytes:
    return split_header(path.read_bytes())[1]


def git(repo: str, *args) -> bytes:
    return subprocess.run(['git', '-C', repo, *args], check=True, capture_output=True).stdout


def list_files(repo: str, sha: str, roots) -> list[str]:
    out = []
    for root in roots:
        names = git(repo, 'ls-tree', '-r', '--name-only', sha, root).decode().split()
        out += [n for n in names if '__pycache__' not in n]
    return sorted(set(out))


def vendor(repo: str):
    for sha, roots in SNAPSHOTS.items():
        dest = snapshot_dir(sha)
        records = []
        for original in list_files(repo, sha, roots):
            data = git(repo, 'show', f'{sha}:{original}')
            blob = git(repo, 'rev-parse', f'{sha}:{original}').decode().strip()
            assert blob == blob_sha(data), original
            relative = original.removeprefix('src/')
            body = rewrite(relative, data)
            assert unrewrite(relative, body) == data, original
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            eol = _eol(data)
            target.write_bytes(header(sha, original, blob, relative, eol) + body)
            changed = sum(a != b for a, b in zip(data.split(b'\n'), body.split(b'\n')))
            records.append(dict(path=relative, original_path=original, original_blob_sha=blob,
                                eol='crlf' if eol == b'\r\n' else 'lf', import_lines_rewritten=changed))
        (dest / '__init__.py').write_text(
            f'"""Vendored snapshot of {REPO} @ {sha} (see VENDOR.json). Not part of the original repo."""\n',
            encoding='utf-8')
        meta = dict(repo=f'https://{REPO}', commit=sha, pip_spec=PIP_SPEC.format(sha=sha),
                    license='none found in the repository at this commit',
                    import_rewrites=REWRITES,
                    note='Bodies are byte-identical to the originals apart from rewritten import lines; '
                         'each file starts with a header comment ending with "' + END + '". '
                         'sdk_*/__init__.py is local glue, not vendored.',
                    files=records)
        (dest / 'VENDOR.json').write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
        print(dest, len(records), 'files')


if __name__ == '__main__':
    vendor(sys.argv[1] if len(sys.argv) > 1 else 'E:/AitBCI/bci-decoder-sdk')
