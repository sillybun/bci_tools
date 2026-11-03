"""Split recorded task sessions without resampling or merging duplicate streams.

XDF 1.0: https://github.com/sccn/xdf/wiki/Specifications
The original stream XML and value bytes are retained. Implicit timestamps are
expanded so that a cropped chunk never depends on a removed preceding sample.
Outputs are checked with pyxdf against the corresponding original samples.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

WIDTHS = {'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8, 'float32': 4, 'double64': 8}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _varint(value):
    width = 1 if value < 256 else 4 if value < 2**32 else 8
    return bytes([width]) + value.to_bytes(width, 'little')


def _take_varint(data, offset):
    width = data[offset]
    if width not in (1, 4, 8) or offset + 1 + width > len(data):
        raise ValueError('Invalid or truncated XDF variable integer')
    end = offset + 1 + width
    return int.from_bytes(data[offset + 1:end], 'little'), end


def _chunks(path):
    with Path(path).open('rb') as handle:
        if handle.read(4) != b'XDF:':
            raise ValueError('Invalid XDF signature')
        while width := handle.read(1):
            if width[0] not in (1, 4, 8):
                raise ValueError('Invalid XDF chunk length')
            raw_length = handle.read(width[0])
            if len(raw_length) != width[0]:
                raise ValueError('Truncated XDF chunk length')
            length = int.from_bytes(raw_length, 'little')
            data = handle.read(length)
            if length < 2 or len(data) != length:
                raise ValueError('Truncated XDF chunk')
            yield struct.unpack_from('<H', data)[0], data[2:]


def _write(handle, tag, payload):
    handle.write(_varint(len(payload) + 2) + struct.pack('<H', tag) + payload)


def _samples(payload, info, last):
    """Yield original timestamp and raw value bytes, including string lengths."""
    count, offset = _take_varint(payload, 4)
    nchan, fmt, period = info
    for _ in range(count):
        width = payload[offset]
        offset += 1
        if width == 8:
            stamp = struct.unpack_from('<d', payload, offset)[0]
            offset += 8
        elif width == 0:
            stamp = last + period
        else:
            raise ValueError('Unsupported XDF sample timestamp encoding')
        start = offset
        if fmt == 'string':
            for _ in range(nchan):
                size, offset = _take_varint(payload, offset)
                offset += size
        else:
            offset += nchan * WIDTHS[fmt]
        if offset > len(payload) or not np.isfinite(stamp):
            raise ValueError('Invalid XDF sample')
        yield stamp, payload[start:offset]
        last = stamp
    if offset != len(payload):
        raise ValueError('Unexpected bytes after XDF samples')


def _session(stream):
    return stream['info'].get('desc', [{}])[0].get('session_name', [''])[0] or ''


def _load(path):
    import pyxdf
    streams, _ = pyxdf.load_xdf(str(path), synchronize_clocks=False, dejitter_timestamps=False)
    return {s['info']['stream_id']: s for s in streams}


def split_recording(source, output, *, context_seconds=0.5):
    """Create output/<session>/xdf/*.xdf plus a verified per-file manifest.

    Task streams are selected by session metadata, never by their name alone.
    Shared streams use observed task sample coverage plus context, capped at
    midpoints between sessions so context cannot duplicate shared samples.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if context_seconds < 0 or not np.isfinite(context_seconds):
        raise ValueError('Context must be finite and nonnegative')
    if output.exists():
        raise FileExistsError(output)
    source_hash = digest(source)
    original = _load(source)
    bounds = defaultdict(list)
    for stream in original.values():
        session = _session(stream)
        times = stream['time_stamps']
        if session and len(times):
            bounds[session].extend((float(times.min()), float(times.max())))
    if len(bounds) < 2:
        raise ValueError('Recording has fewer than two nonempty task sessions')
    ordered = sorted(bounds, key=lambda name: min(bounds[name]))
    for name in ordered:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError(f'Unsafe session directory name: {name}')
    spans = [(min(bounds[name]), max(bounds[name])) for name in ordered]
    if any(a[1] >= b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError('Task sessions overlap in time; manual boundary review is required')
    midpoints = [(a[1] + b[0]) / 2 for a, b in zip(spans, spans[1:])]
    windows = [(max(start - context_seconds, midpoints[i - 1] if i else -np.inf),
                min(stop + context_seconds, midpoints[i] if i < len(midpoints) else np.inf))
               for i, (start, stop) in enumerate(spans)]
    inclusive_end = [i == len(ordered) - 1 or windows[i][1] < midpoints[i] for i in range(len(ordered))]
    headers, file_header, formats, footers = {}, None, {}, {}
    for tag, data in _chunks(source):
        if tag == 1:
            file_header = data
        elif tag == 2:
            sid = struct.unpack_from('<I', data)[0]
            header = ET.fromstring(data[4:])
            nchan = int(header.findtext('channel_count'))
            fmt = header.findtext('channel_format')
            rate = float(header.findtext('nominal_srate'))
            if fmt != 'string' and fmt not in WIDTHS:
                raise ValueError(f'Unsupported XDF data format: {fmt}')
            formats[sid] = (nchan, fmt, 1 / rate if rate else 0)
            headers[sid] = data
        elif tag == 6:
            footers[struct.unpack_from('<I', data)[0]] = data[4:]
    if file_header is None:
        raise ValueError('XDF has no file header')
    owners = {sid: _session(stream) for sid, stream in original.items()}
    selected = [{sid for sid in headers if owners[sid] in ('', name)} for name in ordered]
    handles, paths = [], []
    stats = [defaultdict(lambda: dict(samples=0, first_timestamp=None, last_timestamp=None)) for _ in ordered]
    raw_counts, last_stamps = defaultdict(int), defaultdict(float)
    output.mkdir(parents=True)
    try:
        for i, name in enumerate(ordered):
            path = output / name / 'xdf' / f'{source.stem}_session-{name.rsplit("_", 1)[-1]}.xdf'
            path.parent.mkdir(parents=True)
            paths.append(path)
            handle = path.open('xb')
            handles.append(handle)
            handle.write(b'XDF:')
            _write(handle, 1, file_header)
            for sid, header in headers.items():
                if sid in selected[i]:
                    _write(handle, 2, header)
        for tag, data in _chunks(source):
            if tag == 3:
                sid = struct.unpack_from('<I', data)[0]
                buffers, counts = {}, defaultdict(int)
                for stamp, value in _samples(data, formats[sid], last_stamps[sid]):
                    raw_counts[sid] += 1
                    last_stamps[sid] = stamp
                    if owners[sid]:
                        target = ordered.index(owners[sid]) if owners[sid] in ordered else None
                    else:
                        target = bisect_right(midpoints, stamp)
                        lo, hi = windows[target]
                        if not lo <= stamp <= hi:
                            target = None
                    if target is None:
                        continue
                    buffers.setdefault(target, bytearray()).extend(b'\x08' + struct.pack('<d', stamp) + value)
                    counts[target] += 1
                    row = stats[target][sid]
                    if row['first_timestamp'] is None:
                        row['first_timestamp'] = stamp
                    row['last_timestamp'] = stamp
                    row['samples'] += 1
                for target, buffer in buffers.items():
                    _write(handles[target], 3, struct.pack('<I', sid) + _varint(counts[target]) + buffer)
            elif tag == 4:
                sid = struct.unpack_from('<I', data)[0]
                for i, handle in enumerate(handles):
                    if sid in selected[i]:
                        _write(handle, 4, data)  # Retain complete original clock correction evidence.
            elif tag == 5:
                for handle in handles:
                    _write(handle, 5, data)
            elif tag not in (1, 2, 6):
                raise ValueError(f'Unsupported XDF extension chunk {tag}; refusing to discard it')
        for sid, stream in original.items():
            if raw_counts[sid] != len(stream['time_stamps']):
                raise ValueError(f'Original stream {sid} raw/pyxdf sample counts differ')
        for i, handle in enumerate(handles):
            for sid in selected[i]:
                footer = ET.fromstring(footers.get(sid, b'<info/>'))
                row = stats[i][sid]
                for key, value in (('sample_count', row['samples']),
                                   ('first_timestamp', row['first_timestamp'] or 0),
                                   ('last_timestamp', row['last_timestamp'] or 0)):
                    node = footer.find(key)
                    if node is None:
                        node = ET.SubElement(footer, key)
                    node.text = str(value)
                _write(handle, 6, struct.pack('<I', sid) + ET.tostring(footer, encoding='utf-8'))
    finally:
        for handle in handles:
            handle.close()
    results = []
    for i, path in enumerate(paths):
        loaded = _load(path)
        if set(loaded) != selected[i]:
            raise ValueError('Split XDF stream IDs differ from plan')
        for sid, stream in loaded.items():
            src = original[sid]
            times = src['time_stamps']
            lo, hi = windows[i]
            keep = (np.ones(len(times), bool) if owners[sid] else
                    (times >= lo) & ((times <= hi) if inclusive_end[i] else (times < hi)))
            expected = ([src['time_series'][j] for j in np.flatnonzero(keep)]
                        if isinstance(src['time_series'], list) else src['time_series'][keep])
            equal = (stream['time_series'] == expected if isinstance(expected, list)
                     else np.array_equal(stream['time_series'], expected, equal_nan=True))
            if not np.array_equal(stream['time_stamps'], times[keep]) or not equal:
                raise ValueError(f'Split samples or timestamps differ for stream {sid}')
            if not np.array_equal(stream['clock_times'], src['clock_times']) or not np.array_equal(
                    stream['clock_values'], src['clock_values']):
                raise ValueError(f'Clock correction data differ for stream {sid}')
        row = dict(session=ordered[i], source=str(source), source_sha256=source_hash,
                   output=str(path), output_sha256=digest(path),
                   task_observed_start=spans[i][0], task_observed_end=spans[i][1],
                   shared_window_start=windows[i][0], shared_window_end=windows[i][1],
                   shared_window_end_inclusive=inclusive_end[i],
                   context_seconds=context_seconds, duplicate_streams='preserved',
                   timestamps='original, no synchronization/dejitter/resampling',
                   verification='all retained samples/timestamps and clock offsets exactly equal',
                   streams={str(sid): dict(name=original[sid]['info']['name'][0], **stats[i][sid])
                            for sid in sorted(selected[i])})
        (path.parent.parent / 'xdf_split_manifest.json').write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding='utf-8')
        results.append(row)
    if digest(source) != source_hash:
        raise ValueError('Source XDF changed during splitting')
    return results
