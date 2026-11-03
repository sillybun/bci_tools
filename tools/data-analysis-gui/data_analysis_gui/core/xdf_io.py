"""XDF header scanning and lossless loading of the analysis streams.

Loading keeps the receive timestamps untouched (no clock sync, no dejitter), the
protocol both source projects use. Same-name duplicate streams are merged only
after verifying that overlapping samples are identical.
"""
from __future__ import annotations

import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def spike_stream_aliases(names) -> dict[str, str]:
    """Map a complete pipe pair to the analysis names when no legacy pair exists."""
    pipe = {'spike_counts_pipe': 'spike_counts',
            'spike_counts_pipe_metadata': 'spike_counts_metadata'}
    names = set(names)
    if set(pipe) <= names and not set(pipe.values()) <= names:
        return pipe
    return {}


def read_headers(path) -> list[ET.Element]:
    """Stream XML headers only; sample chunks are skipped with seek()."""
    headers = []
    with Path(path).open('rb') as file:
        if file.read(4) != b'XDF:':
            raise ValueError('Invalid XDF signature')
        size_total = os.fstat(file.fileno()).st_size
        while width := file.read(1):
            if width[0] not in (1, 4, 8):
                raise ValueError('Invalid XDF chunk length')
            raw = file.read(width[0])
            if len(raw) != width[0]:
                raise ValueError('Truncated XDF chunk length')
            size = int.from_bytes(raw, 'little')
            if size < 2 or file.tell() + size > size_total:
                raise ValueError('Truncated XDF chunk')
            tag = int.from_bytes(file.read(2), 'little')
            if tag != 2:
                file.seek(size - 2, 1)
                continue
            if size < 6 or size > 8 * 1024 * 1024:
                raise ValueError('Invalid XDF stream header size')
            file.read(4)
            headers.append(ET.fromstring(file.read(size - 6)))
    return headers


def _stream_id(stream):
    value = stream['info'].get('stream_id', 'unknown')
    return value[0] if isinstance(value, list) else value


def canonical_streams(streams) -> tuple[dict, list[dict]]:
    """One stream per name; merge verified duplicates, refuse conflicting ones."""
    grouped: dict[str, list] = {}
    for stream in streams:
        grouped.setdefault(stream['info']['name'][0], []).append(stream)
    result, audit = {}, []
    for name, candidates in grouped.items():
        nonempty = [s for s in candidates if len(s['time_stamps'])]
        if not nonempty:
            continue
        primary = max(nonempty, key=lambda s: len(s['time_stamps']))
        times = np.asarray(primary['time_stamps'], dtype=float)
        values = primary['time_series']
        is_list = isinstance(values, list)
        values = list(values) if is_list else np.asarray(values)
        for other in nonempty:
            if other is primary:
                continue
            other_times = np.asarray(other['time_stamps'], dtype=float)
            if is_list:
                # Marker streams: keep the larger copy, union unseen timestamps.
                seen = set(times.tolist())
                extra = [(t, v) for t, v in zip(other_times, other['time_series']) if t not in seen]
                if extra:
                    times = np.r_[times, [t for t, _ in extra]]
                    values = values + [v for _, v in extra]
                    order = np.argsort(times, kind='stable')
                    times, values = times[order], [values[i] for i in order]
                audit.append(dict(name=name, merged_stream=_stream_id(other), extra_samples=len(extra)))
                continue
            other_values = np.asarray(other['time_series'])
            index = np.searchsorted(times, other_times)
            shared = index < len(times)
            shared[shared] &= times[index[shared]] == other_times[shared]
            if shared.any() and not np.array_equal(values[index[shared]], other_values[shared]):
                raise ValueError(f'Conflicting duplicate streams named {name}')
            unique = ~shared
            if unique.any():
                times = np.r_[times, other_times[unique]]
                values = np.concatenate([values, other_values[unique]])
                order = np.argsort(times, kind='stable')
                times, values = times[order], values[order]
            audit.append(dict(name=name, merged_stream=_stream_id(other),
                              shared_samples=int(shared.sum()), extra_samples=int(unique.sum())))
        result[name] = dict(info=primary['info'], time_stamps=times, time_series=values)
    aliases = spike_stream_aliases(result)
    for source, target in aliases.items():
        result[target] = result[source]
    for entry in audit:
        entry['name'] = aliases.get(entry['name'], entry['name'])
    return result, audit


def load_streams(path) -> tuple[dict, list[dict]]:
    import pyxdf
    streams, _ = pyxdf.load_xdf(str(path), synchronize_clocks=False, dejitter_timestamps=False)
    return canonical_streams(streams)


def desc_value(stream, key, default=None):
    try:
        return stream['info']['desc'][0][key][0]
    except (KeyError, IndexError, TypeError):
        return default


def channel_labels(stream) -> list[str]:
    try:
        return [c['label'][0] for c in stream['info']['desc'][0]['channels'][0]['channel']]
    except (KeyError, IndexError, TypeError):
        return []


def electrode_ids(stream) -> np.ndarray:
    """One-based electrode id per stored column, from explicit header metadata."""
    raw = desc_value(stream, 'channel_ids')
    if raw:
        ids = np.array([int(v) for v in str(raw).split(',') if v.strip()], dtype=int)
    else:
        labels = channel_labels(stream)
        ids = np.array([int(l.removeprefix('electrode_')) for l in labels], dtype=int)
    if len(ids) != len(np.unique(ids)) or (len(ids) and ids.min() < 1):
        raise ValueError('Electrode ids must be unique and one-based')
    return ids
