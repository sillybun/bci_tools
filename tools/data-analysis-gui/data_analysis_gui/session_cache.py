"""Reusable numeric intermediates keyed by recording content and computation inputs."""
from dataclasses import fields
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import scipy

from . import storage


# Number of leading recording/cohort IDs and any section-specific code dependencies.
# These artifacts never depend on the full selected recording set.
ARTIFACTS = {
    'cohort': (1, ()),
    'trial_window_rates': (1, ()),
    'cohort_velocity_audit': (1, ()),
    's02_code_stats': (1, ('s02_channel_max_count_v1',)),
    's03_fpga': (1, ('shared_frame_loss.py',)),
    's03_transmission_totals': (1, ('shared_frame_loss.py',)),
    'anova': (1, ()),
    'anova_finite': (1, ()),
    's06_temporal_statistics': (1, ('s06_psth_v1', 's01_raw_visualization_v1')),
    'direction_decode': (1, ('s09_direction_decoding_v1',)),
    'decoder_data': (1, ()),
    'decoder_cv': (1, ()),
    'xday_direction': (1, ()),
    'xday_tdr_own': (1, ('x03_cross_day_tdr_v1',)),
    'xday_tdr_plane': (2, ('x03_cross_day_tdr_v1',)),
    'xday_direction_model': (2, ()),
    'xday_transfer': (2, ('x05_cross_day_velocity_decoding_v1',)),
    'xday_ceiling_dir': (1, ('x06_offline_model_transfer_v1',)),
    'xday_model_transfer': (2, ('x06_offline_model_transfer_v1',)),
    'xday_online_check': (1, ('x06_offline_model_transfer_v1',)),
    'xday_tuning_profile': (1, ('x07_tuning_stability_v1', 's05_anova_v1', 's06_psth_v1',
                               's01_raw_visualization_v1')),
}


def _pack(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Object arrays are not supported by the session cache')
        key = f'a{len(arrays)}'
        arrays[key] = value
        return ['array', key]
    if isinstance(value, np.generic):
        return _pack(value.item(), arrays)
    if isinstance(value, dict):
        return ['dict', [[_pack(k, arrays), _pack(v, arrays)] for k, v in value.items()]]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [_pack(v, arrays) for v in value]]
    if value is None or isinstance(value, (str, bool, int, float)):
        return ['scalar', value]
    from .decoders import DecoderData
    if isinstance(value, DecoderData):
        return ['decoder_data', _pack({f.name: getattr(value, f.name) for f in fields(value)}, arrays)]
    raise TypeError(f'Unsupported cache value: {type(value).__name__}')


def _unpack(node, arrays):
    kind, value = node
    if kind == 'array':
        return arrays[value]
    if kind == 'dict':
        return {_unpack(k, arrays): _unpack(v, arrays) for k, v in value}
    if kind == 'list':
        return [_unpack(v, arrays) for v in value]
    if kind == 'tuple':
        return tuple(_unpack(v, arrays) for v in value)
    if kind == 'scalar':
        return value
    if kind == 'decoder_data':
        from .decoders import DecoderData
        return DecoderData(**_unpack(value, arrays))
    raise ValueError(f'Unknown cache encoding: {kind}')


class SessionCache:
    def __init__(self, output_root, *, refresh=False):
        self.root = Path(output_root) / '.session-cache'
        self.refresh = refresh
        self.identities = {}
        self.code = {}
        self.hits = self.misses = 0

    def bind_session(self, session, sha256):
        from .core.session import recording_window_policy

        # Task reconstruction also uses the task mode resolved from the file name/header.
        identity = dict(sha256=sha256, task=session.task_mode)
        window = recording_window_policy(session.recording)
        if window is not None:
            identity['analysis_window_policy'] = window
        self.identities[id(session)] = identity

    def bind_cohort(self, cohort):
        self.identities[id(cohort)] = dict(self.identities[id(cohort.session)],
                                           filter=cohort.criteria.to_dict())

    def _code(self, kind, dependencies):
        if kind not in self.code:
            paths = [storage.PACKAGE / 'session_cache.py', storage.PACKAGE / 'config.py']
            paths += [storage.PACKAGE / 'sections' / path for path in dependencies]
            self.code[kind] = [storage.source_digest(path) for path in paths]
        return self.code[kind]

    def remember(self, key, compute):
        if not isinstance(key, tuple) or not key or key[0] not in ARTIFACTS:
            return compute()
        kind = key[0]
        count, dependencies = ARTIFACTS[kind]
        ids = key[1:count + 1]
        if len(ids) != count or any(identity not in self.identities for identity in ids):
            return compute()
        records = [self.identities[identity] for identity in ids]
        # S02 has a recording-scoped whole-recording key and a cohort-scoped filtered variant.
        settings = key[count + 1:]
        if kind == 's02_code_stats' and settings[-1] != 0:
            cohort = self.identities.get(settings[-1])
            if cohort is None:
                return compute()
            settings = settings[:-1] + (cohort,)
        signature = storage.fingerprint(dict(records=records, kind=kind, settings=settings,
                                             code=self._code(kind, dependencies),
                                             numpy=np.__version__, scipy=scipy.__version__))
        path = self.root / records[0]['sha256'][:20] / kind / f'{signature}.npz'
        if not self.refresh:
            try:
                with np.load(path, allow_pickle=False) as archive:
                    value = _unpack(json.loads(archive['metadata'].item()), archive)
                self.hits += 1
                return value
            except (OSError, ValueError, EOFError, KeyError, TypeError, zipfile.BadZipFile):
                pass
        value = compute()
        self.misses += 1
        self._write(path, value)
        return value

    @staticmethod
    def _write(path, value):
        temp = None
        try:
            arrays = {}
            metadata = json.dumps(_pack(value, arrays), ensure_ascii=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as handle:
                temp = Path(handle.name)
                np.savez_compressed(handle, metadata=metadata, **arrays)
            os.replace(temp, path)
        except (OSError, TypeError, ValueError):
            # The cache is optional. A failed write must not prevent exporting the analysis.
            pass
        finally:
            if temp is not None:
                try:
                    temp.unlink(missing_ok=True)
                except OSError:
                    pass
