"""Temporarily run numpy's bundled OpenBLAS single-threaded (no extra dependency).

The SDK Runtimes replay one frame at a time; each frame solves a ~500×500 system.
With OpenBLAS' default thread pool (24 threads here) such small solves are ~20×
slower than single-threaded. This uses the `scipy_openblas_{get,set}_num_threads64_`
symbols of numpy's bundled OpenBLAS via ctypes and restores the previous count;
if the library or symbols cannot be found it does nothing.
"""
from __future__ import annotations

import contextlib
import ctypes
import glob
import os
import threading

import numpy as np

_LOCK = threading.Lock()
_API = None


def _api():
    global _API
    if _API is None:
        _API = False
        root = os.path.dirname(np.__file__)
        patterns = [os.path.join(root, '..', 'numpy.libs', '*openblas*'), os.path.join(root, '.dylibs', '*openblas*'),
                    os.path.join(root, '..', 'numpy.libs', '*.so*')]
        for path in sum((glob.glob(p) for p in patterns), []):
            try:
                lib = ctypes.CDLL(path)
            except OSError:
                continue
            for suffix in ('64_', ''):
                get = getattr(lib, f'scipy_openblas_get_num_threads{suffix}', None)
                put = getattr(lib, f'scipy_openblas_set_num_threads{suffix}', None)
                if get and put:
                    _API = (get, put)
                    return _API
    return _API


@contextlib.contextmanager
def single_thread():
    api = _api()
    if not api:
        yield
        return
    get, put = api
    with _LOCK:
        previous = int(get())
        put(1)
    try:
        yield
    finally:
        with _LOCK:
            put(previous)
