"""Caching utilities for datasets and small result payloads."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Hashable, Iterable, Tuple

import xarray as xr
import dfm_tools as dfmt


def _normalize_path(path: Any) -> Hashable:
    if isinstance(path, (list, tuple)):
        return tuple(str(Path(p)) for p in path)
    return str(Path(path))


def _kwargs_key(kwargs: dict) -> Hashable:
    if not kwargs:
        return ()
    try:
        return tuple(sorted(kwargs.items()))
    except TypeError:
        return repr(kwargs)


def save_cache(cache_path: Path, payload: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache(cache_path: Path) -> dict:
    with cache_path.open('rb') as f:
        return pickle.load(f)


class DatasetCache:
    """Cache for xarray and dfm-tools datasets to avoid repeated loading."""

    def __init__(self) -> None:
        self._partitioned: dict[Hashable, Any] = {}
        self._xr: dict[Hashable, xr.Dataset] = {}

    def get_partitioned(self, file_pattern: str, **kwargs: Any):
        key = (_normalize_path(file_pattern), _kwargs_key(kwargs))
        if key not in self._partitioned:
            print(f"Loading partitioned dataset: {key[0]}")
            self._partitioned[key] = dfmt.open_partitioned_dataset(file_pattern, **kwargs)
        return self._partitioned[key]

    def get_xr(self, path: Any, **kwargs: Any) -> xr.Dataset:
        key = (_normalize_path(path), _kwargs_key(kwargs))
        if key not in self._xr:
            print(f"Loading dataset: {key[0]}")
            self._xr[key] = xr.open_dataset(path, **kwargs)
        return self._xr[key]

    def close_all(self) -> None:
        for key, ds in {**self._partitioned, **self._xr}.items():
            try:
                ds.close()
            except Exception as exc:
                print(f"Warning: failed to close dataset {key}: {exc}")
        self._partitioned.clear()
        self._xr.clear()
