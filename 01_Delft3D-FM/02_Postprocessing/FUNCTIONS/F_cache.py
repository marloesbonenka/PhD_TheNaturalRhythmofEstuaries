"""Caching utilities for datasets and small result payloads."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Hashable, Iterable, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import dfm_tools as dfmt
from netCDF4 import Dataset as NcDataset


def _normalize_path(path: Any) -> Hashable:
    if isinstance(path, (list, tuple)):
        return tuple(str(Path(p)) for p in path)
    return str(Path(path))


def _kwargs_key(kwargs: dict) -> Hashable:
    if not kwargs:
        return ()
    try:
        # Convert nested dicts to tuples for hashability
        def make_hashable(v):
            if isinstance(v, dict):
                return tuple(sorted((k, make_hashable(val)) for k, val in v.items()))
            elif isinstance(v, list):
                return tuple(make_hashable(item) for item in v)
            return v
        return tuple(sorted((k, make_hashable(v)) for k, v in kwargs.items()))
    except TypeError:
        return repr(kwargs)


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ('M', 'm'):
            data = obj.astype('datetime64[ns]').astype(str).tolist()
            return {'__type__': 'ndarray_datetime', 'data': data}
        return {'__type__': 'ndarray', 'dtype': str(obj.dtype), 'shape': list(obj.shape), 'data': obj.tolist()}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.DataFrame):
        data = obj.to_dict(orient='list')
        return {'__type__': 'dataframe', 'data': _to_jsonable(data)}
    if isinstance(obj, pd.Series):
        data = obj.to_dict()
        return {'__type__': 'series', 'data': _to_jsonable(data)}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.datetime64, np.timedelta64)):
        return str(obj)
    return obj


def _from_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict) and '__type__' in obj:
        obj_type = obj['__type__']
        if obj_type == 'ndarray':
            arr = np.array(obj['data'], dtype=obj.get('dtype', None))
            return arr
        if obj_type == 'ndarray_datetime':
            return np.array(obj['data'], dtype='datetime64[ns]')
        if obj_type == 'dataframe':
            return pd.DataFrame(obj['data'])
        if obj_type == 'series':
            return pd.Series(obj['data'])
    if isinstance(obj, dict):
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


def _encode_attrs(payload: dict) -> str:
    return json.dumps(_to_jsonable(payload), ensure_ascii=True)


def _decode_attrs(payload: str | None) -> dict:
    if not payload:
        return {}
    return _from_jsonable(json.loads(payload))


class DatasetCache:
    """Cache for xarray and dfm-tools datasets to avoid repeated loading."""

    def __init__(self) -> None:
        self._partitioned: dict[Hashable, Any] = {}
        self._xr: dict[Hashable, xr.Dataset] = {}

    def get_partitioned(self, file_pattern: str, variables: list[str] | None = None, **kwargs: Any):
        """Open a partitioned dataset, optionally keeping only selected variables.

        Parameters
        ----------
        file_pattern
            Glob pattern passed to dfm_tools.open_partitioned_dataset.
        variables
            Optional list of variable names to keep (e.g. ['mesh2d_mor_bl']).
            Required UGRID topology variables are always retained to keep
            xugrid merging/topology valid.
        kwargs
            Passed through to dfm_tools.open_partitioned_dataset (and further
            to xarray.open_mfdataset/open_dataset).
        """

        variables_key = tuple(variables) if variables is not None else None
        key = (_normalize_path(file_pattern), variables_key, _kwargs_key(kwargs))
        if key not in self._partitioned:
            print(f"Loading partitioned dataset: {key[0]}")

            # Compose preprocess to keep only the requested vars + required UGRID topology.
            user_preprocess = kwargs.pop('preprocess', None)

            def _keep_topology_and_selected(ds: xr.Dataset) -> xr.Dataset:
                if user_preprocess is not None:
                    ds = user_preprocess(ds)

                if variables is None:
                    return ds

                keep: set[str] = set(variables)
                # Always keep time if present (used all over for indexing)
                if 'time' in ds.variables:
                    keep.add('time')

                # Keep mesh topology and referenced vars so xugrid can merge partitions.
                for name, var in ds.variables.items():
                    if getattr(var, 'attrs', {}).get('cf_role') == 'mesh_topology':
                        keep.add(name)
                        mesh_attrs = getattr(var, 'attrs', {})
                        for attr_name in (
                            'node_coordinates',
                            'face_node_connectivity',
                            'edge_node_connectivity',
                            'face_coordinates',
                            'edge_coordinates',
                            'boundary_node_connectivity',
                            'face_face_connectivity',
                        ):
                            ref = mesh_attrs.get(attr_name)
                            if isinstance(ref, str):
                                keep.update(ref.split())

                # Keep domain/partition indicators (needed for ghost-cell removal).
                # dfm_tools may look for a domain variable during open/merge.
                for name in ds.variables.keys():
                    if 'domain' in name.lower():
                        keep.add(name)

                # Only keep variables that actually exist in this dataset
                keep_existing = [v for v in keep if v in ds.variables]
                return ds[keep_existing]

            if variables is not None or user_preprocess is not None:
                kwargs['preprocess'] = _keep_topology_and_selected

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


# =============================================================================
# Unified Profile Cache System
# =============================================================================

def get_profile_cache_path(model_location: Path, folder: str) -> Path:
    """Get the standard cache path for cross-section profiles."""
    return model_location / f"cache_profiles_{folder}.nc"


def _list_groups(cache_path: Path, root: str) -> list[str]:
    if not cache_path.exists():
        return []
    with NcDataset(cache_path) as nc:
        if root not in nc.groups:
            return []
        return list(nc.groups[root].groups.keys())


def _profiles_to_dataset(data: dict) -> xr.Dataset:
    profiles = np.asarray(data.get('profiles'))
    times = np.asarray(data.get('times') or data.get('time_series'))
    dist = np.asarray(data.get('dist'))
    morfac = data.get('morfac')

    ds = xr.Dataset(
        data_vars={
            'profiles': (['time', 'dist'], profiles),
        },
        coords={
            'time': times,
            'dist': dist,
        },
    )
    if morfac is not None:
        ds['morfac'] = morfac
    ds.attrs['result_type'] = 'profile'
    return ds


def _dataset_to_profile(ds: xr.Dataset) -> dict:
    result = {
        'profiles': ds['profiles'].values,
        'times': ds['time'].values,
        'dist': ds['dist'].values,
    }
    if 'morfac' in ds:
        result['morfac'] = ds['morfac'].values.item() if ds['morfac'].size == 1 else ds['morfac'].values
    return result


def load_profile_cache(
    cache_path: Path, 
    required_x_coords: list, 
    settings: dict = None
) -> tuple[dict, list]:
    """Load profiles from unified cache if available.
    
    Parameters
    ----------
    cache_path : Path
        Path to the cache file.
    required_x_coords : list
        List of x-coordinates needed (in meters).
    settings : dict, optional
        Settings to validate against cached settings (not implemented yet).
        
    Returns
    -------
    tuple[dict, list]
        (results_dict, missing_x_coords)
        results_dict: {cs_name: {'profiles', 'times', 'dist', 'morfac'}}
        missing_x_coords: list of x-coords not found in cache
    """
    if not cache_path.exists():
        return {}, required_x_coords
    
    try:
        found = {}
        missing = []

        for x_coord in required_x_coords:
            cs_name = f"km{int(x_coord / 1000)}"
            group_path = f"profiles/{cs_name}"
            try:
                ds = xr.open_dataset(cache_path, group=group_path)
            except Exception:
                ds = None

            if ds is not None and 'profiles' in ds:
                found[cs_name] = _dataset_to_profile(ds)
                ds.close()
            else:
                missing.append(x_coord)

        return found, missing

    except Exception as e:
        print(f"  Warning: Could not load cache {cache_path}: {e}")
        return {}, required_x_coords


def save_profile_cache(cache_path: Path, results: dict, settings: dict = None) -> None:
    """Save profiles to unified cache, merging with existing data.
    
    Parameters
    ----------
    cache_path : Path
        Path to the cache file.
    results : dict
        {cs_name: {'profiles', 'times', 'dist', 'morfac'}}
    settings : dict, optional
        Settings used to generate the data.
    """
    # Load existing cache if present (to merge)
    existing = {}
    if cache_path.exists():
        for cs_name in _list_groups(cache_path, 'profiles'):
            ds = xr.open_dataset(cache_path, group=f"profiles/{cs_name}")
            existing[cs_name] = _dataset_to_profile(ds)
            ds.close()

    # Merge: new results overwrite existing for same cross-sections
    merged = {**existing, **results}

    root_ds = xr.Dataset()
    root_ds.attrs['settings_json'] = _encode_attrs(settings or {})
    root_ds.to_netcdf(cache_path, mode='w', engine='netcdf4')

    for cs_name, data in merged.items():
        ds = _profiles_to_dataset(data)
        ds.to_netcdf(cache_path, mode='a', group=f"profiles/{cs_name}", engine='netcdf4')


# =============================================================================
# Generic Results Cache (for any analysis type)
# =============================================================================

def load_results_cache(
    cache_path: Path,
    settings: dict = None,
    validate_settings: bool = True,
) -> tuple[dict | None, dict | None]:
    """Load results from cache with optional settings validation.
    
    Parameters
    ----------
    cache_path : Path
        Path to the cache file.
    settings : dict, optional
        Expected settings to validate against cached settings.
    validate_settings : bool
        If True, return None if settings don't match.
        
    Returns
    -------
    tuple[dict | None, dict | None]
        (results, metadata) or (None, None) if cache invalid/missing
    """
    if not cache_path.exists():
        return None, None

    try:
        with NcDataset(cache_path) as nc:
            metadata = _decode_attrs(getattr(nc, 'metadata_json', None))
            cached_settings = _decode_attrs(getattr(nc, 'settings_json', None))

        if validate_settings and settings is not None:
            normalized_settings = _from_jsonable(_to_jsonable(settings))
            if cached_settings != normalized_settings:
                print(f"  Cache settings differ; will recompute.")
                return None, None

        results = {}
        for group_name in _list_groups(cache_path, 'results'):
            ds = xr.open_dataset(cache_path, group=f"results/{group_name}")
            result_type = ds.attrs.get('result_type', 'dataset')
            if result_type == 'dataframe':
                df = ds.to_dataframe().reset_index()
                results[group_name] = df
            elif result_type == 'array':
                results[group_name] = ds['value'].values
            elif result_type == 'dict':
                results[group_name] = {k: ds[k].values for k in ds.data_vars}
            else:
                results[group_name] = ds
            ds.close()

        metadata = metadata or {}
        metadata['settings'] = cached_settings
        return results, metadata

    except Exception as e:
        print(f"  Warning: Could not load cache {cache_path}: {e}")
        return None, None


def save_results_cache(
    cache_path: Path,
    results: dict,
    settings: dict = None,
    metadata: dict = None,
    merge: bool = False,
) -> None:
    """Save results to cache.
    
    Parameters
    ----------
    cache_path : Path
        Path to the cache file.
    results : dict
        Results dictionary (keyed by folder, morfac, etc.).
    settings : dict, optional
        Settings used to generate the data.
    metadata : dict, optional
        Additional metadata (run_names, etc.).
    merge : bool
        If True, merge with existing cache results.
    """
    # Load existing cache if merging
    existing = {}
    if merge and cache_path.exists():
        existing, _ = load_results_cache(cache_path, validate_settings=False)
        if existing is None:
            existing = {}

    # Merge: new results overwrite existing for same keys
    merged = {**existing, **results}

    full_metadata = metadata.copy() if metadata else {}

    root_ds = xr.Dataset()
    root_ds.attrs['metadata_json'] = _encode_attrs(full_metadata)
    root_ds.attrs['settings_json'] = _encode_attrs(settings or {})
    root_ds.to_netcdf(cache_path, mode='w', engine='netcdf4')

    for key, value in merged.items():
        group_name = str(key)
        if isinstance(value, pd.DataFrame):
            ds = value.to_xarray()
            ds.attrs['result_type'] = 'dataframe'
        elif isinstance(value, dict):
            ds = xr.Dataset({k: xr.DataArray(v) for k, v in value.items()})
            ds.attrs['result_type'] = 'dict'
        else:
            arr = np.asarray(value)
            dims = [f"dim_{i}" for i in range(arr.ndim)]
            ds = xr.Dataset({'value': (dims, arr)})
            ds.attrs['result_type'] = 'array'

        ds.to_netcdf(cache_path, mode='a', group=f"results/{group_name}", engine='netcdf4')
