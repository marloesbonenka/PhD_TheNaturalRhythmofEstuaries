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


def _save_pickle(cache_path: Path, payload: dict) -> None:
    """Internal: save dict to pickle file."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_pickle(cache_path: Path) -> dict:
    """Internal: load dict from pickle file."""
    with cache_path.open('rb') as f:
        return pickle.load(f)


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
    return model_location / f"cache_profiles_{folder}.pkl"


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
        cached = _load_pickle(cache_path)
        results = cached.get('results', {})
        
        found = {}
        missing = []
        
        for x_coord in required_x_coords:
            cs_name = f"km{int(x_coord / 1000)}"
            if cs_name in results:
                found[cs_name] = results[cs_name]
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
        try:
            cached = _load_pickle(cache_path)
            existing = cached.get('results', {})
        except:
            pass
    
    # Merge: new results overwrite existing for same cross-sections
    merged = {**existing, **results}
    
    _save_pickle(cache_path, {
        'results': merged,
        'metadata': {
            'settings': settings or {},
        },
    })


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
        cached = _load_pickle(cache_path)
        results = cached.get('results', {})
        metadata = cached.get('metadata', {})
        
        if validate_settings and settings is not None:
            cached_settings = metadata.get('settings', {})
            if cached_settings != settings:
                print(f"  Cache settings differ; will recompute.")
                return None, None
        
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
        try:
            cached = _load_pickle(cache_path)
            existing = cached.get('results', {})
        except:
            pass
    
    # Merge: new results overwrite existing for same keys
    merged = {**existing, **results}
    
    full_metadata = metadata.copy() if metadata else {}
    full_metadata['settings'] = settings or {}
    
    _save_pickle(cache_path, {
        'results': merged,
        'metadata': full_metadata,
    })
