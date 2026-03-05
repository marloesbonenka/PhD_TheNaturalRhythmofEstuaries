"""NetCDF-based cache helpers for map output (no pickle)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import xarray as xr
import xugrid as xu
import dfm_tools as dfmt


def cache_tag_from_bbox(bbox, tag_override=None) -> str:
    if tag_override:
        return str(tag_override)
    if bbox is None:
        return "full"
    return f"bbox_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}"


def get_cache_path(cache_dir: Path, folder_name: str, var_name: str, tag: str | None = None) -> Path:
    tag_suffix = f"_{tag}" if tag and tag != "full" else ""
    return cache_dir / f"mapoutput_{var_name}_{folder_name}{tag_suffix}.nc"


def select_cache_path(cache_dir: Path, folder_name: str, var_name: str, tag: str | None = None) -> Path:
    tagged = get_cache_path(cache_dir, folder_name, var_name, tag)
    legacy = cache_dir / f"assessment_multi_{folder_name}.nc"
    if tagged.exists():
        return tagged
    if legacy.exists():
        return legacy
    return tagged


def load_or_update_map_cache_multi(
    cache_dir: Path,
    folder_name: str,
    run_paths: Iterable,
    var_names,
    bbox=None,
    append_time=True,
    append_vars=True,
    chunks=None,
    cache_tag=None,
):
    datasets = []
    for var_name in var_names:
        cache_path = select_cache_path(cache_dir, folder_name, var_name, cache_tag)
        ds = load_or_update_map_cache(
            cache_path=cache_path,
            run_paths=run_paths,
            var_names=[var_name],
            bbox=bbox,
            append_time=append_time,
            append_vars=append_vars,
            chunks=chunks,
            cache_tag=cache_tag,
        )
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        return None
    if len(datasets) == 1:
        return datasets[0]
    return xr.merge(datasets, compat='no_conflicts', join='outer')


def _filter_new_times(ds_new: xu.UgridDataset, ds_existing: xu.UgridDataset):
    if 'time' not in ds_new.dims or 'time' not in ds_existing.dims:
        return ds_new
    existing_times = ds_existing['time'].values
    new_times = ds_new['time'].values
    keep_mask = ~np.isin(new_times, existing_times)
    if keep_mask.any():
        return ds_new.isel(time=keep_mask)
    return None


def _merge_for_cache(ds_existing, ds_new, append_time: bool, append_vars: bool):
    if ds_existing is None:
        return ds_new
    if not append_time and not append_vars:
        return ds_new

    if not append_time and append_vars:
        new_only_vars = [v for v in ds_new.data_vars if v not in ds_existing.data_vars]
        if not new_only_vars:
            return ds_existing
        ds_new_only = ds_new[new_only_vars]
        if 'time' in ds_existing.dims and 'time' in ds_new_only.dims:
            ds_new_only = ds_new_only.reindex(time=ds_existing['time'])
        return xr.merge([ds_existing, ds_new_only], compat='no_conflicts', join='outer')

    shared_vars = [v for v in ds_new.data_vars if v in ds_existing.data_vars]
    new_only_vars = [v for v in ds_new.data_vars if v not in ds_existing.data_vars] if append_vars else []
    existing_only_vars = [v for v in ds_existing.data_vars if v not in ds_new.data_vars]

    ds_new_filtered = _filter_new_times(ds_new, ds_existing)
    if ds_new_filtered is None and not new_only_vars:
        return ds_existing

    parts = []
    if existing_only_vars:
        parts.append(ds_existing[existing_only_vars])

    if shared_vars:
        if ds_new_filtered is not None:
            shared_concat = xr.concat([ds_existing[shared_vars], ds_new_filtered[shared_vars]], dim='time')
        else:
            shared_concat = ds_existing[shared_vars]
        parts.append(shared_concat)

    if new_only_vars:
        parts.append(ds_new[new_only_vars])

    if not parts:
        return ds_existing

    return xr.merge(parts, compat='no_conflicts', join='outer')


def _to_file_pattern(path_like) -> str:
    path_str = str(path_like)
    if '*' in path_str or path_str.endswith('.nc'):
        return path_str
    return str(Path(path_like) / "output" / "*_map.nc")


def build_masked_map_dataset(run_paths: Iterable, var_names, bbox=None, chunks=None):
    datasets = []
    for run_path in run_paths:
        file_pattern = _to_file_pattern(run_path)
        part_ds = dfmt.open_partitioned_dataset(
            file_pattern,
            variables=var_names,
            chunks=chunks or {'time': 100}
        )
        datasets.append(part_ds)
    if not datasets:
        return None

    full_ds = xu.concat(datasets, dim="time")
    if bbox is not None:
        full_ds = full_ds.ugrid.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[1], bbox[3]))
    return full_ds


def load_or_update_map_cache(
    cache_path: Path,
    run_paths: Iterable,
    var_names,
    bbox=None,
    append_time=True,
    append_vars=True,
    chunks=None,
    cache_tag=None,
):
    ds_existing = None
    if cache_path.exists():
        ds_existing = xu.open_dataset(cache_path)

    ds_new = build_masked_map_dataset(run_paths, var_names, bbox=bbox, chunks=chunks)
    if ds_new is None:
        if ds_existing is not None:
            return ds_existing
        return None

    ds_out = _merge_for_cache(ds_existing, ds_new, append_time, append_vars)

    if ds_out is None:
        if ds_existing is not None:
            ds_existing.close()
        return None

    if cache_tag is not None:
        ds_out.attrs['cache_tag'] = str(cache_tag)
    ds_out.attrs['cache_bbox'] = "full" if bbox is None else str(list(bbox))

    should_write = (not cache_path.exists()) or (ds_out is not ds_existing)
    if should_write:
        comp = dict(zlib=True, complevel=4)
        encoding = {v: comp for v in ds_out.data_vars}
        if hasattr(ds_out, 'ugrid'):
            ds_out.ugrid.to_netcdf(cache_path, encoding=encoding)
        else:
            ds_out.to_netcdf(cache_path, encoding=encoding)

    if ds_existing is not None and ds_existing is not ds_out:
        ds_existing.close()

    return ds_out
