# Managing the `estuary-env` Python environment

This file documents how to add/update packages in the `estuary-env` conda
environment used for Delft3D-FM post-processing + global data analysis. 
`environment.yml` is the **single source of truth**: it should always reflect 
exactly what's installed. Don't `pip install` or `conda install` something 
one-off without also adding it here, or the environment becomes undocumented 
and unreproducible.

## Setup (for reference)
- Installer: **Miniforge** (not Anaconda Navigator, keeps things minimal)
- Channel: **conda-forge only** (no `defaults`)
- Environment file location: keep a backup copy in Nextcloud/GitHub, but
  run the actual `mamba` commands from a local (non-synced) folder if
  possible, to avoid file-lock conflicts during install

## Editing `environment.yml`
- Packages available on conda-forge → add to the main `dependencies:` list
- Packages only on PyPI (e.g. `dfm_tools`) → add under the `pip:` sub-list
  at the bottom (make sure `pip` itself is also listed as a dependency)

```yaml
name: estuary-env
channels:
  - conda-forge
dependencies:
  - python=3.11
  - xarray
  - xugrid
  - numpy
  - pandas
  - matplotlib
  - pillow
  - netcdf4
  - dask
  - ipykernel
  - cmocean
  - pip
  - pip:
      - dfm_tools
```

## Adding or updating a package

**Step 1 — Open the Miniforge Prompt** (Start menu → "Miniforge Prompt")

**Step 2 — Edit `environment.yml`**
Add the new package name to the appropriate list (conda section or `pip:`
section). Save the file.

**Step 3 — Apply the change**

Try the lighter option first:
```
conda activate estuary-env
cd /d <path to folder containing environment.yml>
mamba env update -f environment.yml --prune
```
`--prune` removes anything no longer listed in the file, keeping the
environment matched exactly to the yml.

**Step 4 — If `update` fails with a solver conflict**

This can happen if the environment has been through several partial/failed
installs and ends up in an inconsistent state (as happened on 25 June
2026 — `ipykernel`/`jupyter_client` conflict). Don't try to fight the
solver — rebuild from scratch instead, since the yml file is the actual
source of truth and nothing irreplaceable lives *inside* the environment:

```
mamba env remove -n estuary-env
mamba env create -f environment.yml
```

**Step 5 — Verify**
```
python -c "import xarray, xugrid, numpy, pandas, matplotlib, dask, netCDF4, PIL, cmocean; print('core stack OK')"
python -c "import dfm_tools; print('dfm_tools OK')"
where python
```
Confirm the path points into `...\miniforge3\envs\estuary-env\python.exe`.

**Step 6 — Back in VS Code**
No reconfiguration needed — interpreter path and settings stay the same
across a rebuild, since the environment is recreated at the identical
location/name. If VS Code still shows stale errors (e.g. Pylance import
warnings) after a rebuild:
```
Ctrl+Shift+P → "Developer: Reload Window"
```

## Quick troubleshooting reference

| Symptom | Likely cause | Fix |
|---|---|---|
| `mamba activate` gives "Shell not initialized" | mamba's activate hook isn't set up | Use `conda activate` instead |
| `conda activate <name>` says env not found | `mamba env create` never finished or wasn't run | Run `mamba env create -f environment.yml` first |
| Solver error listing ancient Python 2.7/3.4 versions | `defaults` channel got mixed in, or env is in a broken state | Check `conda config --show channels`; if clean, just remove + recreate the env |
| Pylance shows "Import could not be resolved" but code runs fine in terminal | VS Code's analysis interpreter ≠ terminal's active interpreter | Click interpreter in status bar → reselect `estuary-env` |
| Install/update seems to hang with no output | Often just the dependency *solving* step (slow first time) | Wait a few minutes; check Task Manager for CPU activity before assuming it's frozen |

## One-off installs (use sparingly)
Quick test without editing the yml:
```
conda activate estuary-env
pip install <package>     # PyPI-only packages
mamba install <package>   # conda-forge packages
```
If you keep using it, add it to `environment.yml` afterward so the file
stays accurate.
