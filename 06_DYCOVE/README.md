# DYCOVE vegetation coupling

This folder contains scripts and documentation for coupling Delft3D-FM simulations with the DYCOVE vegetation model.

The DYCOVE workflow uses a dedicated Python environment (`dycove-env`) that is separate from the main project environment (`estuary-env`). This avoids dependency conflicts and improves reproducibility.

The repository contains local copies of:

- bmi-python
- dycove-model
- dfm_tools

These dependencies are included to provide a self-contained and reproducible workflow.

## Directory structure

06_DYCOVE/
│
├── README.md
├── dycove-env.yml
└── scripts/

Suggested usage:

- scripts/ → coupling and processing scripts

No model outputs, configuration files, temporary files, or user-specific settings should be committed to Git.

## Software requirements

### Delft3D-FM

A working Delft3D-FM installation is required (version 2026.01 does NOT work with bmi-toolbox, use an earlier version).

The Delft3D-FM executable and BMI libraries are not installed by the Python environment and must be available locally.

### Included dependencies

This repository contains local copies of:

- bmi-python
- dycove-model
- dfm_tools

These correspond to the versions used during development.

Original repositories:

BMI-python:
https://github.com/openearth/bmi-python

DYCOVE:
https://github.com/Ecomomo-lab/dycove-model

## Creating the Python environment

Open the Miniforge Prompt and navigate to the DYCOVE folder:

    cd <path_to_repository>\06_DYCOVE

Create the environment:

    mamba env create -f environment.yml

Activate the environment:

    conda activate dycove-env

## Installing the local packages

Install BMI-python:

    cd <path_to_repository>\external\bmi-python
    pip install -e .

Install DYCOVE:

    cd <path_to_repository>\external\dycove-model
    pip install -e .

## Verifying the installation

Check core dependencies:

    python -c "import numpy, scipy, pandas, xarray; print('Core dependencies OK')"

Check DYCOVE:

    python -c "import dycove; print('DYCOVE OK')"

Check BMI-python:

    python -c "import bmi; print('BMI OK')"

## Updating the environment

The environment.yml file is the single source of truth for this environment.

When adding or updating packages:

1. Edit environment.yml
2. Save the file
3. Run:

    conda activate dycove-env

    mamba env update -f environment.yml --prune

The --prune flag removes packages that are no longer listed and keeps the environment synchronized with the yaml file.

## Rebuilding the environment

If dependency conflicts occur:

    mamba env remove -n dycove-env

    mamba env create -f environment.yml

Reinstall the local packages:

    cd <path_to_repository>\external\bmi-python
    pip install -e .

    cd <path_to_repository>\external\dycove-model
    pip install -e .

## Version information

Current development setup:

    Python      : 3.10
    DYCOVE      : 0.1.0
    BMI-python  : 0.3.0

Repository versions used during development:

    BMI-python commit:
    c07d892ff87215c8a3399c05b93e5b492b6ae6bc

    DYCOVE commit:
    51480d26b28dcc1e774b2428b4bf9c2400b703bd

For reproducibility, document:

- Delft3D-FM version
- DYCOVE commit hash
- BMI-python commit hash
- Analysis date
- Any local modifications

## Notes

- Keep the DYCOVE workflow separate from the main estuary-env environment.
- Do not install packages without updating environment.yml.
- Store all DYCOVE-related scripts in the scripts folder.
- Do not commit model results, temporary files, user-specific settings, or machine-specific configuration files.
- If bmi-python or dycove-model are modified locally, document the changes before publication or collaboration.