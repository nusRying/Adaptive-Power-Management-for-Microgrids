Conda Environment Backup
======================

This folder contains exported package definitions for each Conda environment.

Per-environment files:
- environment.yml            (portable, no build strings)
- environment.full.yml       (includes build strings)
- conda-list-export.txt      (conda export format)
- conda-explicit-spec.txt    (exact explicit URLs)
- pip-freeze.txt             (pip packages visible from that env)

Recreate examples:
1) conda env create -f <env_folder>\environment.yml
2) conda create --name <new_env_name> --file <env_folder>\conda-list-export.txt
3) conda create --name <new_env_name> --file <env_folder>\conda-explicit-spec.txt

Summary file: summary.json
