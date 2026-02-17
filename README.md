# Adaptive Power Management for Microgrids

RL-based Energy Management System (EMS) scaffold for a grid-tied microgrid with:
- renewable generation,
- battery energy storage system (BESS),
- utility grid interaction,
- hard safety override layer.

This repository is structured to support your phased plan:
1. environment simulation,
2. MDP definition,
3. continuous-control RL training (SAC/DDPG),
4. integration toward converter/PLC control.

## Current Status
- Core training/evaluation scaffold is implemented and tested.
- Synthetic profile fallback is included so you can develop before real data is available.
- Data validation and chronological split tooling is available for train/val/test workflows.

## What Is Implemented
- Custom `Gymnasium` environment for dispatch and cost optimization: `src/envs/microgrid_env.py`
- Config schema + YAML loaders: `src/config.py`, `configs/microgrid.yaml`, `configs/training.yaml`
- Split-path config for dataset routing: `configs/data_splits.yaml`
- Data pipeline (CSV or synthetic generator): `src/data/profiles.py`
- Data validation and split utilities: `src/data/validation.py`, `src/data/splitting.py`
- RL trainer entrypoint (`SAC` or `DDPG`) with optional checkpoint resume: `src/agents/trainer.py`
- Hard safety supervisor for SoC, temperature, and power bounds: `src/safety/supervisor.py`
- Rule-based baseline controller for benchmarking: `src/controllers/rule_based.py`
- Unified evaluation runner and metrics aggregation: `src/evaluation/runner.py`
- Modbus dispatch adapter stub for hardware link: `src/integration/modbus_interface.py`
- Scripts for quick run, training, validation, splitting, and evaluation:
  - `scripts/run_random_policy.py`
  - `scripts/train_rl.py`
  - `scripts/validate_profiles.py`
  - `scripts/split_profiles.py`
  - `scripts/evaluate_policy.py`
  - `scripts/benchmark_policies.py`
  - `scripts/compare_baseline_vs_rl.py`

## Documentation Map
- Full docs index: `docs/README.md`
- End-to-end pipeline: `docs/pipeline.md`
- Training workflow playbook: `docs/training_workflow.md`
- Mathematical formulation: `docs/math.md`
- Design choices and rationale: `docs/design_choices.md`
- File-by-file reference: `docs/file_reference.md`

## Quick Start (When You Are Ready Later)
1. Add time-series data to `data/raw/profiles.csv` (or update config path).
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Validate data quality and schema:
   - `python -m scripts.validate_profiles --input data/raw/profiles.csv`
4. Split profile into train/val/test:
   - `python -m scripts.split_profiles --input data/raw/profiles.csv --output-dir data/processed`
5. Run environment sanity:
   - `python -m scripts.run_random_policy`
6. Evaluate baseline policy:
   - `python -m scripts.evaluate_policy --policy baseline --episodes 20`
7. Start training:
   - `python -m scripts.train_rl --algo sac --split train`
   - Resume from checkpoint (optional):
     - `python -m scripts.train_rl --algo sac --split train --resume-model-path models/sac_microgrid_agent.zip`
8. Evaluate trained RL policy:
   - `python -m scripts.evaluate_policy --policy sac --model-path models/sac_microgrid_agent.zip --split test --episodes 20`
9. Compare random vs baseline vs RL in one run:
   - `python -m scripts.benchmark_policies --split test --episodes 20 --rl-algo sac --rl-model-path models/sac_microgrid_agent.zip`
10. Generate automated baseline-vs-RL delta report:
   - `python -m scripts.compare_baseline_vs_rl --algo sac --model-path models/sac_microgrid_agent.zip --split test`

## Split-Aware Defaults
- Training CLI defaults to `--split train`.
- Evaluation/benchmark/compare CLIs default to `--split test`.
- To use a custom file directly, pass `--profile-csv path/to/file.csv`.
- Split routes are configured in `configs/data_splits.yaml`.

## Data Schema
Expected CSV columns:
- `timestamp` (optional for training logic, useful for analysis)
- `renewable_kw`
- `load_kw`
- `price_import_per_kwh`
- `price_export_per_kwh` (optional; if missing, derived from `price_import_per_kwh * grid.sell_price_factor`)

Template:
- `data/templates/profiles_template.csv`

## High-Level Architecture
`Profiles -> Environment -> RL Policy -> Safety Supervisor -> Dispatch -> (Simulation/Hardware)`

## Control Interface Note
- Policy action is battery command only (`[battery_kw]`).
- Grid power is auto-balanced by the environment from residual demand.
- Legacy 2D actions (`[battery_kw, grid_kw]`) are accepted for backward compatibility and the second term is ignored.

The equations and exact implementation details are documented in:
- `docs/math.md`
- `docs/pipeline.md`
