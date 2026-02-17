# Training Workflow

This document captures the current recommended workflow for RL iteration.

## 1. Environment Setup
Use the project environment and isolate from user site-packages:

```powershell
conda activate comp_vision
$env:PYTHONPATH = "."
$env:PYTHONNOUSERSITE = "1"
```

## 2. Data Preparation
Validate and split profile data before training:

```powershell
python -m scripts.validate_profiles --input data/raw/profiles.csv
python -m scripts.split_profiles --input data/raw/profiles.csv --output-dir data/processed
```

If `data/raw/profiles.csv` is missing, create one from synthetic profiles or place your real dataset there.

## 3. Baseline Reference
Always keep a baseline reference on the same split:

```powershell
python -m scripts.evaluate_policy --policy baseline --microgrid-config configs/microgrid.tuned.yaml --split val --episodes 10
```

## 4. RL Training (Battery-Only Action Interface)
Current recommended training config:
- `configs/training.battery_only_20k.yaml`

First 20k block:

```powershell
python -m scripts.train_rl --microgrid-config configs/microgrid.tuned.yaml --training-config configs/training.battery_only_20k.yaml --algo sac --split train
```

Continue another 20k block:

```powershell
python -m scripts.train_rl --microgrid-config configs/microgrid.tuned.yaml --training-config configs/training.battery_only_20k.yaml --algo sac --split train --resume-model-path models/battery_only_sac/sac_microgrid_agent.zip
```

## 5. Evaluate and Compare
After each block, compare against baseline on validation split:

```powershell
python -m scripts.compare_baseline_vs_rl --algo sac --model-path models/battery_only_sac/sac_microgrid_agent.zip --microgrid-config configs/microgrid.tuned.yaml --split val --episodes 10 --json-out reports/baseline_vs_sac_val_latest.json --markdown-out reports/baseline_vs_sac_val_latest.md
```

## 6. Promotion Rule
Promote model to final test only when validation comparison is consistently better than baseline for:
- `avg_reward` (higher),
- `avg_grid_cost` (lower),
- `avg_penalty_cost` (lower),
- `avg_safety_overrides` (not worse).

Then run final test comparison:

```powershell
python -m scripts.compare_baseline_vs_rl --algo sac --model-path models/battery_only_sac/sac_microgrid_agent.zip --microgrid-config configs/microgrid.tuned.yaml --split test --episodes 20 --json-out reports/baseline_vs_sac_test_final.json --markdown-out reports/baseline_vs_sac_test_final.md
```

## 7. If Training Plateaus
If RL remains close but below baseline after multiple blocks (for example 60k-80k):
1. Increase `degradation_cost_per_kwh` slightly in `configs/microgrid.tuned.yaml`.
2. Retrain from scratch (do not resume from old checkpoint).
3. Repeat block-wise comparison on validation split.
