# End-to-End Pipeline

## 1. Objective
Control a grid-tied microgrid in discrete time to minimize operating cost while respecting battery and grid constraints.

At each time step the controller decides:
- battery charge/discharge power,
- grid import/export power.

The pipeline is implemented so an RL policy can learn this control by interaction.

## 2. System Boundary
Physical and market variables considered:
- Renewable generation (`renewable_kw`)
- Load demand (`load_kw`)
- BESS state of charge (`soc`)
- BESS temperature proxy (`temperature_c`)
- Grid tariffs (`price_import_per_kwh`, `price_export_per_kwh`)
- Grid transfer limits (`max_import_kw`, `max_export_kw`)

## 3. Runtime Flow
Primary code path:
- `scripts/run_random_policy.py`, `scripts/train_rl.py`, `scripts/evaluate_policy.py`, `scripts/compare_baseline_vs_rl.py`
- `src/main.py` or `src/agents/trainer.py`
- `src/envs/microgrid_env.py`
- `src/safety/supervisor.py` (in inference loop example)

Control loop per step:
1. Read current observation from environment.
2. RL policy (or random policy for sanity check) outputs continuous action.
3. Safety supervisor clips or blocks unsafe action.
4. Environment applies battery and grid constraints.
5. Environment computes power balance, cost terms, reward, next state.
6. RL trainer stores transition and updates policy/value networks.

## 4. Data Pipeline
Implemented in `src/data/profiles.py`, `src/data/validation.py`, `src/data/splitting.py`.

Selection logic:
1. If `environment.profile_csv` exists, load real profile data.
2. Otherwise generate synthetic data from deterministic/seasonal shapes plus noise.

Recommended preprocessing workflow:
1. Validate raw CSV:
   - `python -m scripts.validate_profiles --input data/raw/profiles.csv`
2. Split chronologically:
   - `python -m scripts.split_profiles --input data/raw/profiles.csv --output-dir data/processed`
3. Use `configs/data_splits.yaml` to route train/val/test file paths.

CSV mode:
- Required columns:
  - `renewable_kw`
  - `load_kw`
  - `price_import_per_kwh`
- Optional columns:
  - `price_export_per_kwh` (derived from `price_import_per_kwh * sell_price_factor` when missing)
  - `timestamp` (used for continuity checks and chronological splits)
- If profile is shorter than episode horizon, series is tiled to match horizon.

Synthetic fallback mode:
- Solar-like daily curve + wind-like oscillation + noise for renewable.
- Daily demand shape + noise for load.
- Time-of-use-style evening uplift for import tariff.
- Export tariff modeled as fraction of import tariff.

Purpose of synthetic mode:
- keep development unblocked before real data is available,
- allow reproducible tests using fixed seeds.

## 5. Environment Lifecycle
Implemented in `src/envs/microgrid_env.py`.

### 5.1 Reset
On `reset()`:
1. Profiles are loaded/generated for configured horizon.
2. Time index `t` is set to 0.
3. SoC is initialized from config (or optional override).
4. Temperature is reset.
5. Initial observation is returned.

### 5.2 Step
On `step(action)`:
1. Parse action as `[battery_kw, grid_kw]`.
2. Clip grid action to import/export limits.
3. Apply battery limits:
   - power limits,
   - SoC limits through feasible energy in current interval,
   - charge/discharge efficiencies.
4. Compute net power balance.
5. Convert imbalance into unmet-load and curtailment terms.
6. Compute grid cost, degradation cost, and penalties.
7. Compute reward as negative total cost.
8. Update temperature proxy.
9. Advance time and return next observation and diagnostics.

## 6. Training Pipeline
Implemented in `src/agents/trainer.py`.

1. Load `MicrogridConfig` and `TrainingConfig`.
2. Resolve profile CSV from:
   - explicit `--profile-csv`, else
   - selected `--split` from `configs/data_splits.yaml`.
3. Build `MicrogridEnv`.
4. Wrap with `stable_baselines3` `Monitor`.
5. Create chosen algorithm:
   - `SAC` (default),
   - `DDPG`.
6. Train for `total_timesteps`.
7. Save model to `model_dir`.

Output artifact:
- `models/sac_microgrid_agent` or `models/ddpg_microgrid_agent`

## 6.1 Baseline and RL Evaluation
Implemented in `src/controllers/rule_based.py`, `src/evaluation/runner.py`, `src/evaluation/comparison.py`, `scripts/evaluate_policy.py`, `scripts/benchmark_policies.py`, and `scripts/compare_baseline_vs_rl.py`.

Supported evaluated policies:
- `baseline`: heuristic controller for benchmark reference.
- `random`: random action sampler for lower-bound behavior.
- `sac` / `ddpg`: trained RL model loaded from disk.

Evaluation metrics:
- average episode reward,
- grid/degradation/penalty cost decomposition,
- unmet load and curtailment energy,
- import/export energy,
- battery throughput,
- safety override count.

Automated comparison report:
- `scripts/compare_baseline_vs_rl.py` evaluates baseline and RL on the same split and seeds.
- It writes:
  - JSON report with full summaries and deltas.
  - Markdown report with metric-by-metric improvements.

## 7. Safety Pipeline
Implemented in `src/safety/supervisor.py`.

Input:
- action from controller,
- observation containing `soc` and `temperature_c`.

Checks:
1. Power clipping for battery and grid commands.
2. Block discharge near minimum SoC.
3. Block charge near maximum SoC.
4. Block any battery action at high temperature threshold.

Output:
- safe action,
- override flag,
- reason string.

This provides a deterministic guard layer above an RL controller.

## 8. Integration Pipeline (Toward Hardware)
Implemented as adapter scaffold in `src/integration/modbus_interface.py`.

Workflow:
1. Open Modbus TCP connection.
2. Convert setpoints to register values.
3. Write battery/grid setpoints to configured registers.
4. Close connection.

Current status:
- API stub exists.
- Full production integration (timeouts, retries, scaling, alarms, handshake) is not yet implemented.

## 9. Validation Pipeline
Current checks:
- Syntax compile check (`python -m compileall ...`)
- Smoke test for environment (`tests/test_env_smoke.py`)

Recommended next validation layers:
1. Unit tests for reward decomposition and SoC invariants.
2. Regression tests for episode-level cost under fixed seeds.
3. Closed-loop stress tests with randomized scenarios.
4. Hardware-in-the-loop safety validation before field dispatch.

## 10. Implemented vs Planned
Implemented now:
- single-agent RL environment and training path,
- cost-based reward shaping,
- safety supervisor,
- Modbus adapter skeleton,
- baseline benchmarking and metrics pipeline,
- data validation and deterministic splits.

Planned next:
- historical-data pretraining workflow,
- domain randomization for robustness,
- optional multi-agent decomposition (battery/inverter/grid agents),
- tighter coupling with power flow solvers (`pandapower`) or `pymgrid`.
