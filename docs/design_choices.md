# Design Choices and Rationale

This document explains why each major modeling and implementation decision was made for this scaffold.

## 1. Why a Custom Gymnasium Environment First
Choice:
- Start with `Gymnasium`-compatible custom environment (`src/envs/microgrid_env.py`).

Reason:
- Fast iteration on state/action/reward design.
- Transparent physics assumptions and reward decomposition.
- Easy integration with Stable-Baselines3 algorithms.

Tradeoff:
- Lower physical fidelity than full AC power-flow co-simulation.
- Mitigation path: connect with `pandapower` or `pymgrid` once baseline policy behavior is stable.

## 2. Why Continuous Action Space
Choice:
- Battery setpoint is continuous, not discrete.
- Grid setpoint is auto-balanced by the environment from residual demand.

Reason:
- Power electronics setpoints are inherently continuous.
- Discretization introduces quantization error and larger action spaces.
- Battery-only action reduces policy search space and improves training stability.

Tradeoff:
- Less direct policy control over explicit import/export decisions.
- Addressed by deterministic residual grid balancing and still using actor-critic algorithms for battery control.

## 3. Why This State Vector
Current observation:
- current and one-step-ahead renewable,
- current and one-step-ahead load,
- SoC,
- battery temperature proxy,
- current and one-step-ahead import price.

Reason:
- Includes immediate physics state (SoC, temperature).
- Includes short-horizon preview to reduce myopic behavior.
- Includes market signal to support price arbitrage decisions.

Tradeoff:
- Only one-step forecast; longer forecast horizon could improve planning.
- Planned extension: add multi-step forecast features and uncertainty estimates.

## 4. Why This Reward Decomposition
Choice:
- Total reward equals negative sum of:
  - grid transaction cost,
  - degradation proxy cost,
  - reliability/curtailment penalties.

Reason:
- Converts EMS business objective to scalar RL objective.
- Encourages reliability first (high unmet-load penalty), then economics and battery health.

Tradeoff:
- Reward weight tuning is sensitive.
- Recommended practice: scale weights so unmet load remains dominant while avoiding unstable gradients.

## 5. Why Penalize Clipped Battery Commands
Choice:
- Penalize infeasible battery command portion (`E_clip`) in reward.

Reason:
- Helps policy learn feasible command distribution faster.
- Reduces repeated requests that violate SoC constraints.

Tradeoff:
- Over-penalizing can make policy overly conservative.

## 6. Why Include a Hard Safety Supervisor
Choice:
- Deterministic rule-based safety layer outside RL policy.

Reason:
- Prevents unsafe battery operations even if policy output is bad.
- Enables safer transition from simulation to hardware-connected operation.

Tradeoff:
- Introduces non-smooth projection that policy must adapt to.
- In deployment this is acceptable because safety takes precedence.

## 7. Why SAC as Default and DDPG as Optional
Choice:
- Default `SAC`, optional `DDPG`.

Reason:
- SAC is generally more stable in stochastic environments and handles exploration via entropy regularization.
- DDPG remains useful as a deterministic baseline.

Tradeoff:
- SAC can require more compute.
- DDPG can be more brittle under noise and non-stationary profiles.

## 8. Why Synthetic Profiles Fallback
Choice:
- If no CSV exists, generate synthetic renewable/load/price profiles.

Reason:
- Enables immediate development and testing without waiting for real datasets.
- Guarantees reproducibility with fixed random seed.

Tradeoff:
- Synthetic distributions may not match real grid conditions.
- Planned next step: replace with historical profiles and calibrated noise models.

## 9. Why Config-Driven Parameters
Choice:
- Parameters live in YAML config files.

Reason:
- Keeps code stable while enabling scenario sweeps.
- Simplifies experiments and reproducibility.

Tradeoff:
- Bad config values can produce unrealistic behavior.
- Add future config validation tests and boundary checks.

## 10. Why a Modbus Adapter Stub
Choice:
- Provide `ModbusDispatcher` scaffold now.

Reason:
- Defines the software boundary for physical integration early.
- Makes control-plane design explicit (setpoint write interface).

Tradeoff:
- Production-grade communication features are not yet implemented.
- Future work: retries, watchdogs, deadband scaling, signed register mapping, and confirmation reads.

## 11. Why This Time Discretization
Choice:
- Default `Delta t = 0.25` hours (15 minutes), horizon `T = 96` steps (24 hours).

Reason:
- Natural alignment with common energy market intervals and EMS scheduling granularity.
- Reasonable balance between resolution and training cost.

Tradeoff:
- Does not capture second-level converter dynamics.
- If needed, reduce `Delta t` with adjusted degradation and penalty scales.

## 12. Why This Initial Scope (Single-Agent)
Choice:
- Single agent controls battery setpoint only; grid flow is solved by residual balance.

Reason:
- Lower complexity and faster convergence for initial baseline.
- Easier debugging of reward and constraints.
- Avoids degenerate co-optimization where policy issues conflicting battery/grid commands.

Tradeoff:
- Does not model decentralized coordination.
- Planned extension: multi-agent formulation (battery, inverter, grid actor roles).

## 13. What Is Intentionally Deferred
Not yet implemented:
- Offline pretraining pipeline on historical buffers.
- Domain randomization configuration layer.
- Full `pandapower` network coupling for voltage/reactive constraints.
- Multi-objective Pareto analysis.
- HIL validation harness.

Reason for deferment:
- Start with a coherent RL core before introducing high-fidelity and operational complexity.

## 14. Why Add a Rule-Based Baseline
Choice:
- Add a deterministic heuristic controller (`src/controllers/rule_based.py`) and evaluate it with the same metrics as RL.

Reason:
- RL results are not meaningful without a baseline reference.
- Gives immediate benchmark when little real data is available.

Tradeoff:
- Heuristic controller is not globally optimal.
- Still valuable as a sanity floor and regression benchmark.

## 15. Why Add Data Validation and Chronological Split Tools
Choice:
- Add validation and split scripts before heavy RL training.

Reason:
- Prevents silent issues from malformed timestamps, invalid numeric values, and schema drift.
- Ensures reproducible train/val/test comparisons with time-order preserved.

Tradeoff:
- Adds preprocessing step overhead.
- Strongly reduces wasted training cycles caused by bad input data.

## 16. Why Add Automated Baseline-vs-RL Delta Reports
Choice:
- Add a dedicated comparison script that runs baseline and RL on the same seeds and split.

Reason:
- Prevents misleading comparisons caused by inconsistent evaluation conditions.
- Produces objective metric deltas and improvement percentages for iteration tracking.

Tradeoff:
- Adds another reporting artifact to maintain.
- Benefit is higher reproducibility and faster experiment review.
