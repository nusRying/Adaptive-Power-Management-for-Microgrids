# Mathematical Formulation

This document formalizes the current implementation in `src/envs/microgrid_env.py` and related modules.

## 1. Notation
- Time index: `t = 0, 1, ..., T-1`
- Step duration: `Delta t` (hours), from `environment.time_step_hours`
- Renewable power: `P_ren(t)` [kW]
- Load power: `P_load(t)` [kW]
- Battery command from policy: `u_b(t)` [kW]
- Grid command from policy: `u_g(t)` [kW]
- Realized battery power after constraints: `P_b(t)` [kW]
  - `P_b > 0`: discharging battery to serve bus
  - `P_b < 0`: charging battery from bus
- Realized grid power after clipping: `P_g(t)` [kW]
  - `P_g > 0`: importing from utility
  - `P_g < 0`: exporting to utility
- State of charge: `SOC(t)` in `[SOC_min, SOC_max]`
- Import tariff: `c_imp(t)` [currency/kWh]
- Export tariff: `c_exp(t)` [currency/kWh]
  - from `price_export_per_kwh` when available, else derived as `sell_price_factor * c_imp(t)`

## 2. MDP Definition
State vector in code (`obs`):
`s_t = [P_ren(t), P_ren(t+1), P_load(t), P_load(t+1), SOC(t), Temp(t), c_imp(t), c_imp(t+1)]`

Action vector in code:
`a_t = [u_b(t), u_g(t)]`

Transition:
- Deterministic given action and profile sequence, except stochasticity from profile generation/selection.

Reward:
- Negative operating cost with penalty terms:
`r_t = -(C_grid(t) + C_deg(t) + C_penalty(t))`

## 3. Action Bounds
Before environment physics:
- `u_b(t) in [-P_ch_max, P_dis_max]`
- `u_g(t) in [-P_exp_max, P_imp_max]`

where:
- `P_ch_max = battery.max_charge_kw`
- `P_dis_max = battery.max_discharge_kw`
- `P_imp_max = grid.max_import_kw`
- `P_exp_max = grid.max_export_kw`

## 4. Battery Feasibility and SoC Dynamics

### 4.1 Discharge branch (`u_b >= 0`)
Available internal energy above lower SoC:
`E_avail(t) = max(SOC(t) - SOC_min, 0) * E_cap`

Maximum feasible discharge power from SoC limit:
`P_dis_soc_max(t) = E_avail(t) * eta_dis / Delta t`

Applied discharge power:
`P_b(t) = min(u_b(t), P_dis_max, P_dis_soc_max(t))`

SoC update:
`SOC(t+1) = SOC(t) - (P_b(t) * Delta t) / (E_cap * eta_dis)`

### 4.2 Charge branch (`u_b < 0`)
Available room below upper SoC:
`E_room(t) = max(SOC_max - SOC(t), 0) * E_cap`

Maximum feasible charge power from SoC limit:
`P_ch_soc_max(t) = E_room(t) / (Delta t * eta_ch)`

Applied charge power (negative sign):
`P_b(t) = -min(|u_b(t)|, P_ch_max, P_ch_soc_max(t))`

SoC update:
`SOC(t+1) = SOC(t) + (|P_b(t)| * Delta t * eta_ch) / E_cap`

### 4.3 Clipping mismatch term
Energy not honored from requested battery command:
`E_clip(t) = |u_b(t) - P_b(t)| * Delta t`

This term is penalized in reward to discourage infeasible commands.

## 5. Power Balance
Bus net balance:
`P_bal(t) = P_ren(t) + P_b(t) + P_g(t) - P_load(t)`

Deficit energy (unserved load proxy):
`E_unmet(t) = max(0, -P_bal(t)) * Delta t`

Surplus energy (curtailment/export oversupply proxy):
`E_curt(t) = max(0, P_bal(t)) * Delta t`

## 6. Cost Terms

### 6.1 Grid transaction cost
Import cost:
`C_imp(t) = max(0, P_g(t)) * c_imp(t) * Delta t`

Export revenue:
`R_exp(t) = max(0, -P_g(t)) * c_exp(t) * Delta t`

Net grid cost:
`C_grid(t) = C_imp(t) - R_exp(t)`

### 6.2 Battery degradation proxy
`C_deg(t) = |P_b(t)| * Delta t * c_deg`

where `c_deg = battery.degradation_cost_per_kwh`.

### 6.3 Penalty block
`C_penalty(t) = lambda_unmet * E_unmet(t) + lambda_curt * E_curt(t) + 0.25 * lambda_unmet * E_clip(t)`

where:
- `lambda_unmet = reward.unmet_load_penalty_per_kwh`
- `lambda_curt = reward.export_curtail_penalty_per_kwh`

### 6.4 Reward
`r_t = -(C_grid(t) + C_deg(t) + C_penalty(t))`

The optimization target is maximizing expected discounted return:
`J(pi) = E_pi [sum_{t=0}^{T-1} gamma^t * r_t]`

## 7. Temperature Proxy Dynamics
Current implementation uses a simple thermal proxy:
`Temp(t+1) = clip(Temp(t) + 0.01 * |P_b(t)| - 0.02, 15, 60)`

This is not a physics-grade thermal model; it is a control-relevant stress indicator for safety logic.

## 8. Synthetic Profile Equations
If real CSV data is not present, profiles are generated:

Solar-like envelope:
`S(t) = max(0, sin((hour(t)-6)/12 * pi))`

Wind-like component:
`W(t) = clip(0.45 + 0.20 * sin((hour(t)+3) * 2pi/24) + epsilon_w, 0, 1)`

Renewable power:
`P_ren(t) = clip(80*S(t) + 40*W(t), 0, +inf)`

Load profile:
`P_load(t) = clip(110 + 18*sin((hour(t)-17)*2pi/24) + epsilon_l, 60, +inf)`

Import tariff:
`c_imp(t) = clip(0.10 + 0.08*I_evening(t) + epsilon_p, 0.05, +inf)`

Export tariff:
`c_exp(t) = 0.75 * c_imp(t)`

## 9. RL Algorithm Math (SAC and DDPG)
Training API is in `src/agents/trainer.py` and uses Stable-Baselines3 implementations.

### 9.1 Why actor-critic continuous control
Action is continuous and bounded in 2D. Policy gradient actor-critic methods are standard for this regime.

### 9.2 SAC objective (conceptual)
SAC maximizes:
`E[sum gamma^t * (r_t + alpha * H(pi(.|s_t)))]`

where entropy term `H` encourages exploration and robustness under stochastic profiles.

### 9.3 DDPG objective (conceptual)
DDPG uses deterministic policy `mu_theta(s)` and critic `Q_phi(s,a)`:
- actor update through deterministic policy gradient on `Q`,
- critic update by TD error to Bellman target.

### 9.4 Discounting
`gamma` from config (default `0.99`) weights long-term economics and battery health impact.

## 10. Safety Layer Math
Safety supervisor (`src/safety/supervisor.py`) applies:
- hard clipping to power limits,
- SoC threshold rules:
  - if `SOC <= SOC_min + 0.01`, forbid discharge,
  - if `SOC >= SOC_max - 0.01`, forbid charge,
- thermal rule:
  - if `Temp >= 48`, force `P_b = 0`.

This defines a projection:
`a_safe = Pi_safe(a_raw, s_t)`

Where `Pi_safe` is deterministic and non-learned.

## 11. Mapping Math to Code
- State assembly: `MicrogridEnv._get_obs`
- Battery equations: `MicrogridEnv._apply_battery_constraints`
- Balance and reward: `MicrogridEnv.step`
- Thermal update: `MicrogridEnv.step`
- Safety projection: `SafetySupervisor.apply`
- RL optimization wrapper: `train()` in `src/agents/trainer.py`
