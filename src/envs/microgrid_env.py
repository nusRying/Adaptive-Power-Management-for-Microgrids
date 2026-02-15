from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import MicrogridConfig
from src.data.profiles import Profiles, get_profiles


class MicrogridEnv(gym.Env):
    """
    Continuous-control EMS environment.

    Action:
    - action[0]: battery power command in kW (+ discharge, - charge)
    - action[1]: grid power command in kW (+ import, - export)

    Observation:
    - [renew_now, renew_forecast, load_now, load_forecast, soc, temp_c, price_now, price_forecast]
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: MicrogridConfig):
        super().__init__()
        self.cfg = config
        self.horizon = int(self.cfg.environment.episode_horizon)
        self.dt_hours = float(self.cfg.environment.time_step_hours)

        self.action_space = spaces.Box(
            low=np.array(
                [
                    -self.cfg.battery.max_charge_kw,
                    -self.cfg.grid.max_export_kw,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.cfg.battery.max_discharge_kw,
                    self.cfg.grid.max_import_kw,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -20.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1e4, 1e4, 1e4, 1e4, 1.0, 100.0, 10.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._profiles: Profiles | None = None
        self._t = 0
        self._soc = float(self.cfg.battery.soc_init)
        self._temperature_c = float(self.cfg.battery.temperature_c)
        self._last_info: dict[str, Any] = {}

    def _reload_profiles(self, seed: int | None = None) -> None:
        use_seed = self.cfg.environment.seed if seed is None else seed
        self._profiles = get_profiles(
            profile_csv=self.cfg.environment.profile_csv,
            horizon=self.horizon,
            dt_hours=self.dt_hours,
            seed=use_seed,
            sell_price_factor=self.cfg.grid.sell_price_factor,
        )

    def _value_at(self, series: np.ndarray, index: int) -> float:
        safe_idx = int(np.clip(index, 0, self.horizon - 1))
        return float(series[safe_idx])

    def _get_obs(self) -> np.ndarray:
        assert self._profiles is not None
        idx = self._t
        obs = np.array(
            [
                self._value_at(self._profiles.renewable_kw, idx),
                self._value_at(self._profiles.renewable_kw, idx + 1),
                self._value_at(self._profiles.load_kw, idx),
                self._value_at(self._profiles.load_kw, idx + 1),
                self._soc,
                self._temperature_c,
                self._value_at(self._profiles.price_import_per_kwh, idx),
                self._value_at(self._profiles.price_import_per_kwh, idx + 1),
            ],
            dtype=np.float32,
        )
        return obs

    def _apply_battery_constraints(self, cmd_kw: float) -> tuple[float, float]:
        b = self.cfg.battery
        capacity = max(float(b.capacity_kwh), 1e-6)

        if cmd_kw >= 0.0:
            energy_available = max(self._soc - b.soc_min, 0.0) * capacity
            discharge_limit_soc = (energy_available * b.discharge_efficiency) / self.dt_hours
            actual_kw = min(cmd_kw, b.max_discharge_kw, discharge_limit_soc)
            delta_soc = -(actual_kw * self.dt_hours) / (capacity * b.discharge_efficiency)
        else:
            room_available = max(b.soc_max - self._soc, 0.0) * capacity
            charge_limit_soc = room_available / (self.dt_hours * b.charge_efficiency)
            actual_kw = -min(abs(cmd_kw), b.max_charge_kw, charge_limit_soc)
            delta_soc = (-actual_kw * self.dt_hours * b.charge_efficiency) / capacity

        self._soc = float(np.clip(self._soc + delta_soc, b.soc_min, b.soc_max))
        clipped_energy_kwh = abs(cmd_kw - actual_kw) * self.dt_hours
        return float(actual_kw), float(clipped_energy_kwh)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._reload_profiles(seed)
        self._t = 0

        start_soc = self.cfg.battery.soc_init
        if options and "soc_init" in options:
            start_soc = float(options["soc_init"])

        self._soc = float(np.clip(start_soc, self.cfg.battery.soc_min, self.cfg.battery.soc_max))
        self._temperature_c = float(self.cfg.battery.temperature_c)
        self._last_info = {}
        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._profiles is not None

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 2:
            raise ValueError("Action must have exactly 2 values: [battery_kw, grid_kw].")

        idx = self._t
        renewable_kw = self._value_at(self._profiles.renewable_kw, idx)
        load_kw = self._value_at(self._profiles.load_kw, idx)
        price_import = self._value_at(self._profiles.price_import_per_kwh, idx)
        price_export = self._value_at(self._profiles.price_export_per_kwh, idx)

        battery_cmd_kw = float(action[0])
        grid_cmd_kw = float(
            np.clip(action[1], -self.cfg.grid.max_export_kw, self.cfg.grid.max_import_kw)
        )
        battery_kw, clipped_energy_kwh = self._apply_battery_constraints(battery_cmd_kw)

        # Net positive means oversupply, net negative means deficit.
        net_balance_kw = renewable_kw + battery_kw + grid_cmd_kw - load_kw
        unmet_load_kwh = max(0.0, -net_balance_kw) * self.dt_hours
        curtailed_kwh = max(0.0, net_balance_kw) * self.dt_hours

        import_cost = max(0.0, grid_cmd_kw) * price_import * self.dt_hours
        export_revenue = max(0.0, -grid_cmd_kw) * price_export * self.dt_hours
        grid_cost = import_cost - export_revenue

        degradation_cost = (
            abs(battery_kw) * self.dt_hours * self.cfg.battery.degradation_cost_per_kwh
        )
        penalty_cost = (
            unmet_load_kwh * self.cfg.reward.unmet_load_penalty_per_kwh
            + curtailed_kwh * self.cfg.reward.export_curtail_penalty_per_kwh
            + clipped_energy_kwh * self.cfg.reward.unmet_load_penalty_per_kwh * 0.25
        )
        reward = -(grid_cost + degradation_cost + penalty_cost)

        # Simple thermal proxy for future health-based control.
        self._temperature_c = float(
            np.clip(self._temperature_c + 0.01 * abs(battery_kw) - 0.02, 15.0, 60.0)
        )

        self._t += 1
        terminated = self._t >= self.horizon
        truncated = False

        info = {
            "timestep": idx,
            "renewable_kw": renewable_kw,
            "load_kw": load_kw,
            "battery_kw": battery_kw,
            "grid_kw": grid_cmd_kw,
            "soc": self._soc,
            "temperature_c": self._temperature_c,
            "unmet_load_kwh": unmet_load_kwh,
            "curtailed_kwh": curtailed_kwh,
            "cost_grid": grid_cost,
            "cost_degradation": degradation_cost,
            "cost_penalty": penalty_cost,
        }
        self._last_info = info

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        if not self._last_info:
            print("Environment not stepped yet.")
            return
        print(
            "t={timestep} soc={soc:.3f} batt={battery_kw:.1f}kW grid={grid_kw:.1f}kW "
            "load={load_kw:.1f}kW ren={renewable_kw:.1f}kW unmet={unmet_load_kwh:.2f}kWh".format(
                **self._last_info
            )
        )
