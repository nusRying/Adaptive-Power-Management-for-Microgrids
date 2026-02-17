from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import MicrogridConfig


@dataclass
class RuleBasedPolicyConfig:
    low_price_threshold: float = 0.11
    high_price_threshold: float = 0.16
    reserve_soc: float = 0.20
    target_soc: float = 0.70
    high_soc_discharge_bias: float = 0.75
    low_price_charge_fraction: float = 0.5


class RuleBasedController:
    """
    Heuristic baseline for cost-aware dispatch.

    Returns battery command only. Grid is auto-balanced by the environment.
    """

    def __init__(
        self,
        microgrid_config: MicrogridConfig,
        policy_config: RuleBasedPolicyConfig | None = None,
    ):
        self.cfg = microgrid_config
        self.policy = policy_config or RuleBasedPolicyConfig()

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32).reshape(-1)
        if obs.size < 8:
            raise ValueError("Observation must have at least 8 features.")

        renewable_now = float(obs[0])
        load_now = float(obs[2])
        soc = float(obs[4])
        price_now = float(obs[6])

        b = self.cfg.battery
        p = self.policy

        net_excess = max(0.0, renewable_now - load_now)
        net_deficit = max(0.0, load_now - renewable_now)

        battery_cmd_kw = 0.0

        if net_excess > 0.0 and soc < (b.soc_max - 0.01):
            # Absorb renewable surplus first.
            battery_cmd_kw = -min(b.max_charge_kw, net_excess)
        elif net_deficit > 0.0:
            reserve_floor = max(b.soc_min + 0.01, p.reserve_soc)
            if soc > reserve_floor and price_now >= p.high_price_threshold:
                # High tariff period: discharge to offset imports.
                battery_cmd_kw = min(b.max_discharge_kw, net_deficit)
            elif soc > p.high_soc_discharge_bias:
                battery_cmd_kw = min(b.max_discharge_kw, 0.5 * net_deficit)
        elif price_now <= p.low_price_threshold and soc < min(p.target_soc, b.soc_max - 0.01):
            # Opportunistic charging in low-price periods.
            battery_cmd_kw = -min(b.max_charge_kw * p.low_price_charge_fraction, g.max_import_kw)

        return np.array([battery_cmd_kw], dtype=np.float32)
