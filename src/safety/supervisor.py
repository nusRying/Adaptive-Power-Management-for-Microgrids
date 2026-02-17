from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import MicrogridConfig


@dataclass
class SafetyDecision:
    action: np.ndarray
    overridden: bool
    reason: str


class SafetySupervisor:
    """
    Hard safety layer that clips and blocks unsafe actions before dispatch.
    """

    def __init__(self, config: MicrogridConfig):
        self.cfg = config

    def apply(self, action: np.ndarray, observation: np.ndarray) -> SafetyDecision:
        safe = np.asarray(action, dtype=np.float32).copy()
        original = safe.copy()
        reasons: list[str] = []

        if safe.size not in (1, 2):
            raise ValueError(
                "Expected action shape (1,) -> [battery_kw]. "
                "Legacy (2,) -> [battery_kw, grid_kw] is also supported."
            )
        if observation.size < 6:
            raise ValueError("Observation must include SoC and battery temperature.")

        soc = float(observation[4])
        temp_c = float(observation[5])
        b = self.cfg.battery
        g = self.cfg.grid

        safe[0] = float(np.clip(safe[0], -b.max_charge_kw, b.max_discharge_kw))
        if safe.size == 2:
            safe[1] = float(np.clip(safe[1], -g.max_export_kw, g.max_import_kw))

        # SoC guard rails.
        if soc <= b.soc_min + 0.01 and safe[0] > 0.0:
            safe[0] = 0.0
            reasons.append("blocked_discharge_low_soc")
        if soc >= b.soc_max - 0.01 and safe[0] < 0.0:
            safe[0] = 0.0
            reasons.append("blocked_charge_high_soc")

        # Thermal guard rail.
        if temp_c >= 48.0 and abs(safe[0]) > 0.0:
            safe[0] = 0.0
            reasons.append("blocked_battery_high_temp")

        overridden = not np.allclose(original, safe, atol=1e-6)
        reason = ",".join(reasons) if reasons else "none"
        return SafetyDecision(action=safe, overridden=overridden, reason=reason)
