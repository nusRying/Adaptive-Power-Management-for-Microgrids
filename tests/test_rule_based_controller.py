import numpy as np

from src.config import MicrogridConfig
from src.controllers.rule_based import RuleBasedController


def test_rule_based_controller_action_shape_and_bounds() -> None:
    cfg = MicrogridConfig()
    controller = RuleBasedController(cfg)

    obs = np.array([20.0, 20.0, 80.0, 80.0, 0.8, 30.0, 0.2, 0.2], dtype=np.float32)
    action = controller.act(obs)

    assert action.shape == (1,)
    assert -cfg.battery.max_charge_kw <= float(action[0]) <= cfg.battery.max_discharge_kw


def test_rule_based_controller_charges_on_excess_renewable() -> None:
    cfg = MicrogridConfig()
    controller = RuleBasedController(cfg)

    obs = np.array([120.0, 100.0, 60.0, 65.0, 0.5, 30.0, 0.09, 0.09], dtype=np.float32)
    action = controller.act(obs)

    assert float(action[0]) < 0.0
