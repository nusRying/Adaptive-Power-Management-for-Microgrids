from src.config import MicrogridConfig
from src.envs.microgrid_env import MicrogridEnv
import numpy as np


def test_microgrid_env_smoke() -> None:
    env = MicrogridEnv(MicrogridConfig())
    obs, _ = env.reset(seed=7)

    assert obs.shape == (8,)

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        assert obs.shape == (8,)
        assert isinstance(float(reward), float)
        if terminated or truncated:
            break


def test_microgrid_env_accepts_legacy_two_dim_action() -> None:
    env = MicrogridEnv(MicrogridConfig())
    env.reset(seed=11)

    obs, reward, terminated, truncated, _ = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert obs.shape == (8,)
    assert isinstance(float(reward), float)
    assert isinstance(bool(terminated), bool)
    assert isinstance(bool(truncated), bool)
