from src.config import MicrogridConfig
from src.envs.microgrid_env import MicrogridEnv


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

