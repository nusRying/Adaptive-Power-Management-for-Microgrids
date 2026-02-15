from __future__ import annotations

from src.config import load_microgrid_config
from src.envs.microgrid_env import MicrogridEnv
from src.safety.supervisor import SafetySupervisor


def run_random_episode() -> None:
    config = load_microgrid_config()
    env = MicrogridEnv(config)
    supervisor = SafetySupervisor(config)

    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    overrides = 0

    while not done:
        raw_action = env.action_space.sample()
        decision = supervisor.apply(raw_action, obs)
        if decision.overridden:
            overrides += 1

        obs, reward, terminated, truncated, _ = env.step(decision.action)
        total_reward += reward
        done = terminated or truncated

    print(f"Episode reward: {total_reward:.2f}")
    print(f"Safety overrides: {overrides}")


if __name__ == "__main__":
    run_random_episode()

