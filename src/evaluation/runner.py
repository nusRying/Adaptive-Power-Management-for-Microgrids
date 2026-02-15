from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.config import MicrogridConfig
from src.controllers.rule_based import RuleBasedController
from src.envs.microgrid_env import MicrogridEnv
from src.safety.supervisor import SafetySupervisor


PolicyFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class EpisodeMetrics:
    episode: int
    total_reward: float
    grid_cost: float
    degradation_cost: float
    penalty_cost: float
    unmet_load_kwh: float
    curtailed_kwh: float
    import_kwh: float
    export_kwh: float
    battery_throughput_kwh: float
    safety_overrides: int
    steps: int


@dataclass
class EvaluationSummary:
    policy: str
    episodes: int
    avg_reward: float
    avg_grid_cost: float
    avg_degradation_cost: float
    avg_penalty_cost: float
    avg_unmet_load_kwh: float
    avg_curtailed_kwh: float
    avg_import_kwh: float
    avg_export_kwh: float
    avg_battery_throughput_kwh: float
    avg_safety_overrides: float
    details: list[EpisodeMetrics]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["details"] = [asdict(ep) for ep in self.details]
        return data


def random_policy_fn(env: MicrogridEnv) -> PolicyFn:
    def _policy(_: np.ndarray) -> np.ndarray:
        return np.asarray(env.action_space.sample(), dtype=np.float32)

    return _policy


def baseline_policy_fn(config: MicrogridConfig) -> PolicyFn:
    controller = RuleBasedController(config)

    def _policy(obs: np.ndarray) -> np.ndarray:
        return controller.act(obs)

    return _policy


def load_rl_policy_fn(
    algorithm: str,
    model_path: str | Path,
    deterministic: bool = True,
) -> PolicyFn:
    algo = algorithm.lower()
    try:
        from stable_baselines3 import DDPG, SAC
    except ImportError as exc:
        raise RuntimeError(
            "stable-baselines3 is required for RL policy evaluation. Install later with: "
            "pip install -r requirements.txt"
        ) from exc

    model_cls_map = {"sac": SAC, "ddpg": DDPG}
    if algo not in model_cls_map:
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Use one of: sac, ddpg.")

    model_cls = model_cls_map[algo]
    model = model_cls.load(str(model_path))

    def _policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    return _policy


def run_episode(
    env: MicrogridEnv,
    policy_fn: PolicyFn,
    episode_index: int,
    seed: int | None = None,
    use_safety: bool = True,
) -> EpisodeMetrics:
    obs, _ = env.reset(seed=seed)
    supervisor = SafetySupervisor(env.cfg) if use_safety else None

    total_reward = 0.0
    grid_cost = 0.0
    degradation_cost = 0.0
    penalty_cost = 0.0
    unmet_load_kwh = 0.0
    curtailed_kwh = 0.0
    import_kwh = 0.0
    export_kwh = 0.0
    battery_throughput_kwh = 0.0
    safety_overrides = 0
    steps = 0
    done = False

    while not done:
        action = policy_fn(obs)
        if supervisor is not None:
            safe_decision = supervisor.apply(action, obs)
            action = safe_decision.action
            if safe_decision.overridden:
                safety_overrides += 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        total_reward += float(reward)

        grid_cost += float(info.get("cost_grid", 0.0))
        degradation_cost += float(info.get("cost_degradation", 0.0))
        penalty_cost += float(info.get("cost_penalty", 0.0))
        unmet_load_kwh += float(info.get("unmet_load_kwh", 0.0))
        curtailed_kwh += float(info.get("curtailed_kwh", 0.0))
        grid_kw = float(info.get("grid_kw", 0.0))
        batt_kw = float(info.get("battery_kw", 0.0))
        import_kwh += max(0.0, grid_kw) * env.dt_hours
        export_kwh += max(0.0, -grid_kw) * env.dt_hours
        battery_throughput_kwh += abs(batt_kw) * env.dt_hours

    return EpisodeMetrics(
        episode=episode_index,
        total_reward=total_reward,
        grid_cost=grid_cost,
        degradation_cost=degradation_cost,
        penalty_cost=penalty_cost,
        unmet_load_kwh=unmet_load_kwh,
        curtailed_kwh=curtailed_kwh,
        import_kwh=import_kwh,
        export_kwh=export_kwh,
        battery_throughput_kwh=battery_throughput_kwh,
        safety_overrides=safety_overrides,
        steps=steps,
    )


def evaluate_policy(
    env: MicrogridEnv,
    policy_fn: PolicyFn,
    policy_name: str,
    episodes: int = 10,
    seed_start: int = 0,
    use_safety: bool = True,
) -> EvaluationSummary:
    if episodes <= 0:
        raise ValueError("episodes must be greater than 0.")

    details: list[EpisodeMetrics] = []
    for episode in range(episodes):
        seed = seed_start + episode
        metrics = run_episode(
            env=env,
            policy_fn=policy_fn,
            episode_index=episode,
            seed=seed,
            use_safety=use_safety,
        )
        details.append(metrics)

    def _avg(values: list[float]) -> float:
        return float(np.mean(np.asarray(values, dtype=np.float64)))

    return EvaluationSummary(
        policy=policy_name,
        episodes=episodes,
        avg_reward=_avg([m.total_reward for m in details]),
        avg_grid_cost=_avg([m.grid_cost for m in details]),
        avg_degradation_cost=_avg([m.degradation_cost for m in details]),
        avg_penalty_cost=_avg([m.penalty_cost for m in details]),
        avg_unmet_load_kwh=_avg([m.unmet_load_kwh for m in details]),
        avg_curtailed_kwh=_avg([m.curtailed_kwh for m in details]),
        avg_import_kwh=_avg([m.import_kwh for m in details]),
        avg_export_kwh=_avg([m.export_kwh for m in details]),
        avg_battery_throughput_kwh=_avg([m.battery_throughput_kwh for m in details]),
        avg_safety_overrides=_avg([float(m.safety_overrides) for m in details]),
        details=details,
    )

