from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.config import (
    load_microgrid_config,
    resolve_profile_csv_override,
    load_training_config,
)
from src.envs.microgrid_env import MicrogridEnv


def _import_rl_dependencies() -> tuple[dict[str, Any], Any]:
    try:
        from stable_baselines3 import DDPG, SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise RuntimeError(
            "RL dependencies are missing. Install later with: pip install -r requirements.txt"
        ) from exc
    return {"sac": SAC, "ddpg": DDPG}, Monitor


def train(
    microgrid_config_path: str | Path = "configs/microgrid.yaml",
    training_config_path: str | Path = "configs/training.yaml",
    override_algo: str | None = None,
    profile_csv_override: str | None = None,
    resume_model_path: str | Path | None = None,
) -> Path:
    algo_map, monitor_cls = _import_rl_dependencies()

    microgrid_cfg = load_microgrid_config(microgrid_config_path)
    if profile_csv_override:
        microgrid_cfg.environment.profile_csv = profile_csv_override
    train_cfg = load_training_config(training_config_path)

    algo_name = (override_algo or train_cfg.algorithm).lower()
    if algo_name not in algo_map:
        supported = ", ".join(sorted(algo_map))
        raise ValueError(f"Unsupported algorithm '{algo_name}'. Supported: {supported}")

    env = monitor_cls(MicrogridEnv(microgrid_cfg))

    model_cls = algo_map[algo_name]
    if resume_model_path:
        resume_path = Path(resume_model_path)
        if not resume_path.exists():
            raise ValueError(f"Resume model path not found: {resume_path}")
        model = model_cls.load(
            str(resume_path),
            env=env,
            tensorboard_log=train_cfg.tensorboard_log,
            verbose=1,
        )
        reset_num_timesteps = False
    else:
        model = model_cls(
            policy="MlpPolicy",
            env=env,
            learning_rate=train_cfg.learning_rate,
            batch_size=train_cfg.batch_size,
            buffer_size=train_cfg.buffer_size,
            gamma=train_cfg.gamma,
            tau=train_cfg.tau,
            tensorboard_log=train_cfg.tensorboard_log,
            verbose=1,
        )
        reset_num_timesteps = True

    model.learn(
        total_timesteps=train_cfg.total_timesteps,
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
    )

    model_dir = Path(train_cfg.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{algo_name}_microgrid_agent"
    model.save(str(model_path))
    return model_path.with_suffix(".zip")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train RL agent for microgrid EMS.")
    parser.add_argument(
        "--microgrid-config",
        default="configs/microgrid.yaml",
        help="Path to microgrid environment config YAML.",
    )
    parser.add_argument(
        "--training-config",
        default="configs/training.yaml",
        help="Path to training config YAML.",
    )
    parser.add_argument(
        "--algo",
        default=None,
        choices=["sac", "ddpg"],
        help="Optional override for algorithm in training config.",
    )
    parser.add_argument(
        "--data-splits-config",
        default="configs/data_splits.yaml",
        help="Path to data split config YAML.",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test", "none"],
        help="Profile split to use when --profile-csv is not provided.",
    )
    parser.add_argument(
        "--profile-csv",
        default=None,
        help="Explicit profile CSV path override (takes precedence over --split).",
    )
    parser.add_argument(
        "--resume-model-path",
        default=None,
        help=(
            "Optional checkpoint path to continue training from. "
            "When set, timestep counter is continued (no reset)."
        ),
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    profile_csv_override = resolve_profile_csv_override(
        profile_csv=args.profile_csv,
        split=args.split,
        data_splits_config_path=args.data_splits_config,
    )

    model_path = train(
        microgrid_config_path=args.microgrid_config,
        training_config_path=args.training_config,
        override_algo=args.algo,
        profile_csv_override=profile_csv_override,
        resume_model_path=args.resume_model_path,
    )
    print(f"Training complete. Model saved to: {model_path}")


if __name__ == "__main__":
    main()
