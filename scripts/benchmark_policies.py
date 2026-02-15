from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_microgrid_config, resolve_profile_csv_override
from src.envs.microgrid_env import MicrogridEnv
from src.evaluation.runner import (
    baseline_policy_fn,
    evaluate_policy,
    load_rl_policy_fn,
    random_policy_fn,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark random, baseline, and optional RL policy on same environment."
    )
    parser.add_argument("--microgrid-config", default="configs/microgrid.yaml")
    parser.add_argument("--data-splits-config", default="configs/data_splits.yaml")
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "none"],
        help="Profile split to benchmark when --profile-csv is not provided.",
    )
    parser.add_argument(
        "--profile-csv",
        default=None,
        help="Explicit profile CSV path override (takes precedence over --split).",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=1500)
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument(
        "--rl-algo",
        choices=["sac", "ddpg"],
        default=None,
        help="Optional RL algorithm for third benchmark entry.",
    )
    parser.add_argument(
        "--rl-model-path",
        default=None,
        help="Path to RL model zip file. Required when --rl-algo is set.",
    )
    return parser


def _print_result(name: str, summary) -> None:
    print(
        f"{name:>10} | reward={summary.avg_reward:10.3f} | "
        f"grid={summary.avg_grid_cost:9.3f} | penalty={summary.avg_penalty_cost:9.3f} | "
        f"unmet={summary.avg_unmet_load_kwh:8.3f} | overrides={summary.avg_safety_overrides:6.2f}"
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = load_microgrid_config(args.microgrid_config)
    profile_csv_override = resolve_profile_csv_override(
        profile_csv=args.profile_csv,
        split=args.split,
        data_splits_config_path=args.data_splits_config,
    )
    if profile_csv_override:
        cfg.environment.profile_csv = profile_csv_override
    use_safety = not args.no_safety

    random_env = MicrogridEnv(cfg)
    baseline_env = MicrogridEnv(cfg)

    random_summary = evaluate_policy(
        env=random_env,
        policy_fn=random_policy_fn(random_env),
        policy_name="random",
        episodes=args.episodes,
        seed_start=args.seed_start,
        use_safety=use_safety,
    )
    baseline_summary = evaluate_policy(
        env=baseline_env,
        policy_fn=baseline_policy_fn(cfg),
        policy_name="baseline",
        episodes=args.episodes,
        seed_start=args.seed_start,
        use_safety=use_safety,
    )

    print("Benchmark results:")
    _print_result("random", random_summary)
    _print_result("baseline", baseline_summary)

    if args.rl_algo or args.rl_model_path:
        if not args.rl_algo or not args.rl_model_path:
            raise SystemExit("--rl-algo and --rl-model-path must be provided together.")
        model_path = Path(args.rl_model_path)
        if not model_path.exists():
            raise SystemExit(f"Model path not found: {model_path}")
        rl_env = MicrogridEnv(cfg)
        rl_summary = evaluate_policy(
            env=rl_env,
            policy_fn=load_rl_policy_fn(args.rl_algo, model_path),
            policy_name=f"{args.rl_algo}:{model_path.name}",
            episodes=args.episodes,
            seed_start=args.seed_start,
            use_safety=use_safety,
        )
        _print_result(args.rl_algo, rl_summary)


if __name__ == "__main__":
    main()
