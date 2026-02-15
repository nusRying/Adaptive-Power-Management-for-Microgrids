from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser(description="Evaluate controller/policy on microgrid environment.")
    parser.add_argument(
        "--policy",
        default="baseline",
        choices=["baseline", "random", "sac", "ddpg"],
        help="Policy/controller to evaluate.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to RL model file (required for sac/ddpg).",
    )
    parser.add_argument(
        "--microgrid-config",
        default="configs/microgrid.yaml",
        help="Path to microgrid config YAML.",
    )
    parser.add_argument(
        "--data-splits-config",
        default="configs/data_splits.yaml",
        help="Path to data split config YAML.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "none"],
        help="Profile split to evaluate when --profile-csv is not provided.",
    )
    parser.add_argument(
        "--profile-csv",
        default=None,
        help="Explicit profile CSV path override (takes precedence over --split).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=1000,
        help="Initial seed for evaluation episodes.",
    )
    parser.add_argument(
        "--no-safety",
        action="store_true",
        help="Disable SafetySupervisor during rollout.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output path for JSON metrics report.",
    )
    return parser


def _build_policy_fn(policy: str, env: MicrogridEnv):
    if policy == "random":
        return random_policy_fn(env)
    if policy == "baseline":
        return baseline_policy_fn(env.cfg)
    if policy in {"sac", "ddpg"}:
        raise ValueError("RL policy selected but model not loaded.")
    raise ValueError(f"Unknown policy type: {policy}")


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
    env = MicrogridEnv(cfg)

    if args.policy in {"sac", "ddpg"}:
        if not args.model_path:
            raise SystemExit("--model-path is required for RL policy evaluation.")
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise SystemExit(f"Model path not found: {model_path}")
        policy_fn = load_rl_policy_fn(
            algorithm=args.policy,
            model_path=model_path,
            deterministic=True,
        )
        policy_name = f"{args.policy}:{model_path.name}"
    else:
        policy_fn = _build_policy_fn(args.policy, env)
        policy_name = args.policy

    summary = evaluate_policy(
        env=env,
        policy_fn=policy_fn,
        policy_name=policy_name,
        episodes=args.episodes,
        seed_start=args.seed_start,
        use_safety=not args.no_safety,
    )

    print(f"Policy: {summary.policy}")
    print(f"Episodes: {summary.episodes}")
    print(f"Avg reward: {summary.avg_reward:.4f}")
    print(f"Avg grid cost: {summary.avg_grid_cost:.4f}")
    print(f"Avg degradation cost: {summary.avg_degradation_cost:.4f}")
    print(f"Avg penalty cost: {summary.avg_penalty_cost:.4f}")
    print(f"Avg unmet load (kWh): {summary.avg_unmet_load_kwh:.4f}")
    print(f"Avg curtailed (kWh): {summary.avg_curtailed_kwh:.4f}")
    print(f"Avg import (kWh): {summary.avg_import_kwh:.4f}")
    print(f"Avg export (kWh): {summary.avg_export_kwh:.4f}")
    print(f"Avg battery throughput (kWh): {summary.avg_battery_throughput_kwh:.4f}")
    print(f"Avg safety overrides: {summary.avg_safety_overrides:.2f}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as handle:
            json.dump(summary.to_dict(), handle, indent=2)
        print(f"JSON report written to: {out_path}")


if __name__ == "__main__":
    main()
