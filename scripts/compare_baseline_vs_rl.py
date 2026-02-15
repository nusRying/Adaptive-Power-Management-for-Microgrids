from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_microgrid_config, resolve_profile_csv_override
from src.envs.microgrid_env import MicrogridEnv
from src.evaluation.comparison import compare_policy_summaries
from src.evaluation.runner import (
    baseline_policy_fn,
    evaluate_policy,
    load_rl_policy_fn,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare baseline controller vs trained RL policy with delta report."
    )
    parser.add_argument(
        "--algo",
        required=True,
        choices=["sac", "ddpg"],
        help="RL algorithm used for the model.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained RL model zip file.",
    )
    parser.add_argument("--microgrid-config", default="configs/microgrid.yaml")
    parser.add_argument("--data-splits-config", default="configs/data_splits.yaml")
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
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--no-safety", action="store_true")
    parser.add_argument(
        "--json-out",
        default="reports/baseline_vs_rl.json",
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--markdown-out",
        default="reports/baseline_vs_rl.md",
        help="Output Markdown report path.",
    )
    return parser


def _format_improvement(improvement_pct: float | None) -> str:
    if improvement_pct is None:
        return "n/a"
    return f"{improvement_pct:+.2f}%"


def _render_markdown(
    baseline_name: str,
    candidate_name: str,
    report: dict[str, object],
) -> str:
    lines: list[str] = []
    lines.append("# Baseline vs RL Comparison")
    lines.append("")
    lines.append(f"- Baseline: `{baseline_name}`")
    lines.append(f"- Candidate: `{candidate_name}`")
    lines.append(
        f"- Improved metrics: {report['improved_metric_count']}/{report['compared_metric_count']}"
    )
    lines.append("")
    lines.append(
        "| Metric | Objective | Baseline | Candidate | Delta (RL - Base) | Improvement | Improved |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|")
    for metric in report["metrics"]:
        assert isinstance(metric, dict)
        lines.append(
            "| {metric} | {objective} | {baseline:.4f} | {candidate:.4f} | {delta:+.4f} | {impr} | {improved} |".format(
                metric=metric["metric"],
                objective=metric["objective"],
                baseline=float(metric["baseline"]),
                candidate=float(metric["candidate"]),
                delta=float(metric["delta"]),
                impr=_format_improvement(
                    None
                    if metric["improvement_pct"] is None
                    else float(metric["improvement_pct"])
                ),
                improved="n/a" if metric["improved"] is None else str(bool(metric["improved"])),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model path not found: {model_path}")

    cfg = load_microgrid_config(args.microgrid_config)
    profile_csv_override = resolve_profile_csv_override(
        profile_csv=args.profile_csv,
        split=args.split,
        data_splits_config_path=args.data_splits_config,
    )
    if profile_csv_override:
        cfg.environment.profile_csv = profile_csv_override

    baseline_env = MicrogridEnv(cfg)
    rl_env = MicrogridEnv(cfg)
    use_safety = not args.no_safety

    baseline_summary = evaluate_policy(
        env=baseline_env,
        policy_fn=baseline_policy_fn(cfg),
        policy_name="baseline",
        episodes=args.episodes,
        seed_start=args.seed_start,
        use_safety=use_safety,
    )
    rl_summary = evaluate_policy(
        env=rl_env,
        policy_fn=load_rl_policy_fn(args.algo, model_path),
        policy_name=f"{args.algo}:{model_path.name}",
        episodes=args.episodes,
        seed_start=args.seed_start,
        use_safety=use_safety,
    )

    comparison = compare_policy_summaries(baseline_summary, rl_summary)

    print(f"Baseline: {baseline_summary.policy}")
    print(f"Candidate: {rl_summary.policy}")
    print(
        f"Improved metrics: {comparison.improved_metric_count}/{comparison.compared_metric_count}"
    )
    for metric in comparison.metrics:
        improvement = _format_improvement(metric.improvement_pct)
        improved = "n/a" if metric.improved is None else str(metric.improved)
        print(
            f"- {metric.metric}: base={metric.baseline:.4f}, rl={metric.candidate:.4f}, "
            f"delta={metric.delta:+.4f}, improvement={improvement}, improved={improved}"
        )

    payload = {
        "baseline_summary": baseline_summary.to_dict(),
        "candidate_summary": rl_summary.to_dict(),
        "comparison": comparison.to_dict(),
    }

    json_out = Path(args.json_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    with json_out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"JSON report written to: {json_out}")

    markdown_out = Path(args.markdown_out)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    markdown = _render_markdown(
        baseline_name=baseline_summary.policy,
        candidate_name=rl_summary.policy,
        report=comparison.to_dict(),
    )
    with markdown_out.open("w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Markdown report written to: {markdown_out}")


if __name__ == "__main__":
    main()

