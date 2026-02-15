from __future__ import annotations

from dataclasses import asdict, dataclass

from src.evaluation.runner import EvaluationSummary


@dataclass
class MetricComparison:
    metric: str
    objective: str  # "higher", "lower", or "neutral"
    baseline: float
    candidate: float
    delta: float
    improved: bool | None
    improvement_pct: float | None


@dataclass
class ComparisonReport:
    baseline_policy: str
    candidate_policy: str
    episodes: int
    metrics: list[MetricComparison]
    improved_metric_count: int
    compared_metric_count: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["metrics"] = [asdict(metric) for metric in self.metrics]
        return data


def _compute_improvement(
    baseline: float,
    candidate: float,
    objective: str,
) -> tuple[bool | None, float | None]:
    if objective == "neutral":
        return None, None

    if objective == "higher":
        improved = candidate > baseline
        if abs(baseline) < 1e-12:
            return improved, None
        return improved, float((candidate - baseline) / abs(baseline) * 100.0)

    if objective == "lower":
        improved = candidate < baseline
        if abs(baseline) < 1e-12:
            return improved, None
        return improved, float((baseline - candidate) / abs(baseline) * 100.0)

    raise ValueError(f"Unsupported objective '{objective}'.")


def compare_policy_summaries(
    baseline: EvaluationSummary,
    candidate: EvaluationSummary,
) -> ComparisonReport:
    specs = [
        ("avg_reward", "higher"),
        ("avg_grid_cost", "lower"),
        ("avg_degradation_cost", "lower"),
        ("avg_penalty_cost", "lower"),
        ("avg_unmet_load_kwh", "lower"),
        ("avg_curtailed_kwh", "lower"),
        ("avg_import_kwh", "lower"),
        ("avg_export_kwh", "neutral"),
        ("avg_battery_throughput_kwh", "lower"),
        ("avg_safety_overrides", "lower"),
    ]

    metrics: list[MetricComparison] = []
    improved_metric_count = 0
    compared_metric_count = 0

    for attr, objective in specs:
        b = float(getattr(baseline, attr))
        c = float(getattr(candidate, attr))
        delta = c - b
        improved, improvement_pct = _compute_improvement(b, c, objective)
        if improved is not None:
            compared_metric_count += 1
            if improved:
                improved_metric_count += 1
        metrics.append(
            MetricComparison(
                metric=attr,
                objective=objective,
                baseline=b,
                candidate=c,
                delta=delta,
                improved=improved,
                improvement_pct=improvement_pct,
            )
        )

    return ComparisonReport(
        baseline_policy=baseline.policy,
        candidate_policy=candidate.policy,
        episodes=min(baseline.episodes, candidate.episodes),
        metrics=metrics,
        improved_metric_count=improved_metric_count,
        compared_metric_count=compared_metric_count,
    )

