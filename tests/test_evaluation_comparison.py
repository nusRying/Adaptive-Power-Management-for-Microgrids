from src.evaluation.comparison import compare_policy_summaries
from src.evaluation.runner import EvaluationSummary


def _summary(
    policy: str,
    reward: float,
    grid: float,
    deg: float,
    penalty: float,
    unmet: float,
    curtailed: float,
    imp: float,
    exp: float,
    throughput: float,
    overrides: float,
) -> EvaluationSummary:
    return EvaluationSummary(
        policy=policy,
        episodes=3,
        avg_reward=reward,
        avg_grid_cost=grid,
        avg_degradation_cost=deg,
        avg_penalty_cost=penalty,
        avg_unmet_load_kwh=unmet,
        avg_curtailed_kwh=curtailed,
        avg_import_kwh=imp,
        avg_export_kwh=exp,
        avg_battery_throughput_kwh=throughput,
        avg_safety_overrides=overrides,
        details=[],
    )


def test_compare_policy_summaries_directionality() -> None:
    baseline = _summary(
        policy="baseline",
        reward=-120.0,
        grid=100.0,
        deg=5.0,
        penalty=15.0,
        unmet=1.0,
        curtailed=2.0,
        imp=80.0,
        exp=3.0,
        throughput=50.0,
        overrides=2.0,
    )
    candidate = _summary(
        policy="sac:model.zip",
        reward=-90.0,
        grid=85.0,
        deg=4.5,
        penalty=7.0,
        unmet=0.2,
        curtailed=1.5,
        imp=70.0,
        exp=4.0,
        throughput=45.0,
        overrides=1.0,
    )

    report = compare_policy_summaries(baseline, candidate)
    assert report.compared_metric_count > 0
    assert report.improved_metric_count > 0

    metric_map = {item.metric: item for item in report.metrics}
    assert metric_map["avg_reward"].improved is True
    assert metric_map["avg_grid_cost"].improved is True
    assert metric_map["avg_export_kwh"].improved is None

