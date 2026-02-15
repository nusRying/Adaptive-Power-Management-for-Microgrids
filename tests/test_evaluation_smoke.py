from src.config import MicrogridConfig
from src.envs.microgrid_env import MicrogridEnv
from src.evaluation.runner import baseline_policy_fn, evaluate_policy


def test_evaluate_policy_smoke() -> None:
    cfg = MicrogridConfig()
    env = MicrogridEnv(cfg)
    policy = baseline_policy_fn(cfg)

    summary = evaluate_policy(
        env=env,
        policy_fn=policy,
        policy_name="baseline",
        episodes=2,
        seed_start=5,
        use_safety=True,
    )

    assert summary.episodes == 2
    assert len(summary.details) == 2
    assert isinstance(summary.avg_reward, float)

