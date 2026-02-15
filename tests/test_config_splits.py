from pathlib import Path

import pytest

from src.config import load_data_splits_config, resolve_profile_csv_override


def test_load_data_splits_config_and_profile_lookup(tmp_path: Path) -> None:
    yaml_path = tmp_path / "splits.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "data_splits:",
                "  train_profile_csv: data/custom/train.csv",
                "  val_profile_csv: data/custom/val.csv",
                "  test_profile_csv: data/custom/test.csv",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_data_splits_config(yaml_path)
    assert cfg.profile_for_split("train") == "data/custom/train.csv"
    assert cfg.profile_for_split("val") == "data/custom/val.csv"
    assert cfg.profile_for_split("test") == "data/custom/test.csv"


def test_resolve_profile_csv_override_precedence(tmp_path: Path) -> None:
    yaml_path = tmp_path / "splits.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                "data_splits:",
                "  train_profile_csv: data/custom/train.csv",
                "  val_profile_csv: data/custom/val.csv",
                "  test_profile_csv: data/custom/test.csv",
            ]
        ),
        encoding="utf-8",
    )

    explicit = resolve_profile_csv_override(
        profile_csv="data/explicit.csv",
        split="train",
        data_splits_config_path=yaml_path,
    )
    assert explicit == "data/explicit.csv"

    from_split = resolve_profile_csv_override(
        profile_csv=None,
        split="test",
        data_splits_config_path=yaml_path,
    )
    assert from_split == "data/custom/test.csv"

    none_value = resolve_profile_csv_override(
        profile_csv=None,
        split="none",
        data_splits_config_path=yaml_path,
    )
    assert none_value is None


def test_resolve_profile_csv_override_missing_config_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ValueError):
        resolve_profile_csv_override(
            profile_csv=None,
            split="train",
            data_splits_config_path=missing,
        )
