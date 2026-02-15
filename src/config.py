from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvironmentConfig:
    episode_horizon: int = 96
    time_step_hours: float = 0.25
    seed: int = 42
    profile_csv: str | None = "data/raw/profiles.csv"


@dataclass
class BatteryConfig:
    capacity_kwh: float = 150.0
    soc_init: float = 0.5
    soc_min: float = 0.1
    soc_max: float = 0.9
    max_charge_kw: float = 75.0
    max_discharge_kw: float = 75.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    temperature_c: float = 30.0
    degradation_cost_per_kwh: float = 0.02


@dataclass
class GridConfig:
    max_import_kw: float = 250.0
    max_export_kw: float = 150.0
    sell_price_factor: float = 0.8


@dataclass
class RewardConfig:
    unmet_load_penalty_per_kwh: float = 15.0
    export_curtail_penalty_per_kwh: float = 0.1


@dataclass
class MicrogridConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    battery: BatteryConfig = field(default_factory=BatteryConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MicrogridConfig":
        return cls(
            environment=EnvironmentConfig(**data.get("environment", {})),
            battery=BatteryConfig(**data.get("battery", {})),
            grid=GridConfig(**data.get("grid", {})),
            reward=RewardConfig(**data.get("reward", {})),
        )


@dataclass
class TrainingConfig:
    algorithm: str = "sac"
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 200_000
    gamma: float = 0.99
    tau: float = 0.005
    model_dir: str = "models"
    tensorboard_log: str = "runs"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingConfig":
        body = data.get("training", data)
        return cls(**body)


@dataclass
class DataSplitsConfig:
    train_profile_csv: str = "data/processed/profiles_train.csv"
    val_profile_csv: str = "data/processed/profiles_val.csv"
    test_profile_csv: str = "data/processed/profiles_test.csv"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DataSplitsConfig":
        body = data.get("data_splits", data)
        return cls(**body)

    def profile_for_split(self, split: str) -> str:
        split_key = split.lower()
        mapping = {
            "train": self.train_profile_csv,
            "val": self.val_profile_csv,
            "test": self.test_profile_csv,
        }
        if split_key not in mapping:
            supported = ", ".join(sorted(mapping))
            raise ValueError(f"Unsupported split '{split}'. Supported: {supported}")
        return mapping[split_key]


def _load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_microgrid_config(path: str | Path = "configs/microgrid.yaml") -> MicrogridConfig:
    return MicrogridConfig.from_dict(_load_yaml(path))


def load_training_config(path: str | Path = "configs/training.yaml") -> TrainingConfig:
    return TrainingConfig.from_dict(_load_yaml(path))


def load_data_splits_config(path: str | Path = "configs/data_splits.yaml") -> DataSplitsConfig:
    return DataSplitsConfig.from_dict(_load_yaml(path))


def resolve_profile_csv_override(
    profile_csv: str | None,
    split: str | None,
    data_splits_config_path: str | Path = "configs/data_splits.yaml",
) -> str | None:
    if profile_csv:
        return profile_csv
    if split is None or split.lower() == "none":
        return None
    cfg_path = Path(data_splits_config_path)
    if not cfg_path.exists():
        raise ValueError(
            f"Data splits config not found: {cfg_path}. "
            "Provide --profile-csv or create the splits config file."
        )
    splits_cfg = load_data_splits_config(cfg_path)
    return splits_cfg.profile_for_split(split)
