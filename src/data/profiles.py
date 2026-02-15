from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Profiles:
    renewable_kw: np.ndarray
    load_kw: np.ndarray
    price_import_per_kwh: np.ndarray
    price_export_per_kwh: np.ndarray


def _resize(series: np.ndarray, horizon: int) -> np.ndarray:
    if series.size == 0:
        raise ValueError("Profile series cannot be empty.")
    if series.size >= horizon:
        return series[:horizon].astype(np.float32)
    repeats = int(np.ceil(horizon / series.size))
    return np.tile(series, repeats)[:horizon].astype(np.float32)


def build_synthetic_profiles(horizon: int, dt_hours: float, seed: int) -> Profiles:
    rng = np.random.default_rng(seed)
    time_idx = np.arange(horizon, dtype=np.float32)
    hour_of_day = (time_idx * dt_hours) % 24.0

    solar_shape = np.maximum(0.0, np.sin((hour_of_day - 6.0) / 12.0 * np.pi))
    wind_shape = 0.45 + 0.20 * np.sin((hour_of_day + 3.0) * 2.0 * np.pi / 24.0)
    wind_noise = rng.normal(0.0, 0.05, size=horizon)
    wind_shape = np.clip(wind_shape + wind_noise, 0.0, 1.0)
    renewable_kw = np.clip(80.0 * solar_shape + 40.0 * wind_shape, 0.0, None)

    load_base = 110.0 + 18.0 * np.sin((hour_of_day - 17.0) * 2.0 * np.pi / 24.0)
    load_noise = rng.normal(0.0, 4.0, size=horizon)
    load_kw = np.clip(load_base + load_noise, 60.0, None)

    evening_peak = ((hour_of_day >= 17.0) & (hour_of_day <= 22.0)).astype(np.float32)
    price_import = 0.10 + 0.08 * evening_peak + rng.normal(0.0, 0.004, size=horizon)
    price_import = np.clip(price_import, 0.05, None)
    price_export = 0.75 * price_import

    return Profiles(
        renewable_kw=renewable_kw.astype(np.float32),
        load_kw=load_kw.astype(np.float32),
        price_import_per_kwh=price_import.astype(np.float32),
        price_export_per_kwh=price_export.astype(np.float32),
    )


def load_profiles_from_csv(
    csv_path: str | Path, horizon: int, sell_price_factor: float = 0.75
) -> Profiles:
    file_path = Path(csv_path)
    frame = pd.read_csv(file_path)

    required = {
        "renewable_kw",
        "load_kw",
        "price_import_per_kwh",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if "price_export_per_kwh" in frame.columns:
        price_export_series = frame["price_export_per_kwh"].to_numpy(dtype=np.float32)
    else:
        price_export_series = (
            frame["price_import_per_kwh"].to_numpy(dtype=np.float32) * float(sell_price_factor)
        )

    return Profiles(
        renewable_kw=_resize(frame["renewable_kw"].to_numpy(dtype=np.float32), horizon),
        load_kw=_resize(frame["load_kw"].to_numpy(dtype=np.float32), horizon),
        price_import_per_kwh=_resize(
            frame["price_import_per_kwh"].to_numpy(dtype=np.float32), horizon
        ),
        price_export_per_kwh=_resize(price_export_series, horizon),
    )


def get_profiles(
    profile_csv: str | None,
    horizon: int,
    dt_hours: float,
    seed: int,
    sell_price_factor: float = 0.75,
) -> Profiles:
    if profile_csv and Path(profile_csv).exists():
        return load_profiles_from_csv(
            csv_path=profile_csv,
            horizon=horizon,
            sell_price_factor=sell_price_factor,
        )
    return build_synthetic_profiles(horizon, dt_hours, seed)
