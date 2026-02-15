from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_profile_frame(
    frame: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SplitResult:
    for name, value in [
        ("train_ratio", train_ratio),
        ("val_ratio", val_ratio),
        ("test_ratio", test_ratio),
    ]:
        if value <= 0:
            raise ValueError(f"{name} must be > 0. Got {value}.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0. Got {total:.6f} for "
            f"train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )
    if len(frame) < 3:
        raise ValueError("At least 3 rows are required to create train/val/test splits.")

    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce")
        if ts.isna().any():
            raise ValueError("Timestamp column contains invalid values; cannot split chronologically.")
        if not ts.is_monotonic_increasing:
            raise ValueError("Timestamp column must be sorted before splitting.")

    n_rows = len(frame)
    raw_counts = [
        int(n_rows * train_ratio),
        int(n_rows * val_ratio),
        int(n_rows * test_ratio),
    ]
    counts = [max(1, c) for c in raw_counts]

    while sum(counts) > n_rows:
        largest_index = max(range(3), key=lambda idx: counts[idx])
        if counts[largest_index] <= 1:
            break
        counts[largest_index] -= 1

    while sum(counts) < n_rows:
        counts[0] += 1

    n_train, n_val, n_test = counts
    if n_train <= 0 or n_val <= 0 or n_test <= 0 or (n_train + n_val + n_test) != n_rows:
        raise ValueError(
            "Unable to compute non-empty splits. "
            f"Counts train={n_train}, val={n_val}, test={n_test}, total={n_rows}."
        )

    train = frame.iloc[:n_train].reset_index(drop=True)
    val = frame.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = frame.iloc[n_train + n_val :].reset_index(drop=True)
    return SplitResult(train=train, val=val, test=test)


def split_profile_csv(
    input_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> dict[str, Path]:
    frame = pd.read_csv(input_csv)
    splits = split_profile_frame(
        frame=frame,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out / "profiles_train.csv",
        "val": out / "profiles_val.csv",
        "test": out / "profiles_test.csv",
    }
    splits.train.to_csv(paths["train"], index=False)
    splits.val.to_csv(paths["val"], index=False)
    splits.test.to_csv(paths["test"], index=False)
    return paths
