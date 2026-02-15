import pandas as pd
import pytest

from src.data.splitting import split_profile_frame


def test_split_profile_frame_counts() -> None:
    timestamps = pd.date_range("2026-01-01", periods=20, freq="15min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps.astype(str),
            "renewable_kw": [10.0] * 20,
            "load_kw": [12.0] * 20,
            "price_import_per_kwh": [0.1] * 20,
            "price_export_per_kwh": [0.08] * 20,
        }
    )

    result = split_profile_frame(frame, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    assert len(result.train) == 14
    assert len(result.val) == 3
    assert len(result.test) == 3


def test_split_profile_frame_rejects_non_positive_ratio() -> None:
    frame = pd.DataFrame(
        {
            "renewable_kw": [10.0, 10.0, 10.0],
            "load_kw": [12.0, 12.0, 12.0],
            "price_import_per_kwh": [0.1, 0.1, 0.1],
        }
    )
    with pytest.raises(ValueError):
        split_profile_frame(frame, train_ratio=0.0, val_ratio=0.5, test_ratio=0.5)
