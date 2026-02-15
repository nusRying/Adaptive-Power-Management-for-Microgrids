import pandas as pd

from src.data.validation import validate_profiles_frame


def _build_valid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2026-01-01T00:00:00",
                "2026-01-01T00:15:00",
                "2026-01-01T00:30:00",
            ],
            "renewable_kw": [10.0, 12.0, 11.0],
            "load_kw": [15.0, 16.0, 14.0],
            "price_import_per_kwh": [0.11, 0.12, 0.10],
            "price_export_per_kwh": [0.08, 0.09, 0.08],
        }
    )


def test_validate_profiles_frame_ok() -> None:
    report = validate_profiles_frame(_build_valid_frame(), expected_dt_hours=0.25)
    assert report.ok
    assert report.inferred_timestep_minutes is not None
    assert abs(report.inferred_timestep_minutes - 15.0) < 1e-6


def test_validate_profiles_frame_negative_values_fail() -> None:
    frame = _build_valid_frame()
    frame.loc[1, "load_kw"] = -1.0
    report = validate_profiles_frame(frame, expected_dt_hours=0.25)
    assert not report.ok
    assert any("negative values" in item for item in report.errors)

