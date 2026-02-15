import pandas as pd

from src.data.profiles import load_profiles_from_csv


def test_load_profiles_derives_export_price_when_missing(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "renewable_kw": [10.0, 20.0],
            "load_kw": [15.0, 25.0],
            "price_import_per_kwh": [0.10, 0.20],
        }
    )
    csv_path = tmp_path / "profiles.csv"
    frame.to_csv(csv_path, index=False)

    profiles = load_profiles_from_csv(csv_path, horizon=2, sell_price_factor=0.8)
    assert abs(float(profiles.price_export_per_kwh[0]) - 0.08) < 1e-6
    assert abs(float(profiles.price_export_per_kwh[1]) - 0.16) < 1e-6
