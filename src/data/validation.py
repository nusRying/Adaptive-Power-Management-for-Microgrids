from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "renewable_kw",
    "load_kw",
    "price_import_per_kwh",
}

OPTIONAL_COLUMNS = {
    "timestamp",
    "price_export_per_kwh",
}


@dataclass
class ValidationReport:
    file_path: str
    rows: int
    columns: list[str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timestamp_start: str | None = None
    timestamp_end: str | None = None
    inferred_timestep_minutes: float | None = None

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def _count_non_finite(values: pd.Series) -> int:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    return int(np.isnan(arr).sum() + np.isinf(arr).sum())


def validate_profiles_frame(
    frame: pd.DataFrame, expected_dt_hours: float | None = None
) -> ValidationReport:
    report = ValidationReport(
        file_path="<in-memory>",
        rows=int(len(frame)),
        columns=[str(c) for c in frame.columns],
    )

    if frame.empty:
        report.errors.append("CSV has no rows.")
        return report

    missing_required = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_required:
        report.errors.append(f"Missing required columns: {missing_required}")
        return report

    numeric_columns = [
        "renewable_kw",
        "load_kw",
        "price_import_per_kwh",
    ]
    if "price_export_per_kwh" in frame.columns:
        numeric_columns.append("price_export_per_kwh")
    else:
        report.warnings.append(
            "Column 'price_export_per_kwh' is missing. Export price will be derived from import price."
        )

    for col in numeric_columns:
        non_finite = _count_non_finite(frame[col])
        if non_finite > 0:
            report.errors.append(f"Column '{col}' has {non_finite} non-finite values.")

    for col in numeric_columns:
        negative_count = int((pd.to_numeric(frame[col], errors="coerce") < 0).sum())
        if negative_count > 0:
            report.errors.append(f"Column '{col}' has {negative_count} negative values.")

    if "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce")
        invalid_ts = int(ts.isna().sum())
        if invalid_ts > 0:
            report.errors.append(f"Column 'timestamp' has {invalid_ts} invalid datetime values.")
        else:
            if not ts.is_monotonic_increasing:
                report.errors.append("Column 'timestamp' is not sorted in ascending order.")

            duplicate_count = int(ts.duplicated().sum())
            if duplicate_count > 0:
                report.errors.append(f"Column 'timestamp' has {duplicate_count} duplicate entries.")

            report.timestamp_start = str(ts.iloc[0])
            report.timestamp_end = str(ts.iloc[-1])

            if len(ts) > 1:
                diffs = ts.diff().dropna().dt.total_seconds() / 60.0
                if len(diffs) > 0:
                    report.inferred_timestep_minutes = float(diffs.median())
                    if expected_dt_hours is not None:
                        expected_minutes = expected_dt_hours * 60.0
                        if not np.isclose(
                            report.inferred_timestep_minutes,
                            expected_minutes,
                            atol=1.0,
                        ):
                            report.warnings.append(
                                "Inferred timestep "
                                f"{report.inferred_timestep_minutes:.2f} min does not match "
                                f"expected {expected_minutes:.2f} min."
                            )
    else:
        report.warnings.append("Column 'timestamp' is missing. Time continuity checks were skipped.")

    peak_load = float(pd.to_numeric(frame["load_kw"], errors="coerce").max())
    if peak_load > 10_000:
        report.warnings.append(
            f"Peak load appears high ({peak_load:.2f} kW). Verify scaling/units."
        )

    return report


def validate_profiles_csv(
    csv_path: str | Path, expected_dt_hours: float | None = None
) -> ValidationReport:
    file_path = Path(csv_path)
    frame = pd.read_csv(file_path)
    report = validate_profiles_frame(frame, expected_dt_hours=expected_dt_hours)
    report.file_path = str(file_path)
    return report

