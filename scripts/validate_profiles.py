from __future__ import annotations

import argparse
import sys

from src.config import load_microgrid_config
from src.data.validation import validate_profiles_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate microgrid profile CSV before training/evaluation."
    )
    parser.add_argument(
        "--input",
        default="data/raw/profiles.csv",
        help="Path to profile CSV file.",
    )
    parser.add_argument(
        "--microgrid-config",
        default="configs/microgrid.yaml",
        help="Path to microgrid config for default timestep.",
    )
    parser.add_argument(
        "--dt-hours",
        type=float,
        default=None,
        help="Expected timestep in hours. If omitted, reads from microgrid config.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failure.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    expected_dt_hours = args.dt_hours
    if expected_dt_hours is None:
        cfg = load_microgrid_config(args.microgrid_config)
        expected_dt_hours = cfg.environment.time_step_hours

    report = validate_profiles_csv(
        csv_path=args.input,
        expected_dt_hours=expected_dt_hours,
    )

    print(f"Validation file: {report.file_path}")
    print(f"Rows: {report.rows}")
    print(f"Columns: {', '.join(report.columns)}")
    if report.timestamp_start and report.timestamp_end:
        print(f"Timestamp range: {report.timestamp_start} -> {report.timestamp_end}")
    if report.inferred_timestep_minutes is not None:
        print(f"Inferred timestep: {report.inferred_timestep_minutes:.2f} minutes")

    if report.errors:
        print("\nErrors:")
        for item in report.errors:
            print(f"- {item}")
    if report.warnings:
        print("\nWarnings:")
        for item in report.warnings:
            print(f"- {item}")

    failed = (not report.ok) or (args.strict and bool(report.warnings))
    if failed:
        print("\nResult: FAILED")
        sys.exit(1)

    print("\nResult: PASSED")


if __name__ == "__main__":
    main()

