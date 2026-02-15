from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import load_microgrid_config
from src.data.splitting import split_profile_csv
from src.data.validation import validate_profiles_csv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chronologically split profile CSV into train/val/test files."
    )
    parser.add_argument(
        "--input",
        default="data/raw/profiles.csv",
        help="Input profile CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory to write split CSV files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--microgrid-config",
        default="configs/microgrid.yaml",
        help="Path to microgrid config for horizon/timestep checks.",
    )
    parser.add_argument(
        "--manifest",
        default="data/processed/splits_manifest.json",
        help="Path to write split metadata JSON.",
    )
    return parser


def _row_count(path: str) -> int:
    return int(len(pd.read_csv(path)))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = load_microgrid_config(args.microgrid_config)
    report = validate_profiles_csv(
        csv_path=args.input,
        expected_dt_hours=cfg.environment.time_step_hours,
    )
    if not report.ok:
        print("Input CSV failed validation and cannot be split.")
        for item in report.errors:
            print(f"- {item}")
        raise SystemExit(1)

    paths = split_profile_csv(
        input_csv=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    train_rows = _row_count(str(paths["train"]))
    val_rows = _row_count(str(paths["val"]))
    test_rows = _row_count(str(paths["test"]))

    print("Split created:")
    print(f"- train: {paths['train']} ({train_rows} rows)")
    print(f"- val:   {paths['val']} ({val_rows} rows)")
    print(f"- test:  {paths['test']} ({test_rows} rows)")

    horizon = int(cfg.environment.episode_horizon)
    for name, rows in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        if rows < horizon:
            print(
                f"Warning: {name} split has {rows} rows, below episode_horizon={horizon}. "
                "The environment will repeat/tile data."
            )

    manifest = {
        "input": args.input,
        "output_dir": args.output_dir,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "rows": {
            "train": train_rows,
            "val": val_rows,
            "test": test_rows,
        },
        "files": {
            "train": str(paths["train"]),
            "val": str(paths["val"]),
            "test": str(paths["test"]),
        },
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"- manifest: {manifest_path}")


if __name__ == "__main__":
    main()
