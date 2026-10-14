#!/usr/bin/env python3

import argparse
import os

import numpy as np
import pandas as pd


DEFAULT_CSV = (
    "data/recordings/"
    "stop3_clear_0_drift_0p75/"
    "frames.csv"
)

TARGET_GT_RANGE_PX = 50.0
TARGET_CONSTANT_ERROR_PX = 10.0
IMAGE_CENTER_X = 640.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the lateral-drift CARLA run produces "
            "non-degenerate lane-center ground truth."
        )
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help="Path to the recorder frames.csv file.",
    )
    parser.add_argument(
        "--target-range-px",
        type=float,
        default=TARGET_GT_RANGE_PX,
        help=(
            "Minimum desired lane-GT pixel range for the validation gate."
        ),
    )
    parser.add_argument(
        "--target-constant-error-px",
        type=float,
        default=TARGET_CONSTANT_ERROR_PX,
        help=(
            "Minimum desired mean error of the trivial constant x=640 baseline."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        raise FileNotFoundError(
            f"CSV does not exist: {args.csv}"
        )

    df = pd.read_csv(args.csv)

    required = {
        "is_junction",
        "lane_gt_x",
    }

    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            "Missing required CSV columns: "
            + ", ".join(sorted(missing))
        )

    usable = df[
        (df["is_junction"] == 0)
        & df["lane_gt_x"].notna()
    ].copy()

    if usable.empty:
        raise RuntimeError(
            "No usable non-junction lane-GT frames were found."
        )

    gt = usable["lane_gt_x"].to_numpy(dtype=float)

    gt_min = float(np.min(gt))
    gt_max = float(np.max(gt))
    gt_range = gt_max - gt_min
    gt_mean = float(np.mean(gt))
    gt_std = float(np.std(gt))

    constant_errors = np.abs(gt - IMAGE_CENTER_X)
    constant_mean_error = float(np.mean(constant_errors))
    constant_median_error = float(np.median(constant_errors))
    constant_max_error = float(np.max(constant_errors))

    print()
    print("=" * 72)
    print("LATERAL-GEOMETRY VALIDATION")
    print("=" * 72)
    print(f"CSV: {args.csv}")
    print(f"All frames: {len(df)}")
    print(f"Usable non-junction frames: {len(usable)}")
    print()
    print(f"GT min:   {gt_min:.6f} px")
    print(f"GT max:   {gt_max:.6f} px")
    print(f"GT range: {gt_range:.6f} px")
    print(f"GT mean:  {gt_mean:.6f} px")
    print(f"GT std:   {gt_std:.6f} px")
    print()
    print("Constant x=640 baseline")
    print(f"  mean error:   {constant_mean_error:.6f} px")
    print(f"  median error: {constant_median_error:.6f} px")
    print(f"  max error:    {constant_max_error:.6f} px")

    if "lateral_offset_m" in usable.columns:
        offsets = usable["lateral_offset_m"].to_numpy(dtype=float)
        print()
        print("Recorded lateral offset")
        print(f"  min: {np.min(offsets):+.6f} m")
        print(f"  max: {np.max(offsets):+.6f} m")

    range_pass = gt_range >= args.target_range_px
    constant_pass = (
        constant_mean_error >= args.target_constant_error_px
    )

    print()
    print("Validation gates")
    print(
        f"  GT range >= {args.target_range_px:.1f} px: "
        f"{'PASS' if range_pass else 'FAIL'}"
    )
    print(
        "  constant x=640 mean error >= "
        f"{args.target_constant_error_px:.1f} px: "
        f"{'PASS' if constant_pass else 'FAIL'}"
    )
    print()

    if range_pass and constant_pass:
        print("OVERALL: PASS")
        print(
            "The lane geometry is no longer trivially constant. "
            "You can proceed to detector validation."
        )
    else:
        print("OVERALL: FAIL")
        print(
            "Do not collect the full weather sweep yet. "
            "Adjust the drift amplitude/setup and validate again."
        )

    print("=" * 72)
    print()


if __name__ == "__main__":
    main()
