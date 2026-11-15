#!/usr/bin/env python3
"""
Compare Clear-weather lane boundary geometry for Hough, LSD, and UFLD-v2
against CARLA-projected left/right lane-edge ground truth at y=504.

Run from:
    ~/same-camera-different-failures
    conda activate carla
    python lane_clear_boundary_compare.py

No CARLA server is needed. This uses existing CSV files only.

Outputs:
    results/lane_clear_boundary_comparison.csv
    results/lane_clear_boundary_per_frame.csv
"""

import csv
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
GT_PATH = ROOT / "results" / "lane_boundary_gt_clear.csv"

METHODS = {
    "Hough": ROOT / "results" / "lane_hough" / "clear_per_frame.csv",
    "LSD": ROOT / "results" / "lane_lsd" / "clear_per_frame.csv",
    "UFLD-v2": ROOT / "results" / "lane_ufld" / "clear_per_frame.csv",
}

FAILURE_ERROR_PX = 640.0

# Candidate column names so the script works with the existing detector CSVs
# even if Hough/LSD/UFLD used slightly different labels.
SAMPLE_ID_NAMES = [
    "sample_id", "sample", "frame_id", "id",
]

LEFT_NAMES = [
    "left_x_y504",
    "left_x",
    "left_pred_x",
    "left_lane_x",
    "left_x_at_lookahead",
    "left_x_lookahead",
    "pred_left_x",
    "hough_left_x",
    "lsd_left_x",
    "ufld_left_x",
]

RIGHT_NAMES = [
    "right_x_y504",
    "right_x",
    "right_pred_x",
    "right_lane_x",
    "right_x_at_lookahead",
    "right_x_lookahead",
    "pred_right_x",
    "hough_right_x",
    "lsd_right_x",
    "ufld_right_x",
]

CENTER_NAMES = [
    "pred_center_x_y504",
    "pred_x",
    "pred_center_x",
    "lane_center_x",
    "predicted_center_x",
    "hough_pred_x",
    "lsd_pred_x",
    "ufld_pred_x",
]

SUCCESS_NAMES = [
    "success",
    "detected",
    "detection_success",
    "valid",
    "status",
]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_col(fieldnames, candidates, required=False, label="column"):
    # 1) Exact known names first.
    for name in candidates:
        if name in fieldnames:
            return name

    # 2) Robust suffix fallback for detector-prefixed CSVs such as
    #    hough_left_x, lsd_right_x, hough_pred_x.
    suffixes = []
    lower_label = label.lower()

    if "left" in lower_label:
        suffixes = ["left_x", "left_pred_x", "left_lane_x"]
    elif "right" in lower_label:
        suffixes = ["right_x", "right_pred_x", "right_lane_x"]
    elif "center" in lower_label:
        suffixes = ["pred_x", "center_x", "pred_center_x"]

    if suffixes:
        matches = [
            f for f in fieldnames
            if any(f.lower().endswith(suf) for suf in suffixes)
        ]
        if len(matches) == 1:
            return matches[0]

    if required:
        raise RuntimeError(
            f"Could not find {label}. "
            f"Tried: {candidates}\n"
            f"Available columns: {fieldnames}"
        )
    return None


def as_float(value):
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if not math.isfinite(x):
        return None
    return x


def as_bool(value):
    if value is None:
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "success", "detected", "ok", "valid"}:
        return True
    if s in {
        "0", "false", "no", "n", "failure", "failed", "missing",
        "no_detection", "detector_failure", "invalid"
    }:
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return None


def mean_or_nan(xs):
    return float(np.mean(xs)) if xs else float("nan")


def median_or_nan(xs):
    return float(np.median(xs)) if xs else float("nan")


def std_or_nan(xs):
    return float(np.std(xs)) if xs else float("nan")


def summarize_method(name, path, gt_by_id):
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"{name}: empty CSV: {path}")

    fields = list(rows[0].keys())

    sid_col = find_col(
        fields, SAMPLE_ID_NAMES, required=True,
        label=f"{name} sample-id column"
    )
    left_col = find_col(
        fields, LEFT_NAMES, required=True,
        label=f"{name} left-boundary column"
    )
    right_col = find_col(
        fields, RIGHT_NAMES, required=True,
        label=f"{name} right-boundary column"
    )
    center_col = find_col(
        fields, CENTER_NAMES, required=False
    )
    success_col = find_col(
        fields, SUCCESS_NAMES, required=False
    )

    print()
    print(f"{name} detected columns:")
    print("  sample:", sid_col)
    print("  left:  ", left_col)
    print("  right: ", right_col)
    print("  center:", center_col if center_col else "(recomputed)")
    print("  success:", success_col if success_col else "(from boundary presence)")

    per_frame = []

    for r in rows:
        sid_val = as_float(r.get(sid_col))
        if sid_val is None:
            continue
        sid = int(round(sid_val))

        if sid not in gt_by_id:
            continue

        gt = gt_by_id[sid]

        raw_left = as_float(r.get(left_col))
        raw_right = as_float(r.get(right_col))

        declared_success = (
            as_bool(r.get(success_col))
            if success_col else None
        )

        pair_present = (
            raw_left is not None
            and raw_right is not None
            and raw_left != raw_right
        )

        # If detector CSV explicitly says failure, respect it.
        success = pair_present
        if declared_success is False:
            success = False

        if success:
            pred_left = min(raw_left, raw_right)
            pred_right = max(raw_left, raw_right)
            pred_width = pred_right - pred_left

            raw_center = (
                as_float(r.get(center_col))
                if center_col else None
            )
            pred_center = (
                raw_center
                if raw_center is not None
                else 0.5 * (pred_left + pred_right)
            )

            # Recompute midpoint too, so a stale center column cannot hide
            # a boundary mismatch.
            midpoint_from_edges = 0.5 * (pred_left + pred_right)

            left_err = abs(pred_left - gt["gt_left_edge_x"])
            right_err = abs(pred_right - gt["gt_right_edge_x"])
            width_err = abs(pred_width - gt["gt_lane_width_px"])
            center_err = abs(midpoint_from_edges - gt["saved_lane_gt_x"])

            left_signed = pred_left - gt["gt_left_edge_x"]
            right_signed = pred_right - gt["gt_right_edge_x"]
            width_signed = pred_width - gt["gt_lane_width_px"]
            center_signed = midpoint_from_edges - gt["saved_lane_gt_x"]

            failure_aware = center_err
        else:
            pred_left = pred_right = pred_width = None
            pred_center = midpoint_from_edges = None
            left_err = right_err = width_err = center_err = None
            left_signed = right_signed = width_signed = center_signed = None
            failure_aware = FAILURE_ERROR_PX

        per_frame.append({
            "method": name,
            "sample_id": sid,
            "success": int(success),
            "gt_left_edge_x": gt["gt_left_edge_x"],
            "gt_right_edge_x": gt["gt_right_edge_x"],
            "gt_lane_width_px": gt["gt_lane_width_px"],
            "gt_center_x": gt["saved_lane_gt_x"],
            "pred_left_x": "" if pred_left is None else pred_left,
            "pred_right_x": "" if pred_right is None else pred_right,
            "pred_width_px": "" if pred_width is None else pred_width,
            "pred_center_from_edges_x": (
                "" if midpoint_from_edges is None else midpoint_from_edges
            ),
            "left_edge_abs_error_px": "" if left_err is None else left_err,
            "right_edge_abs_error_px": "" if right_err is None else right_err,
            "width_abs_error_px": "" if width_err is None else width_err,
            "center_abs_error_px": "" if center_err is None else center_err,
            "left_edge_signed_error_px": "" if left_signed is None else left_signed,
            "right_edge_signed_error_px": "" if right_signed is None else right_signed,
            "width_signed_error_px": "" if width_signed is None else width_signed,
            "center_signed_error_px": "" if center_signed is None else center_signed,
            "failure_aware_center_error_px": failure_aware,
        })

    # There should be one row per one of the 109 non-junction GT frames.
    n_total = len(per_frame)
    good = [r for r in per_frame if r["success"] == 1]
    n_good = len(good)

    def vals(key):
        return [float(r[key]) for r in good]

    failure_aware_vals = [
        float(r["failure_aware_center_error_px"])
        for r in per_frame
    ]

    summary = {
        "method": name,
        "evaluated_frames": n_total,
        "successful_frames": n_good,
        "failed_frames": n_total - n_good,
        "availability_pct": 100.0 * n_good / n_total if n_total else float("nan"),

        "left_edge_MAE_px": mean_or_nan(vals("left_edge_abs_error_px")),
        "right_edge_MAE_px": mean_or_nan(vals("right_edge_abs_error_px")),
        "boundary_MAE_mean_of_sides_px": mean_or_nan(
            vals("left_edge_abs_error_px")
            + vals("right_edge_abs_error_px")
        ),

        "width_MAE_px": mean_or_nan(vals("width_abs_error_px")),
        "width_signed_bias_px": mean_or_nan(vals("width_signed_error_px")),
        "pred_width_mean_px": mean_or_nan(
            [float(r["pred_width_px"]) for r in good]
        ),
        "pred_width_std_px": std_or_nan(
            [float(r["pred_width_px"]) for r in good]
        ),

        "center_MAE_success_px": mean_or_nan(vals("center_abs_error_px")),
        "center_median_success_px": median_or_nan(vals("center_abs_error_px")),
        "center_signed_bias_px": mean_or_nan(vals("center_signed_error_px")),

        "failure_aware_center_MAE_px": mean_or_nan(failure_aware_vals),
    }

    return summary, per_frame


def main():
    gt_rows = read_csv(GT_PATH)
    if not gt_rows:
        raise RuntimeError(f"Empty GT CSV: {GT_PATH}")

    gt_by_id = {}

    for r in gt_rows:
        sid = int(float(r["sample_id"]))
        gt_by_id[sid] = {
            "saved_lane_gt_x": float(r["saved_lane_gt_x"]),
            "gt_left_edge_x": float(r["gt_left_edge_x"]),
            "gt_right_edge_x": float(r["gt_right_edge_x"]),
            "gt_lane_width_px": float(r["gt_lane_width_px"]),
        }

    print("=" * 86)
    print("CLEAR LANE BOUNDARY COMPARISON AGAINST CARLA GROUND TRUTH @ y=504")
    print("=" * 86)
    print("GT frames:", len(gt_by_id))
    gt_widths = [v["gt_lane_width_px"] for v in gt_by_id.values()]
    print(
        "CARLA GT width mean/std: "
        f"{np.mean(gt_widths):.3f} / {np.std(gt_widths):.3f} px"
    )

    summaries = []
    all_per_frame = []

    for name, path in METHODS.items():
        if not path.exists():
            raise RuntimeError(f"Missing {name} result: {path}")

        summary, frames = summarize_method(
            name, path, gt_by_id
        )
        summaries.append(summary)
        all_per_frame.extend(frames)

    print()
    print("=" * 86)
    print("SUMMARY")
    print("=" * 86)

    header = (
        f"{'Method':<10}"
        f"{'Avail%':>9}"
        f"{'L-edge':>10}"
        f"{'R-edge':>10}"
        f"{'WidthMAE':>11}"
        f"{'WidthBias':>11}"
        f"{'CtrMAE':>10}"
        f"{'FailAware':>11}"
    )
    print(header)

    for s in summaries:
        print(
            f"{s['method']:<10}"
            f"{s['availability_pct']:>9.2f}"
            f"{s['left_edge_MAE_px']:>10.2f}"
            f"{s['right_edge_MAE_px']:>10.2f}"
            f"{s['width_MAE_px']:>11.2f}"
            f"{s['width_signed_bias_px']:>11.2f}"
            f"{s['center_MAE_success_px']:>10.2f}"
            f"{s['failure_aware_center_MAE_px']:>11.2f}"
        )

    print()
    print("Predicted width mean ± std vs GT:")
    gt_mean = float(np.mean(gt_widths))
    gt_std = float(np.std(gt_widths))
    print(f"  CARLA GT : {gt_mean:.3f} ± {gt_std:.3f} px")
    for s in summaries:
        print(
            f"  {s['method']:<8}: "
            f"{s['pred_width_mean_px']:.3f} ± "
            f"{s['pred_width_std_px']:.3f} px"
        )

    summary_path = ROOT / "results" / "lane_clear_boundary_comparison.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    per_frame_path = ROOT / "results" / "lane_clear_boundary_per_frame.csv"
    with per_frame_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_per_frame[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_per_frame)

    print()
    print("Saved summary:", summary_path)
    print("Saved per-frame:", per_frame_path)
    print()
    print("Interpret the raw metrics first.")
    print("Do not introduce a new pass/fail boundary threshold after seeing them.")


if __name__ == "__main__":
    main()
