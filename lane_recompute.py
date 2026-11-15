#!/usr/bin/env python3
"""
Recompute lane metrics on the corrected, method-independent evaluation region.

Eligibility is read from:
    results/lane_marking_lookahead_audit.csv

A frame is eligible iff, at the actual ground point corresponding to y=504:
    - lookahead waypoint is NOT a junction
    - left marking type != NONE
    - right marking type != NONE

This script NEVER changes the W/2 = 640 px non-detection penalty.

It scans:
    results/lane_hough/{clear,fog15,...,fog90}_per_frame.csv
    results/lane_lsd/{clear,fog15,...,fog90}_per_frame.csv
    results/lane_ufld/{clear,fog15,...,fog90}_per_frame.csv  (when available)

It also uses:
    results/lane_boundary_gt_clear.csv
for boundary-geometry metrics. Route geometry is identical across the weather
recordings, so the same CARLA-projected edge GT applies by sample_id.

Outputs:
    results/lane_valid_region/summary.csv
    results/lane_valid_region/<method>_<condition>_per_frame.csv

Run:
    cd ~/same-camera-different-failures
    conda activate carla
    python lane_recompute_valid_region.py
"""

import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
AUDIT_PATH = RESULTS / "lane_marking_lookahead_audit.csv"
EDGE_GT_PATH = RESULTS / "lane_boundary_gt_clear.csv"
OUT_DIR = RESULTS / "lane_valid_region"

FAILURE_ERROR_PX = 640.0
EPSILON_PX = 1.0

CONDITIONS = [
    ("clear", 0),
    ("fog15", 15),
    ("fog30", 30),
    ("fog45", 45),
    ("fog60", 60),
    ("fog75", 75),
    ("fog90", 90),
]

METHODS = {
    "Hough": {
        "dir": RESULTS / "lane_hough",
        "left": ["hough_left_x", "left_x_y504", "left_x"],
        "right": ["hough_right_x", "right_x_y504", "right_x"],
        "pred": ["hough_pred_x", "pred_center_x_y504", "pred_x"],
        "success": ["status", "success"],
        "error": ["error_px"],
    },
    "LSD": {
        "dir": RESULTS / "lane_lsd",
        "left": ["lsd_left_x", "left_x_y504", "left_x"],
        "right": ["lsd_right_x", "right_x_y504", "right_x"],
        "pred": ["lsd_pred_x", "pred_center_x_y504", "pred_x"],
        "success": ["status", "success"],
        "error": ["error_px"],
    },
    "UFLD-v2": {
        "dir": RESULTS / "lane_ufld",
        "left": ["left_x_y504", "ufld_left_x", "left_x"],
        "right": ["right_x_y504", "ufld_right_x", "right_x"],
        "pred": ["pred_center_x_y504", "ufld_pred_x", "pred_x"],
        "success": ["success", "status"],
        "error": ["error_px"],
    },
}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def first_col(fields, choices):
    for c in choices:
        if c in fields:
            return c
    return None


def to_float(x):
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        y = float(s)
    except ValueError:
        return None
    return y if math.isfinite(y) else None


def parse_success(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "yes", "success", "detected", "ok", "valid"}:
        return True
    if s in {
        "0", "false", "no", "failure", "failed", "missing",
        "no_detection", "detector_failure", "invalid",
    }:
        return False
    # Hough/LSD status strings may include wording.
    if "success" in s or "detected" in s or s == "ok":
        return True
    if (
        "fail" in s
        or "missing" in s
        or "no_" in s
        or "none" == s
    ):
        return False
    try:
        return bool(int(float(s)))
    except Exception:
        return None


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def median(xs):
    return float(np.median(xs)) if xs else float("nan")


def std(xs):
    return float(np.std(xs)) if xs else float("nan")


def load_eligible_ids():
    rows = read_csv(AUDIT_PATH)
    eligible = set()

    for r in rows:
        sid = int(float(r["sample_id"]))
        junc = int(float(r["lookahead_is_junction"]))
        left_t = str(r["left_marking_type"]).strip().upper()
        right_t = str(r["right_marking_type"]).strip().upper()

        if (
            junc == 0
            and left_t != "NONE"
            and right_t != "NONE"
        ):
            eligible.add(sid)

    return eligible


def load_edge_gt():
    rows = read_csv(EDGE_GT_PATH)
    out = {}
    for r in rows:
        sid = int(float(r["sample_id"]))
        out[sid] = {
            "gt_center": float(r["saved_lane_gt_x"]),
            "gt_left": float(r["gt_left_edge_x"]),
            "gt_right": float(r["gt_right_edge_x"]),
            "gt_width": float(r["gt_lane_width_px"]),
        }
    return out


def evaluate_file(method_name, cfg, condition_name, severity, path, eligible, gt):
    rows = read_csv(path)
    if not rows:
        raise RuntimeError(f"Empty CSV: {path}")

    fields = list(rows[0].keys())

    sid_col = first_col(fields, ["sample_id"])
    left_col = first_col(fields, cfg["left"])
    right_col = first_col(fields, cfg["right"])
    pred_col = first_col(fields, cfg["pred"])
    success_col = first_col(fields, cfg["success"])
    error_col = first_col(fields, cfg["error"])
    lane_gt_col = first_col(fields, ["lane_gt_x"])

    if sid_col is None:
        raise RuntimeError(f"{path}: no sample_id column")

    # Boundary columns are required for the geometry-aware summary.
    if left_col is None or right_col is None:
        raise RuntimeError(
            f"{path}: boundary columns not found.\n"
            f"Available columns: {fields}"
        )

    filtered = []

    for r in rows:
        sid = int(float(r[sid_col]))
        if sid not in eligible:
            continue

        if sid not in gt:
            raise RuntimeError(f"Missing edge GT for eligible sample {sid}")

        g = gt[sid]

        left = to_float(r.get(left_col))
        right = to_float(r.get(right_col))
        declared = parse_success(r.get(success_col)) if success_col else None

        pair_present = (
            left is not None
            and right is not None
            and left != right
        )

        success = pair_present
        if declared is False:
            success = False

        if success:
            pred_left = min(left, right)
            pred_right = max(left, right)
            pred_width = pred_right - pred_left
            pred_mid = 0.5 * (pred_left + pred_right)

            # The midpoint reconstructed from the boundaries is the canonical
            # center prediction for this geometry audit.
            center_error = abs(pred_mid - g["gt_center"])
            signed_center = pred_mid - g["gt_center"]

            left_error = abs(pred_left - g["gt_left"])
            right_error = abs(pred_right - g["gt_right"])
            width_error = abs(pred_width - g["gt_width"])
            width_bias = pred_width - g["gt_width"]

            failure_aware = center_error
        else:
            pred_left = pred_right = pred_width = pred_mid = None
            center_error = signed_center = None
            left_error = right_error = width_error = width_bias = None
            failure_aware = FAILURE_ERROR_PX

        out = dict(r)
        out.update({
            "eligible_lookahead_region": 1,
            "recomputed_success": int(success),
            "recomputed_pred_left_x": "" if pred_left is None else pred_left,
            "recomputed_pred_right_x": "" if pred_right is None else pred_right,
            "recomputed_pred_center_x": "" if pred_mid is None else pred_mid,
            "gt_left_edge_x": g["gt_left"],
            "gt_right_edge_x": g["gt_right"],
            "gt_lane_width_px": g["gt_width"],
            "recomputed_center_error_px": (
                "" if center_error is None else center_error
            ),
            "recomputed_center_signed_error_px": (
                "" if signed_center is None else signed_center
            ),
            "left_edge_abs_error_px": (
                "" if left_error is None else left_error
            ),
            "right_edge_abs_error_px": (
                "" if right_error is None else right_error
            ),
            "width_abs_error_px": (
                "" if width_error is None else width_error
            ),
            "width_signed_error_px": (
                "" if width_bias is None else width_bias
            ),
            "failure_aware_error_px": failure_aware,
        })
        filtered.append(out)

    # Exact same 88 positions for every method/condition.
    present_ids = {int(float(r["sample_id"])) for r in filtered}
    missing_ids = sorted(eligible - present_ids)
    if missing_ids:
        raise RuntimeError(
            f"{method_name} {condition_name}: missing eligible sample IDs: "
            f"{missing_ids}"
        )

    if len(filtered) != len(eligible):
        raise RuntimeError(
            f"{method_name} {condition_name}: got {len(filtered)} rows, "
            f"expected {len(eligible)}"
        )

    good = [r for r in filtered if int(r["recomputed_success"]) == 1]

    def vals(key):
        return [float(r[key]) for r in good]

    fa = [float(r["failure_aware_error_px"]) for r in filtered]

    summary = {
        "method": method_name,
        "condition": condition_name,
        "severity": severity,
        "eligible_frames": len(filtered),
        "successful_frames": len(good),
        "failed_frames": len(filtered) - len(good),
        "availability_pct": 100.0 * len(good) / len(filtered),
        "conditional_center_MAE_px": mean(vals("recomputed_center_error_px")),
        "conditional_center_median_px": median(vals("recomputed_center_error_px")),
        "conditional_center_signed_bias_px": mean(
            vals("recomputed_center_signed_error_px")
        ),
        "failure_aware_MAE_px": mean(fa),
        "left_edge_MAE_px": mean(vals("left_edge_abs_error_px")),
        "right_edge_MAE_px": mean(vals("right_edge_abs_error_px")),
        "width_MAE_px": mean(vals("width_abs_error_px")),
        "width_signed_bias_px": mean(vals("width_signed_error_px")),
        "pred_width_mean_px": mean(
            [float(r["recomputed_pred_right_x"]) -
             float(r["recomputed_pred_left_x"]) for r in good]
        ),
        "pred_width_std_px": std(
            [float(r["recomputed_pred_right_x"]) -
             float(r["recomputed_pred_left_x"]) for r in good]
        ),
        "R_lane": "",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{method_name.lower().replace('-v2','').replace(' ','_')}_{condition_name}_per_frame.csv"

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(filtered[0].keys()),
        )
        writer.writeheader()
        writer.writerows(filtered)

    return summary


def main():
    eligible = load_eligible_ids()
    gt = load_edge_gt()

    print("=" * 92)
    print("CORRECTED LANE EVALUATION REGION")
    print("=" * 92)
    print("Eligible frames:", len(eligible))
    print("Min sample ID:", min(eligible))
    print("Max sample ID:", max(eligible))
    print("Expected set 51-138:", eligible == set(range(51, 139)))

    if len(eligible) != 88:
        raise RuntimeError(
            f"Expected 88 eligible frames from the marking audit, got {len(eligible)}"
        )

    summaries = []

    for method_name, cfg in METHODS.items():
        for condition_name, severity in CONDITIONS:
            path = cfg["dir"] / f"{condition_name}_per_frame.csv"

            if not path.exists():
                # UFLD fog files do not exist yet; skip until generated.
                continue

            s = evaluate_file(
                method_name,
                cfg,
                condition_name,
                severity,
                path,
                eligible,
                gt,
            )
            summaries.append(s)

    # Compute reliability separately per method from that method's corrected Clear.
    by_method = {}
    for s in summaries:
        by_method.setdefault(s["method"], []).append(s)

    for method, items in by_method.items():
        clear = next(
            (x for x in items if int(x["severity"]) == 0),
            None
        )
        if clear is None:
            continue

        e0 = float(clear["failure_aware_MAE_px"])

        for x in items:
            es = float(x["failure_aware_MAE_px"])
            x["R_lane"] = min(
                1.0,
                (e0 + EPSILON_PX) / (es + EPSILON_PX),
            )

    summaries.sort(
        key=lambda x: (
            x["method"],
            int(x["severity"]),
        )
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summaries[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summaries)

    print()
    print("=" * 128)
    print("CORRECTED SUMMARY — SAME 88 LOOKAHEAD-VALID POSITIONS")
    print("=" * 128)
    print(
        f"{'Method':<9} {'Cond':<7} {'N':>4} {'Succ':>5} {'Avail%':>8} "
        f"{'CondMAE':>9} {'FailAware':>10} {'Ledge':>8} {'Redge':>8} "
        f"{'WidthMAE':>9} {'WidthBias':>10} {'R_lane':>8}"
    )

    for s in summaries:
        print(
            f"{s['method']:<9} "
            f"{s['condition']:<7} "
            f"{int(s['eligible_frames']):>4} "
            f"{int(s['successful_frames']):>5} "
            f"{float(s['availability_pct']):>8.2f} "
            f"{float(s['conditional_center_MAE_px']):>9.2f} "
            f"{float(s['failure_aware_MAE_px']):>10.2f} "
            f"{float(s['left_edge_MAE_px']):>8.2f} "
            f"{float(s['right_edge_MAE_px']):>8.2f} "
            f"{float(s['width_MAE_px']):>9.2f} "
            f"{float(s['width_signed_bias_px']):>10.2f} "
            f"{float(s['R_lane']):>8.3f}"
        )

    print()
    print("Saved:", summary_path)
    print("Filtered per-frame CSVs:", OUT_DIR)
    print()
    print("NOTE: Hough/LSD fog results are recomputed immediately.")
    print("UFLD fog rows will appear automatically after its fog CSVs exist.")


if __name__ == "__main__":
    main()
