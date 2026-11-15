#!/usr/bin/env python3
"""
Offline screen of five phase-shifted lateral trajectories BEFORE final recording.

No CARLA server is required. This uses:
  - the frozen route CSV,
  - the existing Clear frames.csv,
  - the already-created lookahead eligibility audit,
  - CARLA's Python Transform class only for deterministic geometry, and
  - the same common/projection.py helper used by collect_data.py.

Current trajectory:
    offset = A * sin(2*pi*progress + phase_offset)

Candidate pre-freeze trial phases:
    trial_1 = -8 deg
    trial_2 = -4 deg
    trial_3 =  0 deg   <-- preselected Rain/Night trajectory if this set passes
    trial_4 = +4 deg
    trial_5 = +8 deg

The script validates:
  1) phase=0 reproduces the already-logged y=504 lane GT;
  2) the five trajectories have similar nuisance distributions over the
     current valid lane evaluation region;
  3) the projected y=504 GT traces remain comparable.

Outputs:
    results/pre_freeze/trajectory_phase_validation.csv
    results/pre_freeze/trajectory_phase_per_frame.csv
"""

import csv
import math
import sys
from pathlib import Path

import carla
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.projection import (
    build_camera_matrix,
    world_to_image,
    interpolate_x_at_row,
)

# -------------------------
# Frozen experiment values
# -------------------------
IMAGE_W = 1280
IMAGE_H = 720
FOV = 90.0
LOOKAHEAD_ROW = 504.0
LOOK_FORWARD_M = 60.0
VEHICLE_Z_OFFSET_M = 0.25
CAMERA_X = 1.5
CAMERA_Y = 0.0
CAMERA_Z = 1.7

AMPLITUDE_M = 0.75

# Candidate set to screen first.
PHASES_DEG = {
    "trial_1": -8.0,
    "trial_2": -4.0,
    "trial_3":  0.0,
    "trial_4":  4.0,
    "trial_5":  8.0,
}

ROUTE_CSV = ROOT / "data" / "routes" / "stop3_route.csv"
CLEAR_CSV = (
    ROOT
    / "data"
    / "recordings"
    / "stop3_clear_0_drift_0p75"
    / "frames.csv"
)
AUDIT_CSV = ROOT / "results" / "lane_marking_lookahead_audit.csv"

OUT_DIR = ROOT / "results" / "pre_freeze"
SUMMARY_CSV = OUT_DIR / "trajectory_phase_validation.csv"
PER_FRAME_CSV = OUT_DIR / "trajectory_phase_per_frame.csv"

K = build_camera_matrix(IMAGE_W, IMAGE_H, FOV)


def read_csv(path):
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_route():
    rows = read_csv(ROUTE_CSV)
    route = []
    for r in rows:
        route.append({
            "route_index": int(float(r["route_index"])),
            "route_s_m": float(r["route_s_m"]),
            "lane_x": float(r["lane_x"]),
            "lane_y": float(r["lane_y"]),
            "lane_z": float(r["lane_z"]),
            "pitch": float(r["pitch"]),
            "yaw": float(r["yaw"]),
            "roll": float(r["roll"]),
        })

    route.sort(key=lambda x: x["route_index"])
    return route


def eligible_ids():
    rows = read_csv(AUDIT_CSV)
    ids = set()

    for r in rows:
        sid = int(float(r["sample_id"]))
        junc = int(float(r["lookahead_is_junction"]))
        left_type = str(r["left_marking_type"]).strip().upper()
        right_type = str(r["right_marking_type"]).strip().upper()

        if (
            junc == 0
            and left_type != "NONE"
            and right_type != "NONE"
        ):
            ids.add(sid)

    return ids


def phase_pose(sample_id, total_samples, amplitude_m, phase_deg):
    """
    Exact current smooth_lateral_pose equation with one added frozen phase offset.
    """
    progress = (sample_id - 1) / (total_samples - 1)

    phase = (
        2.0 * np.pi * progress
        + np.deg2rad(phase_deg)
    )

    lateral_offset_m = (
        amplitude_m * np.sin(phase)
    )

    # Derive capture span from the actual logged sample spacing.
    # Filled later by caller because ROUTE_STEP can be inferred exactly.
    return float(lateral_offset_m), float(phase)


def phase_heading(amplitude_m, phase, capture_span_m):
    lateral_slope = (
        amplitude_m
        * (2.0 * np.pi / capture_span_m)
        * np.cos(phase)
    )

    heading_deg = np.degrees(
        np.arctan(lateral_slope)
    )

    return float(lateral_slope), float(heading_deg)


def vehicle_transform(route_point, lateral_offset_m, heading_offset_deg):
    yaw_rad = np.deg2rad(route_point["yaw"])

    right_x = -np.sin(yaw_rad)
    right_y = np.cos(yaw_rad)

    x = (
        route_point["lane_x"]
        + lateral_offset_m * right_x
    )
    y = (
        route_point["lane_y"]
        + lateral_offset_m * right_y
    )
    z = (
        route_point["lane_z"]
        + VEHICLE_Z_OFFSET_M
    )

    return carla.Transform(
        carla.Location(
            x=float(x),
            y=float(y),
            z=float(z),
        ),
        carla.Rotation(
            pitch=float(route_point["pitch"]),
            yaw=float(
                route_point["yaw"]
                + heading_offset_deg
            ),
            roll=float(route_point["roll"]),
        ),
    )


def camera_transform_from_vehicle(ego_tf):
    camera_location = ego_tf.transform(
        carla.Location(
            x=CAMERA_X,
            y=CAMERA_Y,
            z=CAMERA_Z,
        )
    )

    return carla.Transform(
        camera_location,
        carla.Rotation(
            pitch=float(ego_tf.rotation.pitch),
            yaw=float(ego_tf.rotation.yaw),
            roll=float(ego_tf.rotation.roll),
        ),
    )


def project_lane_gt_x(route, route_index, camera_tf):
    current_s = route[route_index]["route_s_m"]

    world_to_camera = np.array(
        camera_tf.get_inverse_matrix(),
        dtype=np.float64,
    )

    projected = []

    for idx in range(route_index, len(route)):
        point = route[idx]
        forward_distance = (
            point["route_s_m"] - current_s
        )

        if forward_distance > LOOK_FORWARD_M:
            break

        lane_location = carla.Location(
            x=point["lane_x"],
            y=point["lane_y"],
            z=point["lane_z"],
        )

        result = world_to_image(
            lane_location,
            world_to_camera,
            K,
        )

        if result is None:
            continue

        u, v, depth = result

        if (
            np.isfinite(u)
            and np.isfinite(v)
            and np.isfinite(depth)
        ):
            projected.append((u, v))

    if len(projected) < 2:
        return None

    x = interpolate_x_at_row(
        projected,
        LOOKAHEAD_ROW,
    )

    if x is None or not np.isfinite(x):
        return None

    return float(x)


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "range": float(np.ptp(a)),
        "mean": float(np.mean(a)),
        "mean_abs": float(np.mean(np.abs(a))),
        "std": float(np.std(a, ddof=0)),
    }


def main():
    route = load_route()
    clear_rows = read_csv(CLEAR_CSV)
    valid_ids = eligible_ids()

    print("=" * 86)
    print("OFFLINE FIVE-TRAJECTORY PHASE VALIDATION")
    print("=" * 86)
    print("Candidate phases:", PHASES_DEG)
    print("Preselected Rain/Night config: trial_3")
    print("Eligible lane IDs from frozen lookahead audit:", len(valid_ids))
    print("Eligible min/max:", min(valid_ids), max(valid_ids))

    if len(valid_ids) != 88:
        raise RuntimeError(
            f"Expected 88 current valid lane IDs, got {len(valid_ids)}"
        )

    # Map route_s -> route index robustly through nearest exact value.
    route_s = np.array(
        [r["route_s_m"] for r in route],
        dtype=np.float64,
    )

    # Infer sample spacing and capture span from actual logged recording.
    capture_s = np.array(
        [float(r["route_s_m"]) for r in clear_rows],
        dtype=np.float64,
    )
    if len(capture_s) < 2:
        raise RuntimeError("Clear recording has too few rows.")

    diffs = np.diff(capture_s)
    route_step_m = float(np.median(diffs))
    capture_span_m = float(
        (len(clear_rows) - 1) * route_step_m
    )

    print(f"Total logged samples: {len(clear_rows)}")
    print(f"Detected route step: {route_step_m:.6f} m")
    print(f"Capture span used by drift equation: {capture_span_m:.6f} m")
    print()

    all_frame_rows = []
    by_trial = {}

    for trial_name, phase_deg in PHASES_DEG.items():
        trial_rows = []

        for r in clear_rows:
            sid = int(float(r["sample_id"]))
            if sid not in valid_ids:
                continue

            s = float(r["route_s_m"])
            route_index = int(np.argmin(np.abs(route_s - s)))

            if abs(route_s[route_index] - s) > 1e-6:
                raise RuntimeError(
                    f"Could not exactly match route_s={s} for sample {sid}"
                )

            offset_m, phase = phase_pose(
                sid,
                len(clear_rows),
                AMPLITUDE_M,
                phase_deg,
            )
            slope, heading_deg = phase_heading(
                AMPLITUDE_M,
                phase,
                capture_span_m,
            )

            ego_tf = vehicle_transform(
                route[route_index],
                offset_m,
                heading_deg,
            )
            cam_tf = camera_transform_from_vehicle(
                ego_tf
            )

            gt_x = project_lane_gt_x(
                route,
                route_index,
                cam_tf,
            )

            if gt_x is None:
                raise RuntimeError(
                    f"{trial_name} sample {sid}: no projected y=504 GT"
                )

            row_out = {
                "trial": trial_name,
                "phase_deg": phase_deg,
                "sample_id": sid,
                "route_s_m": s,
                "lateral_offset_m": offset_m,
                "lateral_heading_offset_deg": heading_deg,
                "projected_lane_gt_x_y504": gt_x,
            }

            trial_rows.append(row_out)
            all_frame_rows.append(row_out)

        by_trial[trial_name] = trial_rows

    # ---------------------------------------------------------
    # Self-check: trial_3 phase 0 should reproduce logged GT.
    # ---------------------------------------------------------
    logged_gt = {}
    for r in clear_rows:
        sid = int(float(r["sample_id"]))
        text = str(r.get("lane_gt_x", "")).strip()
        if sid in valid_ids and text:
            logged_gt[sid] = float(text)

    diffs = []
    for r in by_trial["trial_3"]:
        sid = int(r["sample_id"])
        if sid in logged_gt:
            diffs.append(
                abs(
                    float(r["projected_lane_gt_x_y504"])
                    - logged_gt[sid]
                )
            )

    if not diffs:
        raise RuntimeError("Could not perform phase-0 GT self-check.")

    self_mean = float(np.mean(diffs))
    self_max = float(np.max(diffs))

    print("PHASE-0 PROJECTION SELF-CHECK")
    print(f"  mean |reprojected - logged GT|: {self_mean:.6f} px")
    print(f"  max  |reprojected - logged GT|: {self_max:.6f} px")

    # This is only a guard against implementation mismatch.
    if self_max > 0.50:
        raise RuntimeError(
            "Offline projection does not reproduce the current logged GT "
            "closely enough. Do not use this validator yet."
        )

    print("  PASS")
    print()

    summary = []

    for trial_name, phase_deg in PHASES_DEG.items():
        rows = by_trial[trial_name]

        offset_stats = stats(
            [r["lateral_offset_m"] for r in rows]
        )
        gt_stats = stats(
            [r["projected_lane_gt_x_y504"] for r in rows]
        )
        heading_stats = stats(
            [r["lateral_heading_offset_deg"] for r in rows]
        )

        summary.append({
            "trial": trial_name,
            "phase_deg": phase_deg,
            "n": len(rows),
            "offset_min_m": offset_stats["min"],
            "offset_max_m": offset_stats["max"],
            "offset_range_m": offset_stats["range"],
            "offset_mean_m": offset_stats["mean"],
            "offset_mean_abs_m": offset_stats["mean_abs"],
            "offset_std_m": offset_stats["std"],
            "heading_min_deg": heading_stats["min"],
            "heading_max_deg": heading_stats["max"],
            "heading_std_deg": heading_stats["std"],
            "gt_x_min_px": gt_stats["min"],
            "gt_x_max_px": gt_stats["max"],
            "gt_x_range_px": gt_stats["range"],
            "gt_x_mean_px": gt_stats["mean"],
            "gt_x_std_px": gt_stats["std"],
        })

    ref = next(x for x in summary if x["trial"] == "trial_3")

    for x in summary:
        x["delta_offset_range_pct_vs_t3"] = (
            100.0
            * (x["offset_range_m"] - ref["offset_range_m"])
            / ref["offset_range_m"]
        )
        x["delta_offset_std_pct_vs_t3"] = (
            100.0
            * (x["offset_std_m"] - ref["offset_std_m"])
            / ref["offset_std_m"]
        )
        x["delta_offset_mean_abs_pct_vs_t3"] = (
            100.0
            * (
                x["offset_mean_abs_m"]
                - ref["offset_mean_abs_m"]
            )
            / ref["offset_mean_abs_m"]
        )
        x["delta_offset_mean_m_vs_t3"] = (
            x["offset_mean_m"]
            - ref["offset_mean_m"]
        )
        x["delta_gt_x_range_pct_vs_t3"] = (
            100.0
            * (x["gt_x_range_px"] - ref["gt_x_range_px"])
            / ref["gt_x_range_px"]
        )
        x["delta_gt_x_std_pct_vs_t3"] = (
            100.0
            * (x["gt_x_std_px"] - ref["gt_x_std_px"])
            / ref["gt_x_std_px"]
        )
        x["delta_gt_x_mean_px_vs_t3"] = (
            x["gt_x_mean_px"]
            - ref["gt_x_mean_px"]
        )

    print("=" * 140)
    print("MATCHING TABLE — ACTUAL VALID LANE REGION")
    print("=" * 140)
    print(
        f"{'Trial':<8} {'Phase':>7} "
        f"{'OffRange':>9} {'OffMean':>9} {'|Off|':>8} {'OffStd':>8} "
        f"{'GT_Range':>9} {'GT_Mean':>9} {'GT_Std':>8} "
        f"{'dRange%':>8} {'dStd%':>8} {'dGT_R%':>8} {'dGT_Std%':>9}"
    )

    for x in summary:
        print(
            f"{x['trial']:<8} "
            f"{x['phase_deg']:>7.1f} "
            f"{x['offset_range_m']:>9.4f} "
            f"{x['offset_mean_m']:>9.4f} "
            f"{x['offset_mean_abs_m']:>8.4f} "
            f"{x['offset_std_m']:>8.4f} "
            f"{x['gt_x_range_px']:>9.2f} "
            f"{x['gt_x_mean_px']:>9.2f} "
            f"{x['gt_x_std_px']:>8.2f} "
            f"{x['delta_offset_range_pct_vs_t3']:>8.2f} "
            f"{x['delta_offset_std_pct_vs_t3']:>8.2f} "
            f"{x['delta_gt_x_range_pct_vs_t3']:>8.2f} "
            f"{x['delta_gt_x_std_pct_vs_t3']:>9.2f}"
        )

    print()
    print("MEAN SHIFTS RELATIVE TO TRIAL 3")
    for x in summary:
        print(
            f"  {x['trial']}: "
            f"offset mean shift={x['delta_offset_mean_m_vs_t3']:+.4f} m, "
            f"GT-x mean shift={x['delta_gt_x_mean_px_vs_t3']:+.2f} px"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    with PER_FRAME_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_frame_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_frame_rows)

    print()
    print("Saved:")
    print(" ", SUMMARY_CSV)
    print(" ", PER_FRAME_CSV)
    print()
    print(
        "Do not record the final sweep yet. Send this table first; "
        "we will either freeze these five phases or redesign them."
    )


if __name__ == "__main__":
    main()
