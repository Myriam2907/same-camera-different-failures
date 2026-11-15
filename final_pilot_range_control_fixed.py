#!/usr/bin/env python3
"""
Final pilot matched-distance analysis for the Fog-only CARLA study.

Uses EXISTING outputs only:
  - results/sign_hsv_iou/*_per_frame.csv
  - results/sign_template_iou/*_per_frame.csv
  - results/sign_yolo_iou/*_per_frame.csv + *_raw_boxes.csv
  - results/sign_rfdetr_iou/*_per_frame.csv + *_raw_boxes.csv
  - results/pilot/ufld_multirange_summary.csv

Deep STOP thresholds are the already-frozen pilot thresholds:
  YOLO26s   : 0.041
  RF-DETR-L : 0.043

It does NOT change the preregistered R_sign definition.
It does NOT change the primary y=504 R_lane definition.
The matched-distance analysis is a SECONDARY range-control analysis.

Outputs:
  results/pilot/final_stop_calibrated_summary.csv
  results/pilot/final_stop_within40m.csv
  results/pilot/final_stop_matched_distance_bins.csv
  results/pilot/final_cross_task_matched_distance.csv

  results/pilot/final_stop_within40m.png
  results/pilot/final_cross_task_fog90.png
  results/pilot/final_cross_task_retention_fog90.png
"""

from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "pilot"
OUT.mkdir(parents=True, exist_ok=True)

SEVERITIES = [0, 15, 30, 45, 60, 75, 90]
IOU_THRESHOLD = 0.50
WITHIN40_M = 40.0

# Canonical pilot recordings. Deep analysis uses these frames.csv files
# directly for sample_id / route / physical sign distance / GT availability.
# This avoids depending on a detector-specific *_per_frame.csv file.
RECORDINGS = {
    0: ROOT / "data" / "recordings" / "stop3_clear_0_drift_0p75",
    15: ROOT / "data" / "recordings" / "stop3_fog_15_drift_0p75",
    30: ROOT / "data" / "recordings" / "stop3_fog_30_drift_0p75",
    45: ROOT / "data" / "recordings" / "stop3_fog_45_drift_0p75",
    60: ROOT / "data" / "recordings" / "stop3_fog_60_drift_0p75",
    75: ROOT / "data" / "recordings" / "stop3_fog_75_drift_0p75",
    90: ROOT / "data" / "recordings" / "stop3_fog_90_drift_0p75",
}

DETECTORS = {
    "HSV": {
        "root": RESULTS / "sign_hsv_iou",
        "kind": "per_frame",
        "threshold": None,
    },
    "Template": {
        "root": RESULTS / "sign_template_iou",
        "kind": "per_frame",
        "threshold": None,
    },
    "YOLO": {
        "root": RESULTS / "sign_yolo_iou",
        "kind": "raw_boxes",
        "threshold": 0.041,
    },
    "RF-DETR": {
        "root": RESULTS / "sign_rfdetr_iou",
        "kind": "raw_boxes",
        "threshold": 0.043,
    },
}

UFLD_SUMMARY = OUT / "ufld_multirange_summary.csv"

EXPECTED_DEEP_WITHIN40 = {
    "YOLO": {
        0: 100.00,
        15: 97.06,
        30: 94.12,
        45: 100.00,
        60: 97.06,
        75: 92.65,
        90: 83.82,
    },
    "RF-DETR": {
        0: 100.00,
        15: 100.00,
        30: 95.59,
        45: 100.00,
        60: 100.00,
        75: 100.00,
        90: 95.59,
    },
}

def dataset_name(sev):
    return "clear_drift" if sev == 0 else f"fog{sev}_drift"

def require(path):
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path}")
    return path

def choose_col(df, candidates, label, path):
    lower = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    # flexible fallback
    for actual in df.columns:
        a = actual.lower()
        if label == "confidence":
            if ("conf" in a or "score" in a) and "class" not in a:
                return actual
        elif label == "iou":
            if "iou" in a and "threshold" not in a:
                return actual

    raise RuntimeError(
        f"Could not identify {label} column in {path}.\n"
        f"Columns are:\n{list(df.columns)}"
    )

def locate_raw_boxes(root, stem):
    exact = root / f"{stem}_raw_boxes.csv"
    if exact.exists():
        return exact

    candidates = list(root.glob(f"{stem}*raw*box*.csv"))
    if len(candidates) == 1:
        return candidates[0]

    # broader fallback
    candidates = [
        p for p in root.glob("*.csv")
        if stem in p.stem
        and "raw" in p.stem.lower()
        and "box" in p.stem.lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise RuntimeError(
            f"No raw-box CSV found for {stem} under {root}.\n"
            f"Available CSVs:\n" +
            "\n".join(f"  {p.name}" for p in sorted(root.glob("*.csv")))
        )

    raise RuntimeError(
        f"Multiple raw-box CSV candidates for {stem}:\n" +
        "\n".join(f"  {p}" for p in candidates)
    )

def load_metadata_per_frame(root, sev):
    stem = dataset_name(sev)
    path = require(root / f"{stem}_per_frame.csv")
    df = pd.read_csv(path)

    required = {"sample_id", "route_s_m", "distance_to_sign_m"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{path} missing columns: {sorted(missing)}")

    out = df.copy()
    out["sample_id"] = pd.to_numeric(out["sample_id"], errors="raise").astype(int)
    out["route_s_m"] = pd.to_numeric(out["route_s_m"], errors="coerce")
    out["distance_to_sign_m"] = pd.to_numeric(
        out["distance_to_sign_m"], errors="coerce"
    )

    if "sign_in_view" in out.columns:
        out["sign_in_view"] = pd.to_numeric(
            out["sign_in_view"], errors="coerce"
        ).fillna(0).astype(int)
        out = out[out["sign_in_view"] == 1].copy()

    if "gt_available" in out.columns:
        out["gt_available"] = pd.to_numeric(
            out["gt_available"], errors="coerce"
        ).fillna(0).astype(int)
        out = out[out["gt_available"] == 1].copy()

    out = out[
        (out["route_s_m"] <= 0.0)
        & np.isfinite(out["distance_to_sign_m"])
    ].copy()

    return out, path


def load_canonical_recording_metadata(sev):
    root = RECORDINGS[sev]
    path = require(root / "frames.csv")
    df = pd.read_csv(path)

    required = {
        "sample_id", "route_s_m", "distance_to_sign_m", "sign_in_view"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"{path} missing canonical metadata columns: {sorted(missing)}"
        )

    out = df.copy()
    out["sample_id"] = pd.to_numeric(
        out["sample_id"], errors="raise"
    ).astype(int)
    out["route_s_m"] = pd.to_numeric(
        out["route_s_m"], errors="coerce"
    )
    out["distance_to_sign_m"] = pd.to_numeric(
        out["distance_to_sign_m"], errors="coerce"
    )
    out["sign_in_view"] = pd.to_numeric(
        out["sign_in_view"], errors="coerce"
    ).fillna(0).astype(int)

    out = out[
        (out["route_s_m"] <= 0.0)
        & (out["sign_in_view"] == 1)
        & np.isfinite(out["distance_to_sign_m"])
    ].copy()

    return out, path


def build_classical_frame_table(detector, cfg, sev):
    meta, path = load_metadata_per_frame(cfg["root"], sev)

    if "correct_detection" not in meta.columns:
        raise RuntimeError(
            f"{path} has no correct_detection column."
        )

    meta["correct"] = pd.to_numeric(
        meta["correct_detection"], errors="coerce"
    ).fillna(0).astype(int)

    # If these exist, retain them for diagnostics.
    if "false_positive" in meta.columns:
        meta["fp"] = pd.to_numeric(
            meta["false_positive"], errors="coerce"
        ).fillna(0).astype(int)
    else:
        meta["fp"] = 0

    return meta[
        ["sample_id", "route_s_m", "distance_to_sign_m", "correct", "fp"]
    ].copy()

def build_deep_frame_table(detector, cfg, sev):
    # Use recorder metadata directly. The raw-box CSV is all we need from
    # the detector output to reapply the frozen confidence threshold.
    meta, meta_path = load_canonical_recording_metadata(sev)
    stem = dataset_name(sev)
    raw_path = locate_raw_boxes(cfg["root"], stem)
    raw = pd.read_csv(raw_path)

    if "sample_id" not in raw.columns:
        raise RuntimeError(
            f"{raw_path} has no sample_id column.\nColumns: {list(raw.columns)}"
        )

    conf_col = choose_col(
        raw,
        ["confidence", "conf", "score", "raw_confidence",
         "raw_conf", "box_confidence"],
        "confidence",
        raw_path,
    )
    iou_col = choose_col(
        raw,
        ["iou", "gt_iou", "iou_with_gt", "box_iou"],
        "iou",
        raw_path,
    )

    raw["sample_id"] = pd.to_numeric(
        raw["sample_id"], errors="coerce"
    ).astype("Int64")
    raw[conf_col] = pd.to_numeric(raw[conf_col], errors="coerce")
    raw[iou_col] = pd.to_numeric(raw[iou_col], errors="coerce")

    raw = raw.dropna(subset=["sample_id"]).copy()
    raw["sample_id"] = raw["sample_id"].astype(int)

    threshold = float(cfg["threshold"])
    accepted = raw[raw[conf_col] >= threshold].copy()

    # Correct frame = at least one accepted STOP box with IoU >= 0.50.
    correct_ids = set(
        accepted.loc[
            accepted[iou_col] >= IOU_THRESHOLD,
            "sample_id",
        ].astype(int)
    )

    # Frame-level FP diagnostic:
    # at least one accepted box but no correct accepted box.
    accepted_ids = set(accepted["sample_id"].astype(int))
    fp_ids = accepted_ids - correct_ids

    out = meta[
        ["sample_id", "route_s_m", "distance_to_sign_m"]
    ].copy()

    out["correct"] = out["sample_id"].isin(correct_ids).astype(int)
    out["fp"] = out["sample_id"].isin(fp_ids).astype(int)

    print(
        f"{detector:7s} fog{sev:>2}: threshold={threshold:.3f} | "
        f"raw={raw_path.name} | conf={conf_col} | iou={iou_col}"
    )

    return out

def build_all_frame_tables():
    tables = {}

    for detector, cfg in DETECTORS.items():
        if not cfg["root"].exists():
            raise RuntimeError(f"Missing result directory: {cfg['root']}")

        for sev in SEVERITIES:
            if cfg["kind"] == "per_frame":
                t = build_classical_frame_table(detector, cfg, sev)
            else:
                t = build_deep_frame_table(detector, cfg, sev)

            if len(t) != 120:
                print(
                    f"WARNING: {detector} severity {sev} has {len(t)} "
                    f"evaluable approach frames, expected 120."
                )

            tables[(detector, sev)] = t

    return tables

def summarize_stop(tables):
    rows = []

    for detector in DETECTORS:
        clear_D = None

        for sev in SEVERITIES:
            t = tables[(detector, sev)].sort_values("sample_id").copy()
            correct = t[t["correct"] == 1]

            if len(correct):
                first = correct.iloc[0]
                D = float(first["distance_to_sign_m"])
                first_sid = int(first["sample_id"])
            else:
                D = 0.0
                first_sid = None

            if sev == 0:
                clear_D = D

            if clear_D is None or clear_D <= 0:
                R = np.nan
            else:
                R = min(1.0, D / clear_D)

            w = t[t["distance_to_sign_m"] <= WITHIN40_M].copy()

            rows.append({
                "detector": detector,
                "severity": sev,
                "threshold": DETECTORS[detector]["threshold"],
                "approach_n": len(t),
                "approach_correct_n": int(t["correct"].sum()),
                "approach_correct_rate_pct": 100.0 * t["correct"].mean(),
                "within40_n": len(w),
                "within40_correct_n": int(w["correct"].sum()),
                "within40_correct_rate_pct": 100.0 * w["correct"].mean(),
                "first_correct_sample_id": first_sid,
                "D_sign_m": D,
                "R_sign": R,
                "fp_frames": int(t["fp"].sum()),
            })

    return pd.DataFrame(rows)

def matched_distance_bins(tables):
    require(UFLD_SUMMARY)
    lane = pd.read_csv(UFLD_SUMMARY)

    needed = {
        "severity", "row_y", "availability_pct",
        "distance_median_m", "mae_m_success_only",
    }
    missing = needed - set(lane.columns)
    if missing:
        raise RuntimeError(
            f"{UFLD_SUMMARY} missing columns: {sorted(missing)}"
        )

    # Clear row order by physical distance.
    clear_lane = (
        lane[lane["severity"] == 0]
        .sort_values("distance_median_m")
        .copy()
    )

    targets = clear_lane["distance_median_m"].to_numpy(dtype=float)
    rows_y = clear_lane["row_y"].astype(int).to_numpy()

    if len(targets) != 5:
        raise RuntimeError(
            f"Expected 5 lane distance targets, found {len(targets)}."
        )

    # Non-overlapping nearest-target bins:
    # boundaries are halfway between neighboring lane target distances.
    mids = 0.5 * (targets[:-1] + targets[1:])
    left_edge = targets[0] - 0.5 * (targets[1] - targets[0])
    right_edge = targets[-1] + 0.5 * (targets[-1] - targets[-2])
    edges = np.concatenate([[left_edge], mids, [right_edge]])

    stop_rows = []

    for detector in DETECTORS:
        for sev in SEVERITIES:
            t = tables[(detector, sev)].copy()

            for i, target in enumerate(targets):
                lo = float(edges[i])
                hi = float(edges[i + 1])

                if i == len(targets) - 1:
                    z = t[
                        (t["distance_to_sign_m"] >= lo)
                        & (t["distance_to_sign_m"] <= hi)
                    ].copy()
                else:
                    z = t[
                        (t["distance_to_sign_m"] >= lo)
                        & (t["distance_to_sign_m"] < hi)
                    ].copy()

                stop_rows.append({
                    "detector": detector,
                    "severity": sev,
                    "row_y_match": int(rows_y[i]),
                    "lane_target_distance_m": float(target),
                    "bin_lo_m": lo,
                    "bin_hi_m": hi,
                    "stop_n": len(z),
                    "stop_correct_n": int(z["correct"].sum()),
                    "stop_correct_rate_pct": (
                        100.0 * z["correct"].mean() if len(z) else np.nan
                    ),
                    "stop_actual_distance_mean_m": (
                        float(z["distance_to_sign_m"].mean())
                        if len(z) else np.nan
                    ),
                    "stop_actual_distance_median_m": (
                        float(z["distance_to_sign_m"].median())
                        if len(z) else np.nan
                    ),
                })

    stop_bins = pd.DataFrame(stop_rows)

    # Add clear-relative retention for STOP.
    clear_stop = stop_bins[
        stop_bins["severity"] == 0
    ][
        ["detector", "row_y_match", "stop_correct_rate_pct"]
    ].rename(
        columns={"stop_correct_rate_pct": "stop_clear_rate_pct"}
    )

    stop_bins = stop_bins.merge(
        clear_stop,
        on=["detector", "row_y_match"],
        how="left",
    )

    stop_bins["stop_retention_vs_clear"] = np.where(
        stop_bins["stop_clear_rate_pct"] > 0,
        stop_bins["stop_correct_rate_pct"]
        / stop_bins["stop_clear_rate_pct"],
        np.nan,
    )

    # Lane table + clear-relative availability retention.
    lane2 = lane.copy()
    clear_av = lane2[lane2["severity"] == 0][
        ["row_y", "availability_pct"]
    ].rename(columns={"availability_pct": "lane_clear_availability_pct"})

    lane2 = lane2.merge(clear_av, on="row_y", how="left")
    lane2["lane_availability_retention_vs_clear"] = np.where(
        lane2["lane_clear_availability_pct"] > 0,
        lane2["availability_pct"]
        / lane2["lane_clear_availability_pct"],
        np.nan,
    )

    # Cross-task table uses deep STOP models for the central comparison.
    cross = stop_bins[
        stop_bins["detector"].isin(["YOLO", "RF-DETR"])
    ].merge(
        lane2[
            [
                "severity", "row_y", "distance_median_m",
                "availability_pct", "lane_clear_availability_pct",
                "lane_availability_retention_vs_clear",
                "mae_m_success_only",
            ]
        ],
        left_on=["severity", "row_y_match"],
        right_on=["severity", "row_y"],
        how="left",
    )

    return stop_bins, cross, lane2, edges, targets

def validate_expected_deep(summary):
    print("\n" + "=" * 100)
    print("DEEP <=40 m REPRODUCTION CHECK")
    print("=" * 100)

    bad = []

    for detector, expected_map in EXPECTED_DEEP_WITHIN40.items():
        for sev, expected in expected_map.items():
            q = summary[
                (summary["detector"] == detector)
                & (summary["severity"] == sev)
            ]

            if len(q) != 1:
                bad.append((detector, sev, "missing"))
                continue

            got = float(q.iloc[0]["within40_correct_rate_pct"])
            delta = got - expected
            print(
                f"{detector:7s} fog{sev:>2}: "
                f"got={got:6.2f}% expected={expected:6.2f}% "
                f"delta={delta:+.3f} pp"
            )

            if abs(delta) > 0.06:
                bad.append((detector, sev, delta))

    if bad:
        print("\nWARNING: calibrated deep <=40m results did not reproduce "
              "the earlier pilot table exactly.")
        print("Do NOT use the generated deep figures until the discrepancy "
              "is understood.")
        print("Mismatches:", bad)
        return False

    print("\nPASS: calibrated YOLO/RF-DETR <=40m results reproduce the "
          "earlier pilot table.")
    return True

def make_figures(summary, stop_bins, cross, lane2):
    # 1) Fair <=40m all-detector figure.
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for det in ["HSV", "Template", "YOLO", "RF-DETR"]:
        z = summary[summary["detector"] == det].sort_values("severity")
        ax.plot(
            z["severity"],
            z["within40_correct_rate_pct"],
            marker="o",
            label=det,
        )
    ax.set_xlabel("Fog severity")
    ax.set_ylabel("Correct STOP detection rate within 40 m (%)")
    ax.set_title("STOP-sign detection — identical ≤40 m evaluation region")
    ax.set_xticks(SEVERITIES)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "final_stop_within40m.png", dpi=180)
    plt.close(fig)

    # 2) Central matched-distance Fog90: comparable success percentages.
    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    l90 = lane2[lane2["severity"] == 90].sort_values("distance_median_m")
    ax.plot(
        l90["distance_median_m"],
        l90["availability_pct"],
        marker="o",
        linewidth=2,
        label="UFLD-v2 lane availability",
    )

    for det in ["YOLO", "RF-DETR"]:
        z = cross[
            (cross["severity"] == 90)
            & (cross["detector"] == det)
        ].sort_values("lane_target_distance_m")
        ax.plot(
            z["lane_target_distance_m"],
            z["stop_correct_rate_pct"],
            marker="o",
            linewidth=2,
            label=f"{det} STOP correct rate",
        )

    ax.set_xlabel("Physical camera-to-target distance (m)")
    ax.set_ylabel("Frame-level task success (%)")
    ax.set_title("Fog90 — task success at matched physical distance")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "final_cross_task_fog90.png", dpi=180)
    plt.close(fig)

    # 3) Normalized retention versus each task's own Clear baseline.
    # Secondary range-control only; not preregistered reliability.
    fig, ax = plt.subplots(figsize=(9.2, 5.6))

    ax.plot(
        l90["distance_median_m"],
        l90["lane_availability_retention_vs_clear"],
        marker="o",
        linewidth=2,
        label="UFLD availability retention",
    )

    for det in ["YOLO", "RF-DETR"]:
        z = cross[
            (cross["severity"] == 90)
            & (cross["detector"] == det)
        ].sort_values("lane_target_distance_m")
        ax.plot(
            z["lane_target_distance_m"],
            z["stop_retention_vs_clear"],
            marker="o",
            linewidth=2,
            label=f"{det} STOP-rate retention",
        )

    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Physical camera-to-target distance (m)")
    ax.set_ylabel("Fog90 / Clear task-success ratio")
    ax.set_title("Fog90 — matched-distance degradation relative to Clear")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUT / "final_cross_task_retention_fog90.png",
        dpi=180,
    )
    plt.close(fig)

def main():
    print("=" * 100)
    print("FINAL PILOT RANGE-CONTROL ANALYSIS")
    print("=" * 100)
    print("Deep thresholds:")
    print("  YOLO    = 0.041")
    print("  RF-DETR = 0.043")
    print("Correct STOP detection: accepted box with IoU >= 0.50")
    print()

    tables = build_all_frame_tables()
    summary = summarize_stop(tables)

    summary_path = OUT / "final_stop_calibrated_summary.csv"
    summary.to_csv(summary_path, index=False)

    within = summary[
        [
            "detector", "severity", "threshold",
            "within40_n", "within40_correct_n",
            "within40_correct_rate_pct",
            "D_sign_m", "R_sign",
        ]
    ].copy()

    within_path = OUT / "final_stop_within40m.csv"
    within.to_csv(within_path, index=False)

    print("\n" + "=" * 100)
    print("FAIR <=40 m STOP CORRECT-DETECTION RATE (%)")
    print("=" * 100)
    p = within.pivot(
        index="severity",
        columns="detector",
        values="within40_correct_rate_pct",
    ).reindex(SEVERITIES)
    order = [c for c in ["HSV", "Template", "YOLO", "RF-DETR"] if c in p.columns]
    print(p[order].round(2).to_string())

    print("\n" + "=" * 100)
    print("PREREGISTERED FIRST-DETECTION-DISTANCE RELIABILITY")
    print("=" * 100)
    r = within.pivot(
        index="severity",
        columns="detector",
        values="R_sign",
    ).reindex(SEVERITIES)
    print(r[order].round(4).to_string())

    deep_ok = validate_expected_deep(summary)

    stop_bins, cross, lane2, edges, targets = matched_distance_bins(tables)

    stop_bins_path = OUT / "final_stop_matched_distance_bins.csv"
    cross_path = OUT / "final_cross_task_matched_distance.csv"
    stop_bins.to_csv(stop_bins_path, index=False)
    cross.to_csv(cross_path, index=False)

    print("\n" + "=" * 100)
    print("MATCHED PHYSICAL-DISTANCE BINS")
    print("=" * 100)
    for i, target in enumerate(targets):
        print(
            f"lane target {target:6.2f} m : "
            f"STOP bin [{edges[i]:.2f}, {edges[i+1]:.2f}] m"
        )

    print("\n" + "=" * 100)
    print("FOG90 MATCHED-DISTANCE CENTRAL TABLE")
    print("=" * 100)

    fog90_rows = []
    l90 = lane2[lane2["severity"] == 90].copy()

    for _, lr in l90.sort_values("distance_median_m").iterrows():
        row_y = int(lr["row_y"])
        d = float(lr["distance_median_m"])

        out = {
            "distance_m": d,
            "row_y": row_y,
            "UFLD_availability_pct": float(lr["availability_pct"]),
            "UFLD_availability_retention": float(
                lr["lane_availability_retention_vs_clear"]
            ),
            "UFLD_success_only_MAE_m": float(lr["mae_m_success_only"]),
        }

        for det in ["YOLO", "RF-DETR"]:
            q = cross[
                (cross["severity"] == 90)
                & (cross["detector"] == det)
                & (cross["row_y_match"] == row_y)
            ]
            if len(q) == 1:
                out[f"{det}_STOP_rate_pct"] = float(
                    q.iloc[0]["stop_correct_rate_pct"]
                )
                out[f"{det}_retention"] = float(
                    q.iloc[0]["stop_retention_vs_clear"]
                )
                out[f"{det}_n"] = int(q.iloc[0]["stop_n"])

        fog90_rows.append(out)

    fog90_table = pd.DataFrame(fog90_rows)
    print(fog90_table.round(4).to_string(index=False))

    if deep_ok:
        make_figures(summary, stop_bins, cross, lane2)
        print("\nFigures generated.")
    else:
        print("\nFigures NOT generated because the calibrated deep reproduction "
              "check failed.")

    print("\nSaved:")
    print(" ", summary_path)
    print(" ", within_path)
    print(" ", stop_bins_path)
    print(" ", cross_path)

    if deep_ok:
        print(" ", OUT / "final_stop_within40m.png")
        print(" ", OUT / "final_cross_task_fog90.png")
        print(" ", OUT / "final_cross_task_retention_fog90.png")

    print()
    print("Interpretation guard:")
    print(
        "  - R_sign remains the first-correct-detection-distance ratio."
    )
    print(
        "  - matched-distance retention is SECONDARY range-control analysis,"
    )
    print(
        "    not a replacement for preregistered R_sign or primary y=504 R_lane."
    )
    print(
        "  - success-only lane MAE must be read together with availability "
        "because of survivor bias."
    )

if __name__ == "__main__":
    main()
