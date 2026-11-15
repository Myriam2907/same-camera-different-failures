#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SELECTION = (
    ROOT
    / "results"
    / "final"
    / "stop_matched_25m"
    / "preregistered_25m_frames.csv"
)

LANE_FRAME = (
    ROOT
    / "results"
    / "final"
    / "lane_ufld_range"
    / "final_per_frame.csv"
)

STOP_FRAME = (
    ROOT
    / "results"
    / "final"
    / "stop_matched_25m"
    / "matched_25m_per_frame.csv"
)

OUT = (
    ROOT
    / "results"
    / "final"
    / "same_frame_25m"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEVERITIES = [0, 15, 30, 45, 60, 75, 90]
TRIALS = [1, 2, 3, 4, 5]

EXPECTED_INTERSECTION = {103, 104, 105}


# ============================================================
# LOAD
# ============================================================

selection = pd.read_csv(SELECTION)
lane = pd.read_csv(LANE_FRAME)
stop = pd.read_csv(STOP_FRAME)

lane25 = lane[
    lane["row_label"] == "target_25m"
].copy()


# ============================================================
# FIND GEOMETRY-ONLY INTERSECTION
# ============================================================

intersection_rows = []

print("=" * 110)
print("FIXED SAME-FRAME 25 m INTERSECTION")
print("=" * 110)

for trial in TRIALS:

    frozen_ids = set(
        selection[
            selection["trial"] == trial
        ]["sample_id"]
        .astype(int)
    )

    # Lane geometry eligibility does not depend on weather.
    clear_lane_ids = set(
        lane25[
            (lane25["severity"] == 0)
            & (lane25["trial"] == trial)
        ]["sample_id"]
        .astype(int)
    )

    keep = frozen_ids & clear_lane_ids

    print(
        f"trial {trial}: "
        f"frozen={sorted(frozen_ids)} "
        f"-> intersection={sorted(keep)}"
    )

    if keep != EXPECTED_INTERSECTION:
        raise RuntimeError(
            f"Unexpected intersection for trial {trial}: "
            f"{sorted(keep)}"
        )

    part = selection[
        (selection["trial"] == trial)
        & (selection["sample_id"].isin(sorted(keep)))
    ].copy()

    intersection_rows.append(part)


intersection = pd.concat(
    intersection_rows,
    ignore_index=True,
)

intersection_path = (
    OUT
    / "frozen_same_frame_intersection.csv"
)

intersection.to_csv(
    intersection_path,
    index=False,
)


# ============================================================
# PHYSICAL RANGE SUMMARY
# ============================================================

print()
print("=" * 110)
print("PHYSICAL RANGE OF FIXED INTERSECTION")
print("=" * 110)

print(
    "STOP range:",
    f"{intersection['distance_to_sign_m'].min():.3f}",
    "to",
    f"{intersection['distance_to_sign_m'].max():.3f}",
    "m",
)

print(
    "STOP median:",
    f"{intersection['distance_to_sign_m'].median():.3f} m",
)


# ============================================================
# LANE SAME-FRAME RESULTS
# ============================================================

lane_rows = []

for severity in SEVERITIES:

    for trial in TRIALS:

        ids = set(
            intersection[
                intersection["trial"] == trial
            ]["sample_id"]
            .astype(int)
        )

        g = lane25[
            (lane25["severity"] == severity)
            & (lane25["trial"] == trial)
            & (lane25["sample_id"].isin(ids))
        ].copy()

        if len(g) != 3:
            raise RuntimeError(
                f"Lane severity={severity}, trial={trial}: "
                f"expected 3 rows, got {len(g)}"
            )

        success = (
            pd.to_numeric(
                g["success"],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

        cond = pd.to_numeric(
            g["conditional_error_m"],
            errors="coerce",
        )

        successful_error = cond[
            success == 1
        ].dropna()

        lane_rows.append(
            {
                "severity": severity,
                "trial": trial,
                "n_frames": 3,
                "n_success": int(success.sum()),
                "availability_pct":
                    100.0 * success.mean(),

                "conditional_mae_m":
                    (
                        float(successful_error.mean())
                        if len(successful_error)
                        else np.nan
                    ),

                "lane_camera_range_mean_m":
                    float(
                        g["camera_range_m"].mean()
                    ),
            }
        )


lane_trial = pd.DataFrame(
    lane_rows
)

lane_summary = (
    lane_trial
    .groupby(
        "severity",
        as_index=False,
    )
    .agg(
        lane_availability_mean_pct=(
            "availability_pct",
            "mean",
        ),

        lane_availability_std_pct=(
            "availability_pct",
            "std",
        ),

        lane_conditional_mae_m_mean=(
            "conditional_mae_m",
            "mean",
        ),

        lane_conditional_mae_m_std=(
            "conditional_mae_m",
            "std",
        ),

        lane_conditional_configs_n=(
            "conditional_mae_m",
            "count",
        ),

        lane_camera_range_mean_m=(
            "lane_camera_range_mean_m",
            "mean",
        ),
    )
)


# ============================================================
# STOP SAME-FRAME RESULTS
# ============================================================

stop_rows = []

for detector in sorted(
    stop["detector"].unique()
):

    for severity in SEVERITIES:

        for trial in TRIALS:

            ids = set(
                intersection[
                    intersection["trial"] == trial
                ]["sample_id"]
                .astype(int)
            )

            g = stop[
                (stop["detector"] == detector)
                & (stop["severity"] == severity)
                & (stop["trial"] == trial)
                & (stop["sample_id"].isin(ids))
            ].copy()

            if len(g) != 3:
                raise RuntimeError(
                    f"{detector}, severity={severity}, "
                    f"trial={trial}: expected 3 rows, "
                    f"got {len(g)}"
                )

            output = (
                pd.to_numeric(
                    g["output"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

            correct = (
                pd.to_numeric(
                    g["correct_detection"],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

            if (correct > output).any():
                raise RuntimeError(
                    f"{detector}: correct detection "
                    f"without output."
                )

            stop_rows.append(
                {
                    "detector": detector,
                    "severity": severity,
                    "trial": trial,
                    "n_frames": 3,

                    "output_availability_pct":
                        100.0 * output.mean(),

                    "correct_detection_pct":
                        100.0 * correct.mean(),
                }
            )


stop_trial = pd.DataFrame(
    stop_rows
)

stop_summary = (
    stop_trial
    .groupby(
        [
            "detector",
            "severity",
        ],
        as_index=False,
    )
    .agg(
        output_availability_mean_pct=(
            "output_availability_pct",
            "mean",
        ),

        output_availability_std_pct=(
            "output_availability_pct",
            "std",
        ),

        correct_detection_mean_pct=(
            "correct_detection_pct",
            "mean",
        ),

        correct_detection_std_pct=(
            "correct_detection_pct",
            "std",
        ),
    )
)


# ============================================================
# BUILD COMPARISON
# ============================================================

output_wide = stop_summary.pivot(
    index="severity",
    columns="detector",
    values="output_availability_mean_pct",
)

correct_wide = stop_summary.pivot(
    index="severity",
    columns="detector",
    values="correct_detection_mean_pct",
)

comparison = (
    lane_summary
    .set_index("severity")
    .copy()
)

for detector in [
    "HSV",
    "Template",
    "YOLO",
    "RF-DETR",
]:

    comparison[
        f"{detector}_output_pct"
    ] = output_wide[detector]

    comparison[
        f"{detector}_correct_pct"
    ] = correct_wide[detector]


comparison = comparison.reset_index()


# ============================================================
# SAVE
# ============================================================

lane_trial_path = (
    OUT
    / "intersection_lane_per_trial.csv"
)

lane_summary_path = (
    OUT
    / "intersection_lane_summary.csv"
)

stop_trial_path = (
    OUT
    / "intersection_stop_per_trial.csv"
)

stop_summary_path = (
    OUT
    / "intersection_stop_summary.csv"
)

comparison_path = (
    OUT
    / "intersection_lane_stop_comparison.csv"
)

lane_trial.to_csv(
    lane_trial_path,
    index=False,
)

lane_summary.to_csv(
    lane_summary_path,
    index=False,
)

stop_trial.to_csv(
    stop_trial_path,
    index=False,
)

stop_summary.to_csv(
    stop_summary_path,
    index=False,
)

comparison.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 120)
print("LANE ON EXACT SAME 3 FRAMES / CONFIGURATION")
print("=" * 120)

print(
    lane_summary.to_string(
        index=False
    )
)


print()
print("=" * 120)
print("SAME-FRAME OUTPUT AVAILABILITY")
print("=" * 120)

cols = [
    "severity",
    "lane_availability_mean_pct",
    "HSV_output_pct",
    "Template_output_pct",
    "YOLO_output_pct",
    "RF-DETR_output_pct",
]

print(
    comparison[
        cols
    ].to_string(
        index=False
    )
)


print()
print("=" * 120)
print("SAME-FRAME STOP CORRECT DETECTION")
print("=" * 120)

cols = [
    "severity",
    "HSV_correct_pct",
    "Template_correct_pct",
    "YOLO_correct_pct",
    "RF-DETR_correct_pct",
]

print(
    comparison[
        cols
    ].to_string(
        index=False
    )
)


print()
print("=" * 120)
print("SAVED")
print("=" * 120)

print(intersection_path)
print(lane_trial_path)
print(lane_summary_path)
print(stop_trial_path)
print(stop_summary_path)
print(comparison_path)

print()
print(
    "NOTE: This is a post-hoc supplementary "
    "same-frame control."
)

print(
    "No replacement frames were selected."
)

print(
    "Fixed intersection = {103, 104, 105} "
    "for every configuration."
)
