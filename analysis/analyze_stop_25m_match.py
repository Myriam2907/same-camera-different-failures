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

OUT = (
    ROOT
    / "results"
    / "final"
    / "stop_matched_25m"
)

LANE_SUMMARY = (
    ROOT
    / "results"
    / "final"
    / "lane_ufld_range"
    / "posthoc_split_metrics_summary.csv"
)

SEVERITIES = [0, 15, 30, 45, 60, 75, 90]
TRIALS = [1, 2, 3, 4, 5]


# ============================================================
# FROZEN DETECTOR DEFINITIONS
# ============================================================

DETECTORS = {
    "HSV": {
        "root": ROOT / "results" / "final" / "stop_hsv",
        "suffix": "_per_frame.csv",
        "output_col": "raw_detected",
        "correct_col": "correct_detection",
        "iou_col": "iou",
    },

    "Template": {
        "root": ROOT / "results" / "final" / "stop_template",
        "suffix": "_per_frame.csv",
        "output_col": "detected",
        "correct_col": "correct_detection",
        "iou_col": "iou",
    },

    "YOLO": {
        "root": ROOT / "results" / "final" / "stop_yolo",
        "suffix": "_frames.csv",
        "output_col": "detected_at_operating_threshold",
        "correct_col": "correct_iou_detection",
        "iou_col": "best_accepted_iou",
    },

    "RF-DETR": {
        "root": ROOT / "results" / "final" / "stop_rfdetr",
        "suffix": "_frames.csv",
        "output_col": "detected_at_operating_threshold",
        "correct_col": "correct_iou_detection",
        "iou_col": "best_accepted_iou",
    },
}


def condition_name(severity):
    if severity == 0:
        return "clear"
    return f"fog{severity}"


# ============================================================
# LOAD FROZEN FRAME SELECTION
# ============================================================

selection = pd.read_csv(SELECTION)

required_sel = {
    "trial",
    "sample_id",
    "distance_to_sign_m",
}

missing = required_sel - set(selection.columns)

if missing:
    raise RuntimeError(
        f"Selection file missing columns: {sorted(missing)}"
    )

# Hard guard: 5 frames × 5 configurations.
for trial in TRIALS:

    part = selection[
        selection["trial"] == trial
    ]

    if len(part) != 5:
        raise RuntimeError(
            f"Trial {trial}: expected exactly 5 "
            f"frozen frames, got {len(part)}"
        )


print("=" * 100)
print("POST-HOC MATCHED-DISTANCE STOP ANALYSIS")
print("=" * 100)

print(
    "Frozen selected physical range:",
    f"{selection['distance_to_sign_m'].min():.3f}",
    "to",
    f"{selection['distance_to_sign_m'].max():.3f}",
    "m",
)

print(
    "Frozen selected median:",
    f"{selection['distance_to_sign_m'].median():.3f} m",
)

print(
    "Frames per controlled configuration: 5"
)


# ============================================================
# PROCESS ALL EXISTING DETECTOR OUTPUTS
# ============================================================

per_trial_rows = []
per_frame_rows = []


for detector, spec in DETECTORS.items():

    print()
    print("=" * 100)
    print(detector)
    print("=" * 100)

    for severity in SEVERITIES:

        cond = condition_name(severity)

        for trial in TRIALS:

            path = (
                spec["root"]
                / f"{cond}_trial_{trial}{spec['suffix']}"
            )

            if not path.exists():
                raise RuntimeError(
                    f"Missing detector file: {path}"
                )

            df = pd.read_csv(path)

            required = {
                "sample_id",
                "distance_to_sign_m",
                spec["output_col"],
                spec["correct_col"],
                spec["iou_col"],
            }

            missing = required - set(df.columns)

            if missing:
                raise RuntimeError(
                    f"{path} missing columns: "
                    f"{sorted(missing)}"
                )

            frozen = (
                selection[
                    selection["trial"] == trial
                ][
                    [
                        "sample_id",
                        "distance_to_sign_m",
                    ]
                ]
                .rename(
                    columns={
                        "distance_to_sign_m":
                            "frozen_distance_m"
                    }
                )
            )

            matched = frozen.merge(
                df[
                    [
                        "sample_id",
                        "distance_to_sign_m",
                        spec["output_col"],
                        spec["correct_col"],
                        spec["iou_col"],
                    ]
                ],
                on="sample_id",
                how="left",
                validate="one_to_one",
            )

            if len(matched) != 5:
                raise RuntimeError(
                    f"{detector} severity={severity} "
                    f"trial={trial}: expected 5 matched frames."
                )

            if matched[
                spec["output_col"]
            ].isna().any():
                raise RuntimeError(
                    f"{detector} severity={severity} "
                    f"trial={trial}: missing selected frame."
                )

            # Geometry consistency check.
            max_distance_diff = float(
                np.abs(
                    matched["distance_to_sign_m"]
                    - matched["frozen_distance_m"]
                ).max()
            )

            if max_distance_diff > 0.05:
                raise RuntimeError(
                    f"{detector} severity={severity} "
                    f"trial={trial}: distance mismatch "
                    f"{max_distance_diff:.6f} m"
                )

            output = (
                pd.to_numeric(
                    matched[
                        spec["output_col"]
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

            correct = (
                pd.to_numeric(
                    matched[
                        spec["correct_col"]
                    ],
                    errors="coerce",
                )
                .fillna(0)
                .astype(int)
            )

            # Correct detection cannot exist without an
            # accepted detector output.
            if (correct > output).any():
                raise RuntimeError(
                    f"{detector} severity={severity} "
                    f"trial={trial}: correct > output."
                )

            iou = pd.to_numeric(
                matched[
                    spec["iou_col"]
                ],
                errors="coerce",
            )

            n = len(matched)

            n_output = int(
                output.sum()
            )

            n_correct = int(
                correct.sum()
            )

            output_availability_pct = (
                100.0 * n_output / n
            )

            correct_detection_pct = (
                100.0 * n_correct / n
            )

            # IoU only among frames where the detector
            # actually emitted an accepted output.
            output_iou = iou[
                output == 1
            ].dropna()

            conditional_iou_mean = (
                float(output_iou.mean())
                if len(output_iou)
                else np.nan
            )

            conditional_iou_std = (
                float(output_iou.std(ddof=1))
                if len(output_iou) > 1
                else np.nan
            )

            per_trial_rows.append(
                {
                    "detector":
                        detector,

                    "severity":
                        severity,

                    "trial":
                        trial,

                    "n_selected":
                        n,

                    "n_output":
                        n_output,

                    "output_availability_pct":
                        output_availability_pct,

                    "n_correct":
                        n_correct,

                    "correct_detection_pct":
                        correct_detection_pct,

                    "conditional_iou_mean":
                        conditional_iou_mean,

                    "conditional_iou_std":
                        conditional_iou_std,

                    "median_distance_m":
                        float(
                            matched[
                                "distance_to_sign_m"
                            ].median()
                        ),
                }
            )

            for i in range(n):

                per_frame_rows.append(
                    {
                        "detector":
                            detector,

                        "severity":
                            severity,

                        "trial":
                            trial,

                        "sample_id":
                            int(
                                matched.iloc[i][
                                    "sample_id"
                                ]
                            ),

                        "distance_to_sign_m":
                            float(
                                matched.iloc[i][
                                    "distance_to_sign_m"
                                ]
                            ),

                        "output":
                            int(
                                output.iloc[i]
                            ),

                        "correct_detection":
                            int(
                                correct.iloc[i]
                            ),

                        "iou":
                            (
                                float(iou.iloc[i])
                                if pd.notna(
                                    iou.iloc[i]
                                )
                                else np.nan
                            ),
                    }
                )


per_trial = pd.DataFrame(
    per_trial_rows
)

per_frame = pd.DataFrame(
    per_frame_rows
)


# ============================================================
# SUMMARY ACROSS THE FIVE CONTROLLED CONFIGURATIONS
# ============================================================

summary_rows = []


for (
    detector,
    severity,
), g in per_trial.groupby(
    [
        "detector",
        "severity",
    ]
):

    summary_rows.append(
        {
            "detector":
                detector,

            "severity":
                int(severity),

            "n_configurations":
                len(g),

            "selected_frames_per_configuration":
                int(
                    g["n_selected"].iloc[0]
                ),

            "median_distance_mean_m":
                float(
                    g[
                        "median_distance_m"
                    ].mean()
                ),

            "output_availability_mean_pct":
                float(
                    g[
                        "output_availability_pct"
                    ].mean()
                ),

            "output_availability_std_pct":
                float(
                    g[
                        "output_availability_pct"
                    ].std(ddof=1)
                ),

            "correct_detection_mean_pct":
                float(
                    g[
                        "correct_detection_pct"
                    ].mean()
                ),

            "correct_detection_std_pct":
                float(
                    g[
                        "correct_detection_pct"
                    ].std(ddof=1)
                ),

            "conditional_iou_mean":
                float(
                    g[
                        "conditional_iou_mean"
                    ].mean()
                ),

            "conditional_iou_config_std":
                float(
                    g[
                        "conditional_iou_mean"
                    ].std(ddof=1)
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


detector_order = {
    "HSV": 0,
    "Template": 1,
    "YOLO": 2,
    "RF-DETR": 3,
}

summary["_order"] = (
    summary["detector"]
    .map(detector_order)
)

summary = (
    summary
    .sort_values(
        [
            "_order",
            "severity",
        ]
    )
    .drop(
        columns="_order"
    )
    .reset_index(drop=True)
)


# ============================================================
# JOIN WITH 25 m LANE AVAILABILITY
# ============================================================

if not LANE_SUMMARY.exists():
    raise RuntimeError(
        f"Missing lane summary: {LANE_SUMMARY}"
    )

lane = pd.read_csv(
    LANE_SUMMARY
)

lane25 = lane[
    lane["row_label"]
    == "target_25m"
][
    [
        "severity",
        "availability_mean_pct",
        "availability_std_pct",
        "conditional_mae_m_mean",
        "conditional_mae_m_std",
    ]
].copy()

lane25 = lane25.rename(
    columns={
        "availability_mean_pct":
            "lane_output_availability_mean_pct",

        "availability_std_pct":
            "lane_output_availability_std_pct",

        "conditional_mae_m_mean":
            "lane_conditional_mae_m_mean",

        "conditional_mae_m_std":
            "lane_conditional_mae_m_std",
    }
)


comparison = summary.merge(
    lane25,
    on="severity",
    how="left",
)


# ============================================================
# SAVE
# ============================================================

per_frame_path = (
    OUT
    / "matched_25m_per_frame.csv"
)

per_trial_path = (
    OUT
    / "matched_25m_per_trial.csv"
)

summary_path = (
    OUT
    / "matched_25m_summary.csv"
)

comparison_path = (
    OUT
    / "matched_25m_lane_stop_comparison.csv"
)

per_frame.to_csv(
    per_frame_path,
    index=False,
)

per_trial.to_csv(
    per_trial_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)

comparison.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 120)
print("MATCHED ~25 m STOP RESULTS")
print("=" * 120)

for detector in [
    "HSV",
    "Template",
    "YOLO",
    "RF-DETR",
]:

    g = summary[
        summary["detector"]
        == detector
    ]

    print()
    print(detector)

    print(
        g[
            [
                "severity",
                "output_availability_mean_pct",
                "output_availability_std_pct",
                "correct_detection_mean_pct",
                "correct_detection_std_pct",
                "conditional_iou_mean",
            ]
        ].to_string(
            index=False
        )
    )


print()
print("=" * 120)
print("LANE vs STOP OUTPUT AVAILABILITY AT ~25 m")
print("=" * 120)

wide = comparison.pivot_table(
    index="severity",
    columns="detector",
    values="output_availability_mean_pct",
)

wide.insert(
    0,
    "Lane_UFLD",
    lane25
    .set_index("severity")[
        "lane_output_availability_mean_pct"
    ],
)

print(
    wide.to_string()
)


print()
print("=" * 120)
print("STOP CORRECT-DETECTION RATE AT ~25 m")
print("=" * 120)

correct_wide = summary.pivot(
    index="severity",
    columns="detector",
    values="correct_detection_mean_pct",
)

print(
    correct_wide.to_string()
)


print()
print("=" * 120)
print("SAVED")
print("=" * 120)

print(per_frame_path)
print(per_trial_path)
print(summary_path)
print(comparison_path)

print()
print(
    "Interpretation rule:"
)

print(
    "- output availability = detector emitted an accepted result"
)

print(
    "- correct-detection rate = accepted result also passed IoU >= 0.5"
)

print(
    "- these are post-hoc matched-distance diagnostics"
)
