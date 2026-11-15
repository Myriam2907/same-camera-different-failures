#!/usr/bin/env python3

from pathlib import Path
import itertools

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

SEVERITIES = [
    0,
    15,
    30,
    45,
    60,
    75,
    90,
]

TRIALS = [
    1,
    2,
    3,
    4,
    5,
]


# ============================================================
# LOAD
# ============================================================

selection = pd.read_csv(
    SELECTION
)

lane = pd.read_csv(
    LANE_FRAME
)

stop = pd.read_csv(
    STOP_FRAME
)


# ============================================================
# HARD CHECK THE FROZEN STOP SELECTION
# ============================================================

for trial in TRIALS:

    s = selection[
        selection["trial"]
        == trial
    ]

    if len(s) != 5:
        raise RuntimeError(
            f"Trial {trial}: expected "
            f"5 frozen frames, got {len(s)}"
        )


print("=" * 110)
print(
    "EXACT SAME-FRAME ~25 m "
    "LANE-vs-STOP CHECK"
)
print("=" * 110)

print(
    "STOP physical range:",
    f"{selection['distance_to_sign_m'].min():.3f}",
    "to",
    f"{selection['distance_to_sign_m'].max():.3f}",
    "m",
)

print(
    "STOP median physical distance:",
    f"{selection['distance_to_sign_m'].median():.3f} m",
)

print(
    "Lane target range:",
    "~25.053 m",
)

print(
    "Frozen sample IDs by trial:"
)

for trial in TRIALS:

    ids = (
        selection[
            selection["trial"]
            == trial
        ]["sample_id"]
        .astype(int)
        .tolist()
    )

    print(
        f"  trial {trial}: {ids}"
    )


# ============================================================
# CREATE EXPECTED SAME-FRAME KEYS
# ============================================================

expected_rows = []

for severity, trial in itertools.product(
    SEVERITIES,
    TRIALS,
):

    selected_trial = selection[
        selection["trial"]
        == trial
    ]

    for _, row in selected_trial.iterrows():

        expected_rows.append(
            {
                "severity":
                    severity,

                "trial":
                    trial,

                "sample_id":
                    int(
                        row["sample_id"]
                    ),

                "stop_distance_m":
                    float(
                        row[
                            "distance_to_sign_m"
                        ]
                    ),
            }
        )

expected = pd.DataFrame(
    expected_rows
)


# ============================================================
# FILTER LANE TO FROZEN 25 m ROW
# ============================================================

lane25 = lane[
    lane["row_label"]
    == "target_25m"
].copy()

required_lane = {
    "severity",
    "trial",
    "sample_id",
    "camera_range_m",
    "success",
    "conditional_error_m",
}

missing = (
    required_lane
    - set(lane25.columns)
)

if missing:
    raise RuntimeError(
        f"Lane file missing columns: "
        f"{sorted(missing)}"
    )


# ============================================================
# EXACT FRAME MATCH
# ============================================================

same = expected.merge(
    lane25[
        [
            "severity",
            "trial",
            "sample_id",
            "camera_range_m",
            "success",
            "conditional_error_m",
        ]
    ],
    on=[
        "severity",
        "trial",
        "sample_id",
    ],
    how="left",
    indicator=True,
    validate="one_to_one",
)


# ============================================================
# GEOMETRY COVERAGE CHECK
# ============================================================

missing_lane = same[
    same["_merge"]
    != "both"
]

print()
print("=" * 110)
print(
    "LANE GEOMETRY COVERAGE "
    "ON FROZEN STOP FRAMES"
)
print("=" * 110)

if len(missing_lane):

    print(
        "WARNING: some frozen STOP frames "
        "are not geometrically eligible "
        "for the lane 25 m row."
    )

    print(
        missing_lane[
            [
                "severity",
                "trial",
                "sample_id",
            ]
        ].to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Do NOT reselect frames. "
        "Inspect missing geometry first."
    )

else:

    print(
        "PASS: every frozen STOP frame "
        "is also eligible for UFLD at 25 m."
    )

    print(
        "Total matched lane rows:",
        len(same),
    )


same = same.drop(
    columns="_merge"
)


# ============================================================
# PER-CONFIGURATION LANE RESULTS
# ============================================================

lane_trial_rows = []

for (
    severity,
    trial,
), g in same.groupby(
    [
        "severity",
        "trial",
    ]
):

    n = len(g)

    if n != 5:
        raise RuntimeError(
            f"Severity {severity}, "
            f"trial {trial}: "
            f"expected 5 frames, got {n}"
        )

    success = pd.to_numeric(
        g["success"],
        errors="coerce",
    ).fillna(0).astype(int)

    n_success = int(
        success.sum()
    )

    availability = (
        100.0
        * n_success
        / n
    )

    cond = pd.to_numeric(
        g["conditional_error_m"],
        errors="coerce",
    )

    cond_success = cond[
        success == 1
    ].dropna()

    lane_trial_rows.append(
        {
            "severity":
                int(severity),

            "trial":
                int(trial),

            "n_frames":
                n,

            "n_success":
                n_success,

            "availability_pct":
                availability,

            "conditional_mae_m":
                (
                    float(
                        cond_success.mean()
                    )
                    if len(cond_success)
                    else np.nan
                ),

            "stop_distance_median":
                float(
                    g[
                        "stop_distance_m"
                    ].median()
                ),

            "lane_range_median":
                float(
                    g[
                        "camera_range_m"
                    ].median()
                ),
        }
    )


lane_trial = pd.DataFrame(
    lane_trial_rows
)


# ============================================================
# LANE SAME-FRAME SUMMARY
# ============================================================

lane_summary_rows = []

for severity, g in lane_trial.groupby(
    "severity"
):

    lane_summary_rows.append(
        {
            "severity":
                int(severity),

            "lane_same_frame_availability_mean_pct":
                float(
                    g[
                        "availability_pct"
                    ].mean()
                ),

            "lane_same_frame_availability_std_pct":
                float(
                    g[
                        "availability_pct"
                    ].std(ddof=1)
                ),

            "lane_same_frame_conditional_mae_m_mean":
                float(
                    g[
                        "conditional_mae_m"
                    ].mean()
                ),

            "lane_same_frame_conditional_mae_m_std":
                float(
                    g[
                        "conditional_mae_m"
                    ].std(ddof=1)
                ),

            "lane_conditional_configs_n":
                int(
                    g[
                        "conditional_mae_m"
                    ].notna().sum()
                ),
        }
    )


lane_summary = pd.DataFrame(
    lane_summary_rows
)


# ============================================================
# STOP SUMMARY ON THE EXACT SAME FRAMES
# ============================================================

stop_required = {
    "detector",
    "severity",
    "trial",
    "sample_id",
    "output",
    "correct_detection",
}

missing = (
    stop_required
    - set(stop.columns)
)

if missing:
    raise RuntimeError(
        f"STOP file missing columns: "
        f"{sorted(missing)}"
    )


# Verify stop file is exactly the frozen frame set.
for detector in stop[
    "detector"
].unique():

    part = stop[
        stop["detector"]
        == detector
    ]

    check = expected.merge(
        part[
            [
                "severity",
                "trial",
                "sample_id",
            ]
        ],
        on=[
            "severity",
            "trial",
            "sample_id",
        ],
        how="left",
        indicator=True,
    )

    if (
        check["_merge"]
        != "both"
    ).any():

        raise RuntimeError(
            f"{detector}: STOP matched file "
            "does not cover frozen frame set."
        )


stop_trial = (
    stop
    .groupby(
        [
            "detector",
            "severity",
            "trial",
        ],
        as_index=False,
    )
    .agg(
        n_frames=(
            "sample_id",
            "count",
        ),

        n_output=(
            "output",
            "sum",
        ),

        n_correct=(
            "correct_detection",
            "sum",
        ),
    )
)

stop_trial[
    "output_availability_pct"
] = (
    100.0
    * stop_trial[
        "n_output"
    ]
    / stop_trial[
        "n_frames"
    ]
)

stop_trial[
    "correct_detection_pct"
] = (
    100.0
    * stop_trial[
        "n_correct"
    ]
    / stop_trial[
        "n_frames"
    ]
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
# BUILD SAME-FRAME COMPARISON TABLE
# ============================================================

wide_output = stop_summary.pivot(
    index="severity",
    columns="detector",
    values="output_availability_mean_pct",
)

wide_correct = stop_summary.pivot(
    index="severity",
    columns="detector",
    values="correct_detection_mean_pct",
)

final = (
    lane_summary
    .set_index(
        "severity"
    )[
        [
            "lane_same_frame_availability_mean_pct",
            "lane_same_frame_availability_std_pct",
            "lane_same_frame_conditional_mae_m_mean",
            "lane_same_frame_conditional_mae_m_std",
        ]
    ]
    .copy()
)

for detector in [
    "HSV",
    "Template",
    "YOLO",
    "RF-DETR",
]:

    final[
        f"{detector}_output_pct"
    ] = wide_output[
        detector
    ]

    final[
        f"{detector}_correct_pct"
    ] = wide_correct[
        detector
    ]


final = final.reset_index()


# ============================================================
# SAVE
# ============================================================

same_path = (
    OUT
    / "same_frame_lane_per_frame.csv"
)

lane_trial_path = (
    OUT
    / "same_frame_lane_per_trial.csv"
)

lane_summary_path = (
    OUT
    / "same_frame_lane_summary.csv"
)

comparison_path = (
    OUT
    / "same_frame_lane_stop_comparison.csv"
)

same.to_csv(
    same_path,
    index=False,
)

lane_trial.to_csv(
    lane_trial_path,
    index=False,
)

lane_summary.to_csv(
    lane_summary_path,
    index=False,
)

final.to_csv(
    comparison_path,
    index=False,
)


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 120)
print(
    "UFLD LANE ON THE EXACT SAME "
    "25 STOP FRAMES"
)
print("=" * 120)

print(
    lane_summary.to_string(
        index=False
    )
)


print()
print("=" * 120)
print(
    "EXACT SAME-FRAME, "
    "SAME-RANGE COMPARISON"
)
print("=" * 120)

display_cols = [
    "severity",
    "lane_same_frame_availability_mean_pct",
    "HSV_output_pct",
    "HSV_correct_pct",
    "Template_output_pct",
    "Template_correct_pct",
    "YOLO_output_pct",
    "YOLO_correct_pct",
    "RF-DETR_output_pct",
    "RF-DETR_correct_pct",
]

print(
    final[
        display_cols
    ].to_string(
        index=False
    )
)


print()
print("=" * 120)
print("SAVED")
print("=" * 120)

print(same_path)
print(lane_trial_path)
print(lane_summary_path)
print(comparison_path)

print()
print(
    "This is a post-hoc exact-frame "
    "range-control analysis."
)

print(
    "No detector thresholds, frozen rows, "
    "or selected frame IDs were changed."
)
