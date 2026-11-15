#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent

OUT = (
    ROOT
    / "results"
    / "final"
    / "stop_rfdetr"
)

SEVERITIES = [0, 15, 30, 45, 60, 75, 90]
TRIALS = [1, 2, 3, 4, 5]


def dataset_name(severity, trial):

    if severity == 0:
        return f"clear_trial_{trial}"

    return f"fog{severity}_trial_{trial}"


# ============================================================
# CHECK ALL 35 SAVED RF-DETR FRAME FILES
# ============================================================

expected = []

for severity in SEVERITIES:

    for trial in TRIALS:

        name = dataset_name(
            severity,
            trial,
        )

        expected.append(
            OUT
            / f"{name}_frames.csv"
        )


missing = [
    path
    for path in expected
    if not path.exists()
]


print("=" * 95)
print("FINAL RF-DETR-L RECOVERY")
print("=" * 95)

print(
    "Expected frame CSVs:",
    len(expected),
)

print(
    "Found:",
    len(expected) - len(missing),
)


if missing:

    print()

    for path in missing:
        print("MISSING:", path)

    raise RuntimeError(
        "RF-DETR output is incomplete."
    )


print(
    "PASS: all 35 RF-DETR frame CSVs exist."
)


# ============================================================
# REQUIRED FRAME-LEVEL COLUMNS
# ============================================================

required_columns = {

    "dataset",
    "severity",
    "sample_id",
    "route_s_m",
    "distance_to_sign_m",

    "accepted_detection_count",
    "detected_at_operating_threshold",
    "correct_iou_detection",

    "false_positive_frame",
    "false_positive_boxes",

    "best_accepted_iou",
}


# ============================================================
# RECONSTRUCT PER-TRIAL RESULTS
# ============================================================

rows = []


for severity in SEVERITIES:

    for trial in TRIALS:

        name = dataset_name(
            severity,
            trial,
        )

        path = (
            OUT
            / f"{name}_frames.csv"
        )


        df = pd.read_csv(
            path
        )


        missing_columns = (
            required_columns
            - set(df.columns)
        )


        if missing_columns:

            raise RuntimeError(
                f"{path} missing columns: "
                f"{sorted(missing_columns)}"
            )


        # STOP-sign encounter = approach phase.
        approach = df[
            df[
                "route_s_m"
            ] <= 0.0
        ].copy()


        approach = (
            approach
            .sort_values(
                "sample_id"
            )
            .reset_index(
                drop=True
            )
        )


        if len(approach) == 0:

            raise RuntimeError(
                f"No evaluable approach frames "
                f"for {name}"
            )


        n_frames = len(
            approach
        )


        # ----------------------------------------------------
        # Accepted detections
        # ----------------------------------------------------

        detection_count = int(
            approach[
                "detected_at_operating_threshold"
            ]
            .astype(int)
            .sum()
        )


        # ----------------------------------------------------
        # Correct STOP detections: IoU >= 0.50
        # ----------------------------------------------------

        correct_count = int(
            approach[
                "correct_iou_detection"
            ]
            .astype(int)
            .sum()
        )


        # ----------------------------------------------------
        # False positives
        # ----------------------------------------------------

        false_positive_frames = int(
            approach[
                "false_positive_frame"
            ]
            .astype(int)
            .sum()
        )


        false_positive_boxes = int(
            approach[
                "false_positive_boxes"
            ]
            .astype(int)
            .sum()
        )


        detection_rate = (
            detection_count
            / n_frames
        )


        correct_detection_rate = (
            correct_count
            / n_frames
        )


        # ----------------------------------------------------
        # First CORRECT detection
        # ----------------------------------------------------

        correct_rows = approach[
            approach[
                "correct_iou_detection"
            ].astype(int) == 1
        ]


        if len(correct_rows) > 0:

            first = (
                correct_rows
                .sort_values(
                    "sample_id"
                )
                .iloc[0]
            )


            first_sample = int(
                first[
                    "sample_id"
                ]
            )


            first_route = float(
                first[
                    "route_s_m"
                ]
            )


            first_distance = float(
                first[
                    "distance_to_sign_m"
                ]
            )

        else:

            first_sample = np.nan
            first_route = np.nan

            # Frozen project rule:
            # never correctly detected -> D(s) = 0.
            first_distance = 0.0


        # ----------------------------------------------------
        # Accepted detection IoU
        # ----------------------------------------------------

        accepted = approach[
            approach[
                "detected_at_operating_threshold"
            ].astype(int) == 1
        ]


        accepted_iou = (
            accepted[
                "best_accepted_iou"
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .dropna()
        )


        if len(accepted_iou) > 0:

            mean_accepted_iou = float(
                accepted_iou.mean()
            )

        else:

            mean_accepted_iou = np.nan


        rows.append({

            "dataset":
                name,

            "severity":
                severity,

            "trial":
                trial,

            "evaluable_approach_frames":
                n_frames,

            "detection_count":
                detection_count,

            "detection_rate":
                detection_rate,

            "correct_detection_count":
                correct_count,

            "correct_detection_rate":
                correct_detection_rate,

            "false_positive_frames":
                false_positive_frames,

            "false_positive_boxes":
                false_positive_boxes,

            "mean_best_iou_when_accepted":
                mean_accepted_iou,

            "first_correct_detection_sample":
                first_sample,

            "first_correct_detection_route_s_m":
                first_route,

            "first_correct_detection_distance_m":
                first_distance,
        })


per_trial = pd.DataFrame(
    rows
)


if len(per_trial) != 35:

    raise RuntimeError(
        f"Expected 35 trial rows, "
        f"got {len(per_trial)}"
    )


# ============================================================
# PAIRED R_sign
#
# Fog trial j is normalized against Clear trial j.
#
# R_sign_j(s) = min(1, D_j(s) / D_j(0))
# ============================================================

per_trial[
    "R_sign"
] = np.nan


for trial in TRIALS:

    clear = per_trial[
        (
            per_trial[
                "severity"
            ] == 0
        )
        &
        (
            per_trial[
                "trial"
            ] == trial
        )
    ]


    if len(clear) != 1:

        raise RuntimeError(
            f"Trial {trial}: "
            "Clear baseline is not unique."
        )


    d0 = float(
        clear.iloc[0][
            "first_correct_detection_distance_m"
        ]
    )


    if not np.isfinite(d0) or d0 <= 0.0:

        raise RuntimeError(
            f"Trial {trial}: "
            f"invalid Clear D(0)={d0}"
        )


    trial_rows = per_trial[
        per_trial[
            "trial"
        ] == trial
    ].index


    for idx in trial_rows:

        d = float(
            per_trial.loc[
                idx,
                "first_correct_detection_distance_m",
            ]
        )


        if not np.isfinite(d):
            d = 0.0


        per_trial.loc[
            idx,
            "R_sign",
        ] = min(
            1.0,
            max(
                0.0,
                d,
            )
            / d0,
        )


# ============================================================
# SAVE PER-TRIAL RESULTS
# ============================================================

per_trial_path = (
    OUT
    / "final_per_trial.csv"
)


per_trial.to_csv(
    per_trial_path,
    index=False,
)


# ============================================================
# FIVE-TRIAL SUMMARY
# ============================================================

summary_rows = []


for severity in SEVERITIES:

    block = per_trial[
        per_trial[
            "severity"
        ] == severity
    ].copy()


    if len(block) != 5:

        raise RuntimeError(
            f"Fog {severity}: "
            f"expected 5 trials, "
            f"got {len(block)}"
        )


    summary_rows.append({

        "severity":
            severity,

        "n_trials":
            len(block),

        "correct_detection_rate_mean_pct":
            float(
                block[
                    "correct_detection_rate"
                ].mean()
                * 100.0
            ),

        "correct_detection_rate_std_pct":
            float(
                block[
                    "correct_detection_rate"
                ].std(
                    ddof=1
                )
                * 100.0
            ),

        "first_correct_distance_mean_m":
            float(
                block[
                    "first_correct_detection_distance_m"
                ].mean()
            ),

        "first_correct_distance_std_m":
            float(
                block[
                    "first_correct_detection_distance_m"
                ].std(
                    ddof=1
                )
            ),

        "false_positive_frames_mean":
            float(
                block[
                    "false_positive_frames"
                ].mean()
            ),

        "false_positive_boxes_mean":
            float(
                block[
                    "false_positive_boxes"
                ].mean()
            ),

        "R_sign_mean":
            float(
                block[
                    "R_sign"
                ].mean()
            ),

        "R_sign_std":
            float(
                block[
                    "R_sign"
                ].std(
                    ddof=1
                )
            ),
    })


summary = pd.DataFrame(
    summary_rows
)


summary_path = (
    OUT
    / "final_summary.csv"
)


summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

print()
print("=" * 125)
print("FINAL RF-DETR-L RESULTS")
print("=" * 125)

print(
    summary
    .round(4)
    .to_string(
        index=False
    )
)


print()
print(
    "PAIRED R_sign VALUES:"
)


print(
    per_trial[
        [
            "severity",
            "trial",
            "first_correct_detection_distance_m",
            "R_sign",
        ]
    ]
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("Reliability thresholds:")


for threshold in [
    0.40,
    0.50,
    0.60,
]:

    failed = summary[
        summary[
            "R_sign_mean"
        ] < threshold
    ]


    if len(failed) == 0:

        print(
            f"R_sign < {threshold:.2f}: "
            "not reached"
        )

    else:

        first_severity = int(
            failed.iloc[0][
                "severity"
            ]
        )

        print(
            f"R_sign < {threshold:.2f}: "
            f"Fog {first_severity}"
        )


print()
print("Saved:")
print(per_trial_path)
print(summary_path)

print()
print(
    "PASS: final RF-DETR-L "
    "STOP-sign analysis recovered successfully."
)
