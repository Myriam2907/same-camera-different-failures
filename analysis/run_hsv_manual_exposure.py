#!/usr/bin/env python3

from pathlib import Path
import shutil
import numpy as np
import pandas as pd

from sign import hsv_stop as hsv


ROOT = Path(__file__).resolve().parents[1]

RECORDINGS = ROOT / "data" / "recordings_manualexp"

OUT = (
    ROOT
    / "results"
    / "final"
    / "stop_hsv_manualexp"
)

# Remove only OLD manual-exposure HSV results.
if OUT.exists():
    shutil.rmtree(OUT)

OUT.mkdir(parents=True, exist_ok=True)

# Make HSV debug/results go into the manual-control folder.
hsv.OUTPUT_ROOT = str(OUT)
hsv.DEBUG_ROOT = str(OUT / "debug")

SEVERITIES = [0, 30, 45]
TRIALS = [1, 2, 3, 4, 5]


def recording_name(severity, trial):
    if severity == 0:
        return f"stop3_clear_0_drift_0p75_trial_{trial}"

    return f"stop3_fog_{severity}_drift_0p75_trial_{trial}"


rows = []

print("=" * 90)
print("HSV MANUAL-EXPOSURE CONTROL")
print("=" * 90)

for severity in SEVERITIES:

    for trial in TRIALS:

        name = recording_name(
            severity,
            trial,
        )

        root = RECORDINGS / name

        if not root.exists():
            raise RuntimeError(
                f"Missing recording: {root}"
            )

        dataset_name = (
            f"manual_fog{severity}_trial_{trial}"
            if severity != 0
            else f"manual_clear_trial_{trial}"
        )

        dataset_info = {
            "root": str(root),
            "severity": severity,
        }

        print()
        print("=" * 70)
        print(
            f"Severity {severity} | Trial {trial}"
        )
        print("=" * 70)

        summary = hsv.process_dataset(
            dataset_name,
            dataset_info,
        )

        row = {
            "severity": severity,
            "trial": trial,
            **summary,
        }

        rows.append(row)

        print(
            "Correct detection rate:",
            f"{100.0 * summary['correct_detection_rate']:.2f}%"
        )

        print(
            "False positives:",
            summary["false_positive_count"],
        )

        print(
            "First correct distance:",
            summary[
                "first_correct_detection_distance_m"
            ],
        )


per_trial = pd.DataFrame(rows)

# ============================================================
# R_sign relative to MANUAL clear of the SAME trial
# ============================================================

clear_dist = (
    per_trial[
        per_trial["severity"] == 0
    ][
        [
            "trial",
            "first_correct_detection_distance_m",
        ]
    ]
    .rename(
        columns={
            "first_correct_detection_distance_m":
                "clear_first_distance_m"
        }
    )
)

per_trial = per_trial.merge(
    clear_dist,
    on="trial",
    validate="many_to_one",
)

per_trial["R_sign"] = np.minimum(
    1.0,
    per_trial[
        "first_correct_detection_distance_m"
    ]
    / per_trial[
        "clear_first_distance_m"
    ],
)

per_trial[
    "correct_detection_rate_pct"
] = (
    100.0
    * per_trial["correct_detection_rate"]
)

per_trial.to_csv(
    OUT / "manual_per_trial.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_df = (
    per_trial
    .groupby(
        "severity",
        as_index=False,
    )
    .agg(
        n_trials=(
            "trial",
            "count",
        ),
        correct_detection_rate_mean_pct=(
            "correct_detection_rate_pct",
            "mean",
        ),
        correct_detection_rate_std_pct=(
            "correct_detection_rate_pct",
            lambda x: np.std(x, ddof=1),
        ),
        first_correct_distance_mean_m=(
            "first_correct_detection_distance_m",
            "mean",
        ),
        first_correct_distance_std_m=(
            "first_correct_detection_distance_m",
            lambda x: np.std(x, ddof=1),
        ),
        false_positive_mean=(
            "false_positive_count",
            "mean",
        ),
        R_sign_mean=(
            "R_sign",
            "mean",
        ),
        R_sign_std=(
            "R_sign",
            lambda x: np.std(x, ddof=1),
        ),
    )
)

summary_df.to_csv(
    OUT / "manual_summary.csv",
    index=False,
)

print()
print("=" * 100)
print("HSV MANUAL-EXPOSURE SUMMARY")
print("=" * 100)

print(
    summary_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print()
print("Saved:")
print(OUT / "manual_per_trial.csv")
print(OUT / "manual_summary.csv")
