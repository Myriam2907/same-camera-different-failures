#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

TARGET_DISTANCE_M = 25.053306
K_CLOSEST = 5

TRIALS = [1, 2, 3, 4, 5]

SEVERITIES = [
    0,
    15,
    30,
    45,
    60,
    75,
    90,
]

OUT = (
    ROOT
    / "results"
    / "final"
    / "stop_matched_25m"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DETECTOR FILE LOCATIONS
#
# Used ONLY to verify that selected sample IDs exist.
# No detection/correctness result is read here.
# ============================================================

DETECTORS = {
    "HSV": {
        "root":
            ROOT
            / "results"
            / "final"
            / "stop_hsv",

        "suffix":
            "_per_frame.csv",
    },

    "Template": {
        "root":
            ROOT
            / "results"
            / "final"
            / "stop_template",

        "suffix":
            "_per_frame.csv",
    },

    "YOLO": {
        "root":
            ROOT
            / "results"
            / "final"
            / "stop_yolo",

        "suffix":
            "_frames.csv",
    },

    "RF-DETR": {
        "root":
            ROOT
            / "results"
            / "final"
            / "stop_rfdetr",

        "suffix":
            "_frames.csv",
    },
}


def condition_name(
    severity,
):
    if severity == 0:
        return "clear"

    return f"fog{severity}"


# ============================================================
# SELECT FRAMES USING ONLY ORIGINAL RECORDING GEOMETRY
# ============================================================

selected_rows = []


print("=" * 90)
print("POST-HOC 25 m STOP RANGE-MATCH PREREGISTRATION")
print("=" * 90)

print(
    "Target physical distance:",
    f"{TARGET_DISTANCE_M:.6f} m",
)

print(
    "Selection rule:",
    f"{K_CLOSEST} closest frames per controlled configuration",
)

print(
    "Selection information:",
    "recording geometry ONLY",
)

print()


for trial in TRIALS:

    recording = (
        ROOT
        / "data"
        / "recordings"
        / (
            f"stop3_clear_0_"
            f"drift_0p75_trial_{trial}"
        )
        / "frames.csv"
    )

    if not recording.exists():
        raise RuntimeError(
            f"Missing recording: {recording}"
        )

    df = pd.read_csv(
        recording
    )

    required = {
        "sample_id",
        "route_s_m",
        "distance_to_sign_m",
        "sign_in_view",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{recording} missing "
            f"{sorted(missing)}"
        )

    eligible = df[
        (
            df[
                "sign_in_view"
            ]
            == 1
        )
        &
        (
            df[
                "distance_to_sign_m"
            ].notna()
        )
    ].copy()

    eligible[
        "distance_error_m"
    ] = np.abs(
        eligible[
            "distance_to_sign_m"
        ]
        - TARGET_DISTANCE_M
    )

    # Geometry-only ordering.
    # sample_id is deterministic tie-break.
    selected = (
        eligible
        .sort_values(
            [
                "distance_error_m",
                "sample_id",
            ]
        )
        .head(
            K_CLOSEST
        )
        .sort_values(
            "distance_to_sign_m"
        )
        .copy()
    )

    if len(
        selected
    ) != K_CLOSEST:
        raise RuntimeError(
            f"Trial {trial}: "
            f"expected {K_CLOSEST} frames, "
            f"found {len(selected)}."
        )

    print(
        f"TRIAL {trial}"
    )

    print(
        selected[
            [
                "sample_id",
                "route_s_m",
                "distance_to_sign_m",
                "distance_error_m",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "  range:",
        f"{selected['distance_to_sign_m'].min():.3f}",
        "to",
        f"{selected['distance_to_sign_m'].max():.3f}",
        "m",
    )

    print(
        "  median:",
        f"{selected['distance_to_sign_m'].median():.3f} m",
    )

    print()

    for _, row in (
        selected.iterrows()
    ):

        selected_rows.append(
            {
                "trial":
                    trial,

                "sample_id":
                    int(
                        row[
                            "sample_id"
                        ]
                    ),

                "route_s_m":
                    float(
                        row[
                            "route_s_m"
                        ]
                    ),

                "distance_to_sign_m":
                    float(
                        row[
                            "distance_to_sign_m"
                        ]
                    ),

                "abs_distance_error_m":
                    float(
                        row[
                            "distance_error_m"
                        ]
                    ),
            }
        )


selection = pd.DataFrame(
    selected_rows
)


# ============================================================
# GLOBAL RANGE CHECK
# ============================================================

print("=" * 90)
print("GLOBAL SELECTED RANGE")
print("=" * 90)

print(
    "N selected:",
    len(selection),
)

print(
    "Minimum distance:",
    f"{selection['distance_to_sign_m'].min():.6f} m",
)

print(
    "Maximum distance:",
    f"{selection['distance_to_sign_m'].max():.6f} m",
)

print(
    "Median distance:",
    f"{selection['distance_to_sign_m'].median():.6f} m",
)

print(
    "Mean absolute target error:",
    f"{selection['abs_distance_error_m'].mean():.6f} m",
)


# ============================================================
# VERIFY SAMPLE IDS EXIST IN ALL EXISTING OUTPUTS
#
# Read sample_id ONLY.
# No detector outcome is inspected.
# ============================================================

print()
print("=" * 90)
print("OUTCOME-BLIND FILE COVERAGE CHECK")
print("=" * 90)


for detector, info in (
    DETECTORS.items()
):

    checked = 0

    for severity in SEVERITIES:

        condition = (
            condition_name(
                severity
            )
        )

        for trial in TRIALS:

            filename = (
                f"{condition}_"
                f"trial_{trial}"
                f"{info['suffix']}"
            )

            path = (
                info["root"]
                / filename
            )

            if not path.exists():
                raise RuntimeError(
                    f"Missing {detector} file: "
                    f"{path}"
                )

            # Deliberately read ONLY sample_id.
            ids = pd.read_csv(
                path,
                usecols=[
                    "sample_id"
                ],
            )

            ids = set(
                ids[
                    "sample_id"
                ].astype(int)
            )

            wanted = set(
                selection[
                    selection[
                        "trial"
                    ]
                    == trial
                ][
                    "sample_id"
                ].astype(int)
            )

            missing_ids = (
                wanted - ids
            )

            if missing_ids:
                raise RuntimeError(
                    f"{detector} "
                    f"{condition} "
                    f"trial {trial}: "
                    f"missing sample IDs "
                    f"{sorted(missing_ids)}"
                )

            checked += 1

    print(
        f"{detector:10s}: "
        f"PASS ({checked} files)"
    )


# ============================================================
# SAVE FROZEN SELECTION
# ============================================================

csv_path = (
    OUT
    / "preregistered_25m_frames.csv"
)

selection.to_csv(
    csv_path,
    index=False,
)


txt_path = (
    OUT
    / "preregistered_25m_frames.txt"
)

with txt_path.open(
    "w"
) as f:

    f.write(
        "POST-HOC STOP-SIGN 25 m "
        "RANGE-CONTROL PREREGISTRATION\n\n"
    )

    f.write(
        "This is a post-hoc range-control analysis, "
        "not an original preregistered metric.\n"
    )

    f.write(
        "Frame selection was frozen using physical "
        "distance only, before matched-distance "
        "detector outcomes were inspected.\n\n"
    )

    f.write(
        f"Target distance: "
        f"{TARGET_DISTANCE_M:.6f} m\n"
    )

    f.write(
        f"Frames per controlled configuration: "
        f"{K_CLOSEST}\n"
    )

    f.write(
        "Rule: choose the five sign-in-view frames "
        "with smallest absolute distance from the "
        "target, using Clear recording geometry. "
        "Reuse the same sample IDs for the paired "
        "weather conditions.\n\n"
    )

    f.write(
        selection.to_string(
            index=False
        )
    )

    f.write("\n")


print()
print("=" * 90)
print("FROZEN SELECTION SAVED")
print("=" * 90)

print(csv_path)
print(txt_path)

print()
print(
    "Do not change the target, K, or selected "
    "sample IDs after inspecting detector outcomes."
)
