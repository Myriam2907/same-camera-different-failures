#!/usr/bin/env python3

from pathlib import Path
import argparse
import importlib
import sys
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent

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


# ============================================================
# SETTINGS
# ============================================================

SETTINGS = {

    "yolo": {
        "module":
            "sign.yolo_stop",

        "threshold":
            0.0410,

        "output":
            ROOT
            / "results"
            / "final"
            / "stop_yolo",

        "label":
            "YOLO26s",
    },

    "rfdetr": {
        "module":
            "sign.rfdetr_stop",

        "threshold":
            0.0430,

        "output":
            ROOT
            / "results"
            / "final"
            / "stop_rfdetr",

        "label":
            "RF-DETR-L",
    },
}


# ============================================================
# DATASET HELPERS
# ============================================================

def condition_name(
    severity,
):

    if severity == 0:
        return "clear"

    return f"fog{severity}"


def recording_path(
    severity,
    trial,
):

    if severity == 0:

        name = (
            f"stop3_clear_0_"
            f"drift_0p75_trial_{trial}"
        )

    else:

        name = (
            f"stop3_fog_{severity}_"
            f"drift_0p75_trial_{trial}"
        )

    return (
        ROOT
        / "data"
        / "recordings"
        / name
    )


def build_final_datasets():

    datasets = {}

    order = []


    for severity in SEVERITIES:

        for trial in TRIALS:

            condition = (
                condition_name(
                    severity
                )
            )

            dataset_name = (
                f"{condition}_trial_{trial}"
            )

            root = recording_path(
                severity,
                trial,
            )


            csv_path = (
                root
                / "frames.csv"
            )


            if not csv_path.exists():

                raise RuntimeError(
                    f"Missing final dataset:\n"
                    f"{csv_path}"
                )


            datasets[
                dataset_name
            ] = {

                "root":
                    str(root),

                "severity":
                    severity,

                "trial":
                    trial,
            }


            order.append({

                "dataset_name":
                    dataset_name,

                "severity":
                    severity,

                "trial":
                    trial,

                "condition":
                    condition,
            })


    if len(datasets) != 35:

        raise RuntimeError(
            f"Expected 35 datasets, "
            f"got {len(datasets)}"
        )


    return datasets, order


# ============================================================
# REDIRECT OLD OUTPUT PATHS
# ============================================================

def redirect_output_paths(
    module,
    new_output,
):

    new_output.mkdir(
        parents=True,
        exist_ok=True,
    )


    if not hasattr(
        module,
        "OUTPUT_ROOT",
    ):

        raise RuntimeError(
            "Detector module does not "
            "have OUTPUT_ROOT."
        )


    old_output = Path(
        module.OUTPUT_ROOT
    )


    # Change paths derived from the old output
    # directory as well, such as:
    #
    # DEBUG_ROOT
    # CALIBRATION_PATH
    # SUMMARY_PATH
    #
    # if they exist.

    items = list(
        vars(module).items()
    )


    for name, value in items:

        if isinstance(
            value,
            str,
        ):

            path = Path(
                value
            )

            try:

                relative = (
                    path.relative_to(
                        old_output
                    )
                )

            except ValueError:

                continue


            setattr(
                module,
                name,
                str(
                    new_output
                    / relative
                ),
            )


        elif isinstance(
            value,
            Path,
        ):

            try:

                relative = (
                    value.relative_to(
                        old_output
                    )
                )

            except ValueError:

                continue


            setattr(
                module,
                name,
                new_output
                / relative,
            )


    module.OUTPUT_ROOT = new_output


# ============================================================
# FREEZE CALIBRATION
# ============================================================

def freeze_threshold(
    module,
    threshold,
):

    """
    Do NOT recalibrate using final data.

    Any calibration function in the existing
    detector is replaced by a function that
    returns the already-frozen pilot threshold.
    """

    patched = []


    def frozen_calibration(
        *args,
        **kwargs,
    ):

        print()

        print(
            "Using FROZEN pilot threshold:",
            f"{threshold:.4f}",
        )

        print(
            "Final Clear/Fog trials "
            "do NOT affect calibration."
        )

        return threshold


    for name, value in list(
        vars(module).items()
    ):

        lower = name.lower()


        if (
            callable(value)
            and
            "calibrat" in lower
        ):

            setattr(
                module,
                name,
                frozen_calibration,
            )

            patched.append(
                name
            )


    print(
        "Calibration functions replaced:",
        patched,
    )


    # Also update likely confidence-threshold
    # globals, while preserving IoU=0.50.

    for name, value in list(
        vars(module).items()
    ):

        lower = name.lower()


        if not isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        ):

            continue


        if (
            "iou" in lower
        ):

            # Keep IoU correctness rule unchanged.
            continue


        if (
            (
                "conf" in lower
                and
                "threshold" in lower
            )
            or
            (
                "score" in lower
                and
                "threshold" in lower
            )
            or
            lower
            in {
                "frozen_threshold",
                "detection_threshold",
            }
        ):

            print(
                f"Setting {name} "
                f"from {value} "
                f"to {threshold}"
            )

            setattr(
                module,
                name,
                threshold,
            )


# ============================================================
# FIND SUMMARY CREATED BY EXISTING SCRIPT
# ============================================================

def find_summary(
    output_dir,
):

    candidates = [

        output_dir
        / "summary.csv",

        output_dir
        / "results.csv",

    ]


    for path in candidates:

        if path.exists():
            return path


    found = list(
        output_dir.glob(
            "*summary*.csv"
        )
    )


    if len(found) == 1:

        return found[0]


    raise RuntimeError(
        "Could not uniquely find detector "
        f"summary under:\n{output_dir}"
    )


# ============================================================
# BUILD FINAL FIVE-TRIAL SUMMARY
# ============================================================

def make_final_summary(
    raw_summary_path,
    order,
    output_dir,
):

    df = pd.read_csv(
        raw_summary_path
    )


    print()

    print(
        "Rows in detector summary:",
        len(df),
    )


    if len(df) != 35:

        raise RuntimeError(

            "Expected exactly 35 detector "
            f"summary rows, got {len(df)}.\n"

            "Stopping rather than silently "
            "pairing the wrong trials."
        )


    required = {

        "correct_detection_rate",

        "false_positive_count",

        "first_correct_detection_distance_m",
    }


    missing = (
        required
        - set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(

            "Detector summary missing columns: "
            f"{sorted(missing)}"
        )


    # DATASETS is an ordered dictionary,
    # so detector main() processes rows in
    # exactly this order.

    order_df = pd.DataFrame(
        order
    )


    df[
        "trial"
    ] = order_df[
        "trial"
    ].values


    df[
        "severity"
    ] = order_df[
        "severity"
    ].values


    df[
        "condition"
    ] = order_df[
        "condition"
    ].values


    df[
        "final_dataset_name"
    ] = order_df[
        "dataset_name"
    ].values


    # ========================================================
    # PAIRED R_sign
    # ========================================================

    df[
        "R_sign"
    ] = 0.0


    for trial in TRIALS:

        clear = df[
            (
                df["trial"]
                == trial
            )
            &
            (
                df["severity"]
                == 0
            )
        ]


        if len(clear) != 1:

            raise RuntimeError(
                f"Trial {trial}: "
                "missing unique Clear baseline."
            )


        d0 = clear.iloc[0][
            "first_correct_detection_distance_m"
        ]


        if pd.isna(d0):
            d0 = 0.0


        d0 = float(
            d0
        )


        if d0 <= 0:

            raise RuntimeError(
                f"Trial {trial}: "
                f"invalid Clear D(0)={d0}"
            )


        indices = df[
            df["trial"]
            == trial
        ].index


        for idx in indices:

            d = df.loc[
                idx,
                "first_correct_detection_distance_m",
            ]


            if pd.isna(d):
                d = 0.0


            d = max(
                0.0,
                float(d),
            )


            df.loc[
                idx,
                "R_sign",
            ] = min(
                1.0,
                d / d0,
            )


    per_trial_path = (
        output_dir
        / "final_per_trial.csv"
    )


    df.to_csv(
        per_trial_path,
        index=False,
    )


    # ========================================================
    # AGGREGATE 5 CONTROLLED CONFIGURATIONS
    # ========================================================

    rows = []


    for severity in SEVERITIES:

        block = df[
            df["severity"]
            == severity
        ]


        distance = (

            block[
                "first_correct_detection_distance_m"
            ]
            .fillna(0.0)
            .astype(float)
        )


        rows.append({

            "severity":
                severity,

            "n_trials":
                len(block),

            "correct_detection_rate_mean_pct":
                float(
                    100.0
                    * block[
                        "correct_detection_rate"
                    ]
                    .astype(float)
                    .mean()
                ),

            "correct_detection_rate_std_pct":
                float(
                    100.0
                    * block[
                        "correct_detection_rate"
                    ]
                    .astype(float)
                    .std(
                        ddof=1
                    )
                ),

            "first_correct_distance_mean_m":
                float(
                    distance.mean()
                ),

            "first_correct_distance_std_m":
                float(
                    distance.std(
                        ddof=1
                    )
                ),

            "false_positive_mean":
                float(
                    block[
                        "false_positive_count"
                    ]
                    .astype(float)
                    .mean()
                ),

            "R_sign_mean":
                float(
                    block[
                        "R_sign"
                    ]
                    .mean()
                ),

            "R_sign_std":
                float(
                    block[
                        "R_sign"
                    ]
                    .std(
                        ddof=1
                    )
                ),
        })


    summary = pd.DataFrame(
        rows
    )


    summary_path = (
        output_dir
        / "final_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    return (
        df,
        summary,
        per_trial_path,
        summary_path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--detector",
        required=True,
        choices=[
            "yolo",
            "rfdetr",
        ],
    )

    args = parser.parse_args()


    settings = SETTINGS[
        args.detector
    ]


    threshold = settings[
        "threshold"
    ]


    output_dir = settings[
        "output"
    ]


    label = settings[
        "label"
    ]


    print()

    print(
        "=" * 90
    )

    print(
        f"FINAL {label} "
        f"STOP-SIGN SWEEP"
    )

    print(
        "=" * 90
    )

    print(
        "Frozen confidence threshold:",
        f"{threshold:.4f}",
    )

    print(
        "IoU correctness rule remains:",
        "IoU >= 0.50",
    )

    print(
        "Expected datasets:",
        35,
    )


    # --------------------------------------------------------
    # Import EXISTING detector implementation
    # --------------------------------------------------------

    detector = importlib.import_module(
        settings[
            "module"
        ]
    )


    # --------------------------------------------------------
    # Final datasets
    # --------------------------------------------------------

    datasets, order = (
        build_final_datasets()
    )


    detector.DATASETS = (
        datasets
    )


    # --------------------------------------------------------
    # Protect pilot results and direct all output
    # to results/final/
    # --------------------------------------------------------

    redirect_output_paths(
        detector,
        output_dir,
    )


    # --------------------------------------------------------
    # Never retune on final data
    # --------------------------------------------------------

    freeze_threshold(
        detector,
        threshold,
    )


    print()

    print(
        "Starting existing detector main()..."
    )

    print()


    # This preserves the actual detector,
    # preprocessing, model loading, class handling,
    # IoU evaluation, and inference code already
    # used in the pilot.
    # The detector's own main() has its own argparse parser.
    # Hide this wrapper's --detector argument from it.
    saved_argv = sys.argv[:]

    try:
        sys.argv = [sys.argv[0]]
        detector.main()

    finally:
        sys.argv = saved_argv


    raw_summary = (
        find_summary(
            output_dir
        )
    )


    (
        per_trial,
        summary,
        per_trial_path,
        summary_path,
    ) = make_final_summary(

        raw_summary,

        order,

        output_dir,
    )


    print()

    print(
        "=" * 110
    )

    print(
        f"FINAL {label} RESULTS"
    )

    print(
        "=" * 110
    )


    print(
        summary
        .round(4)
        .to_string(
            index=False
        )
    )


    print()

    print(
        "Reliability thresholds:"
    )


    for threshold_value in [
        0.40,
        0.50,
        0.60,
    ]:

        failed = summary[
            summary[
                "R_sign_mean"
            ]
            < threshold_value
        ]


        if len(
            failed
        ) == 0:

            print(
                f"R_sign < "
                f"{threshold_value:.2f}: "
                "not reached"
            )

        else:

            first = int(
                failed.iloc[0][
                    "severity"
                ]
            )

            print(
                f"R_sign < "
                f"{threshold_value:.2f}: "
                f"Fog {first}"
            )


    print()

    print(
        "Saved:"
    )

    print(
        per_trial_path
    )

    print(
        summary_path
    )

    print()

    print(
        f"PASS: final {label} "
        "STOP-sign analysis complete."
    )


if __name__ == "__main__":
    main()
