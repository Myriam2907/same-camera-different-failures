#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

from sign import hsv_stop as hsv


ROOT = Path(__file__).resolve().parent

OUT = (
    ROOT
    / "results"
    / "final"
    / "stop_hsv"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

# Make the existing detector write into
# the FINAL results directory.
hsv.OUTPUT_ROOT = str(OUT)


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


def condition_name(severity):

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


def main():

    print()
    print("=" * 80)
    print("FINAL HSV STOP-SIGN SWEEP")
    print("=" * 80)

    summaries = []


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


            if not (
                root
                / "frames.csv"
            ).exists():

                raise RuntimeError(
                    f"Missing dataset: {root}"
                )


            print()
            print(
                "=" * 70
            )

            print(
                f"{condition.upper()} "
                f"| trial {trial}"
            )

            print(
                "=" * 70
            )


            dataset_info = {

                "root":
                    str(root),

                "severity":
                    severity,
            }


            summary = (
                hsv.process_dataset(
                    dataset_name,
                    dataset_info,
                )
            )


            summary[
                "trial"
            ] = trial

            summary[
                "severity"
            ] = severity

            summary[
                "condition"
            ] = condition


            summaries.append(
                summary
            )


            print(
                "Correct detection rate:",
                f"{100.0 * summary['correct_detection_rate']:.2f}%"
            )

            print(
                "False positives:",
                summary[
                    "false_positive_count"
                ]
            )

            print(
                "First correct distance:",
                summary[
                    "first_correct_detection_distance_m"
                ]
            )


    df = pd.DataFrame(
        summaries
    )


    per_trial_path = (
        OUT
        / "final_per_trial.csv"
    )


    df.to_csv(
        per_trial_path,
        index=False,
    )


    # ========================================================
    # PAIRED R_sign
    #
    # R_sign,j(s) =
    # min(1, D_j(s) / D_j(0))
    #
    # D=0 if there is no correct detection.
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
                f"clear baseline missing."
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
                f"clear D(0) is invalid: {d0}"
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


            d = float(
                d
            )


            r = min(
                1.0,
                d / d0,
            )


            df.loc[
                idx,
                "R_sign",
            ] = r


    # overwrite with R_sign included
    df.to_csv(
        per_trial_path,
        index=False,
    )


    # ========================================================
    # SUMMARY ACROSS FIVE CONTROLLED CONFIGURATIONS
    # ========================================================

    rows = []


    for severity in SEVERITIES:

        block = df[
            df["severity"]
            == severity
        ]


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
                    ].mean()
                ),

            "correct_detection_rate_std_pct":
                float(
                    100.0
                    * block[
                        "correct_detection_rate"
                    ].std(
                        ddof=1
                    )
                ),

            "first_correct_distance_mean_m":
                float(
                    block[
                        "first_correct_detection_distance_m"
                    ]
                    .fillna(0.0)
                    .mean()
                ),

            "first_correct_distance_std_m":
                float(
                    block[
                        "first_correct_detection_distance_m"
                    ]
                    .fillna(0.0)
                    .std(
                        ddof=1
                    )
                ),

            "false_positive_mean":
                float(
                    block[
                        "false_positive_count"
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
        rows
    )


    summary_path = (
        OUT
        / "final_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    print()
    print("=" * 100)
    print("FINAL HSV RESULTS")
    print("=" * 100)

    print(
        summary.round(4).to_string(
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
            summary["R_sign_mean"]
            < threshold
        ]


        if len(failed) == 0:

            print(
                f"R_sign < {threshold:.2f}: "
                f"not reached"
            )

        else:

            severity = int(
                failed.iloc[0][
                    "severity"
                ]
            )

            print(
                f"R_sign < {threshold:.2f}: "
                f"Fog {severity}"
            )


    print()
    print("Saved:")
    print(per_trial_path)
    print(summary_path)

    print()
    print(
        "PASS: final HSV "
        "STOP-sign analysis complete."
    )


if __name__ == "__main__":

    main()
