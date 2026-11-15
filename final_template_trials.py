#!/usr/bin/env python3

from pathlib import Path
import pandas as pd

from sign import template_stop as template


ROOT = Path(__file__).resolve().parent

OUT = (
    ROOT
    / "results"
    / "final"
    / "stop_template"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

template.OUTPUT_ROOT = str(OUT)

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
    print("FINAL TEMPLATE STOP-SIGN SWEEP")
    print("=" * 80)

    print()
    print("Creating frozen STOP template...")

    (
        stop_template,
        template_mask,
    ) = template.create_template()


    # IMPORTANT:
    # This calls the existing pilot Clear-only calibration.
    # It does NOT use any of the 35 final Fog results.
    threshold = template.calibrate_threshold(
        stop_template,
        template_mask,
    )


    print()
    print(
        "Frozen template threshold:",
        threshold,
    )

    print(
        "The final Fog trials do NOT "
        "influence this threshold."
    )


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
            print("=" * 70)

            print(
                f"{condition.upper()} "
                f"| trial {trial}"
            )

            print("=" * 70)


            dataset_info = {

                "root":
                    str(root),

                "severity":
                    severity,
            }


            summary = (
                template.process_dataset(
                    dataset_name,
                    dataset_info,
                    stop_template,
                    template_mask,
                    threshold,
                )
            )


            summary["trial"] = trial

            summary["severity"] = severity

            summary["condition"] = condition


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


    # ========================================================
    # PAIRED R_sign
    # ========================================================

    df["R_sign"] = 0.0


    for trial in TRIALS:

        clear = df[
            (df["trial"] == trial)
            &
            (df["severity"] == 0)
        ]


        if len(clear) != 1:

            raise RuntimeError(
                f"Trial {trial}: "
                "missing Clear baseline."
            )


        d0 = clear.iloc[0][
            "first_correct_detection_distance_m"
        ]


        if pd.isna(d0):
            d0 = 0.0


        d0 = float(d0)


        if d0 <= 0:

            raise RuntimeError(
                f"Trial {trial}: "
                f"invalid Clear D(0)={d0}"
            )


        idx = df[
            df["trial"] == trial
        ].index


        for i in idx:

            d = df.loc[
                i,
                "first_correct_detection_distance_m",
            ]


            if pd.isna(d):
                d = 0.0


            r = min(
                1.0,
                float(d) / d0,
            )


            df.loc[
                i,
                "R_sign",
            ] = r


    per_trial_path = (
        OUT
        / "final_per_trial.csv"
    )


    df.to_csv(
        per_trial_path,
        index=False,
    )


    # ========================================================
    # FIVE-CONFIG SUMMARY
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


    result = pd.DataFrame(
        rows
    )


    summary_path = (
        OUT
        / "final_summary.csv"
    )


    result.to_csv(
        summary_path,
        index=False,
    )


    print()
    print("=" * 100)
    print("FINAL TEMPLATE RESULTS")
    print("=" * 100)

    print(
        result
        .round(4)
        .to_string(
            index=False
        )
    )


    print()
    print("Reliability thresholds:")


    for threshold_value in [
        0.40,
        0.50,
        0.60,
    ]:

        failed = result[
            result[
                "R_sign_mean"
            ]
            < threshold_value
        ]


        if len(failed) == 0:

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
    print("Saved:")
    print(per_trial_path)
    print(summary_path)

    print()
    print(
        "PASS: final Template "
        "STOP-sign analysis complete."
    )


if __name__ == "__main__":
    main()
