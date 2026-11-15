#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RECORDINGS = ROOT / "data" / "recordings"
POSITION_CSV = (
    ROOT / "results" / "final" /
    "fourier_monitor" / "position_distances.csv"
)

OUT = (
    ROOT / "results" / "final" /
    "local_magnitude_audit"
)

OUT.mkdir(parents=True, exist_ok=True)

SEVERITIES = [0, 15, 30, 45, 60, 75, 90]
TRIALS = [1, 2, 3, 4, 5]

WINDOW_MIN = 20.0
WINDOW_MAX = 26.0


def recording_dir(severity, trial):
    weather = "clear" if severity == 0 else "fog"

    p = RECORDINGS / (
        f"stop3_{weather}_{severity}_"
        f"drift_0p75_trial_{trial}"
    )

    if not p.is_dir():
        raise RuntimeError(f"Missing: {p}")

    return p


# ------------------------------------------------------------
# Recover the exact same common 20–26 m positions
# ------------------------------------------------------------

pos = pd.read_csv(POSITION_CSV)

clear = pos[pos["severity"] == 0].copy()

sets = []

for trial in TRIALS:
    q = clear[
        (clear["trial"] == trial)
        & (clear["distance_to_sign_m"] >= WINDOW_MIN)
        & (clear["distance_to_sign_m"] <= WINDOW_MAX)
    ]

    sets.append(set(q["sample_id"].astype(int)))

sample_ids = sorted(set.intersection(*sets))

print("=" * 80)
print("20–26 m MAGNITUDE AUDIT")
print("=" * 80)
print(f"Common positions: {len(sample_ids)}")
print(f"IDs: {sample_ids}")

if len(sample_ids) != 12:
    print(
        "WARNING: expected 12 positions from "
        "the previous analysis."
    )


# ------------------------------------------------------------
# Direct image magnitude measurements
# ------------------------------------------------------------

rows = []

for severity in SEVERITIES:
    for trial in TRIALS:

        frame_dir = (
            recording_dir(severity, trial)
            / "frames"
        )

        for sample_id in sample_ids:

            path = frame_dir / f"{sample_id:04d}.png"

            image = cv2.imread(str(path))

            if image is None:
                raise RuntimeError(
                    f"Could not read {path}"
                )

            gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            ).astype(np.float64)

            # Direct RMS contrast:
            # standard deviation of pixel intensity.
            mean_intensity = float(gray.mean())

            rms_contrast = float(
                np.sqrt(
                    np.mean(
                        (gray - mean_intensity) ** 2
                    )
                )
            )

            # Direct edge/detail magnitude.
            gx = cv2.Sobel(
                gray,
                cv2.CV_64F,
                1,
                0,
                ksize=3,
            )

            gy = cv2.Sobel(
                gray,
                cv2.CV_64F,
                0,
                1,
                ksize=3,
            )

            gradient = np.sqrt(
                gx ** 2 + gy ** 2
            )

            mean_gradient = float(
                gradient.mean()
            )

            rows.append({
                "severity": severity,
                "trial": trial,
                "sample_id": sample_id,
                "mean_intensity": mean_intensity,
                "rms_contrast": rms_contrast,
                "mean_gradient": mean_gradient,
            })


frame_df = pd.DataFrame(rows)

frame_df.to_csv(
    OUT / "per_frame.csv",
    index=False,
)


# ------------------------------------------------------------
# Position first -> one value per trial
# ------------------------------------------------------------

trial_df = (
    frame_df
    .groupby(
        ["severity", "trial"],
        as_index=False,
    )
    .agg(
        mean_intensity=(
            "mean_intensity",
            "mean",
        ),
        rms_contrast=(
            "rms_contrast",
            "mean",
        ),
        mean_gradient=(
            "mean_gradient",
            "mean",
        ),
        positions_n=(
            "sample_id",
            "nunique",
        ),
    )
)


# ------------------------------------------------------------
# Normalize magnitude to each trial's own clear value
#
# 1.0 = clear
# lower = less contrast/detail remains
# ------------------------------------------------------------

clear_trial = (
    trial_df[
        trial_df["severity"] == 0
    ][
        [
            "trial",
            "rms_contrast",
            "mean_gradient",
        ]
    ]
    .rename(
        columns={
            "rms_contrast":
                "clear_rms_contrast",
            "mean_gradient":
                "clear_mean_gradient",
        }
    )
)

trial_df = trial_df.merge(
    clear_trial,
    on="trial",
    how="left",
    validate="many_to_one",
)

trial_df["rms_contrast_ratio"] = (
    trial_df["rms_contrast"]
    / trial_df["clear_rms_contrast"]
)

trial_df["gradient_ratio"] = (
    trial_df["mean_gradient"]
    / trial_df["clear_mean_gradient"]
)

trial_df.to_csv(
    OUT / "per_trial.csv",
    index=False,
)


# ------------------------------------------------------------
# Severity summary
# ------------------------------------------------------------

summary = (
    trial_df
    .groupby(
        "severity",
        as_index=False,
    )
    .agg(
        mean_intensity=(
            "mean_intensity",
            "mean",
        ),
        rms_contrast_ratio_mean=(
            "rms_contrast_ratio",
            "mean",
        ),
        rms_contrast_ratio_std=(
            "rms_contrast_ratio",
            lambda x: np.std(x, ddof=1),
        ),
        gradient_ratio_mean=(
            "gradient_ratio",
            "mean",
        ),
        gradient_ratio_std=(
            "gradient_ratio",
            lambda x: np.std(x, ddof=1),
        ),
    )
)

summary.to_csv(
    OUT / "summary.csv",
    index=False,
)


# ------------------------------------------------------------
# Fog30 vs Fog45 paired audit
# ------------------------------------------------------------

q30 = (
    trial_df[
        trial_df["severity"] == 30
    ][
        [
            "trial",
            "rms_contrast_ratio",
            "gradient_ratio",
        ]
    ]
    .rename(
        columns={
            "rms_contrast_ratio":
                "fog30_rms_ratio",
            "gradient_ratio":
                "fog30_gradient_ratio",
        }
    )
)

q45 = (
    trial_df[
        trial_df["severity"] == 45
    ][
        [
            "trial",
            "rms_contrast_ratio",
            "gradient_ratio",
        ]
    ]
    .rename(
        columns={
            "rms_contrast_ratio":
                "fog45_rms_ratio",
            "gradient_ratio":
                "fog45_gradient_ratio",
        }
    )
)

paired = q30.merge(
    q45,
    on="trial",
    validate="one_to_one",
)

paired["rms_45_minus_30"] = (
    paired["fog45_rms_ratio"]
    - paired["fog30_rms_ratio"]
)

paired["gradient_45_minus_30"] = (
    paired["fog45_gradient_ratio"]
    - paired["fog30_gradient_ratio"]
)

paired.to_csv(
    OUT / "fog30_vs_fog45.csv",
    index=False,
)


print("\n" + "=" * 80)
print("DIRECT MAGNITUDE SUMMARY")
print("=" * 80)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\n" + "=" * 80)
print("FOG30 VS FOG45 — PAIRED")
print("=" * 80)

print(
    paired.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\nInterpretation:")
print(
    "Lower RMS/gradient ratio = less visual "
    "contrast/detail remains relative to clear."
)
