#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

ROOTS = {
    "automatic": ROOT / "data" / "recordings",
    "manual": ROOT / "data" / "recordings_manualexp",
}

OUT = ROOT / "results" / "final" / "manual_exposure_control"
OUT.mkdir(parents=True, exist_ok=True)

SEVERITIES = [0, 30, 45]
TRIALS = [1, 2, 3, 4, 5]

WINDOW_MIN = 20.0
WINDOW_MAX = 26.0


def rec_name(severity, trial):
    if severity == 0:
        return f"stop3_clear_0_drift_0p75_trial_{trial}"

    return f"stop3_fog_{severity}_drift_0p75_trial_{trial}"


# ============================================================
# FIND COMMON 20–26 m SAMPLE IDS ACROSS BOTH EXPOSURE MODES
# ============================================================

all_sets = []

for mode, root in ROOTS.items():

    for severity in SEVERITIES:

        for trial in TRIALS:

            rec = root / rec_name(severity, trial)

            csv_path = rec / "frames.csv"

            if not csv_path.exists():
                raise RuntimeError(
                    f"Missing CSV: {csv_path}"
                )

            df = pd.read_csv(csv_path)

            q = df[
                (df["distance_to_sign_m"] >= WINDOW_MIN)
                & (df["distance_to_sign_m"] <= WINDOW_MAX)
            ]

            ids = set(
                q["sample_id"]
                .astype(int)
                .tolist()
            )

            all_sets.append(ids)


common_ids = sorted(
    set.intersection(*all_sets)
)

print("=" * 90)
print("COMMON 20–26 m POSITIONS")
print("=" * 90)

print(f"Positions: {len(common_ids)}")
print(f"IDs: {common_ids}")

if not common_ids:
    raise RuntimeError(
        "No common 20–26 m positions."
    )


# ============================================================
# IMAGE METRICS
# ============================================================

rows = []

for mode, root in ROOTS.items():

    for severity in SEVERITIES:

        for trial in TRIALS:

            rec = root / rec_name(
                severity,
                trial,
            )

            frame_dir = rec / "frames"

            for sid in common_ids:

                frame_path = (
                    frame_dir
                    / f"{sid:04d}.png"
                )

                image = cv2.imread(
                    str(frame_path)
                )

                if image is None:
                    raise RuntimeError(
                        f"Could not read {frame_path}"
                    )

                gray = cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                ).astype(np.float64)

                mean_intensity = float(
                    gray.mean()
                )

                rms_contrast = float(
                    np.sqrt(
                        np.mean(
                            (
                                gray
                                - mean_intensity
                            ) ** 2
                        )
                    )
                )

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

                mean_gradient = float(
                    np.sqrt(
                        gx ** 2
                        + gy ** 2
                    ).mean()
                )

                rows.append({
                    "exposure_mode": mode,
                    "severity": severity,
                    "trial": trial,
                    "sample_id": sid,
                    "mean_intensity": mean_intensity,
                    "rms_contrast": rms_contrast,
                    "mean_gradient": mean_gradient,
                })


frame_df = pd.DataFrame(rows)

frame_df.to_csv(
    OUT / "per_frame.csv",
    index=False,
)


# ============================================================
# POSITION AVERAGE -> ONE VALUE PER TRIAL
# ============================================================

trial_df = (
    frame_df
    .groupby(
        [
            "exposure_mode",
            "severity",
            "trial",
        ],
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


# ============================================================
# NORMALIZE EACH EXPOSURE MODE TO ITS OWN CLEAR TRIAL
# ============================================================

clear = (
    trial_df[
        trial_df["severity"] == 0
    ][
        [
            "exposure_mode",
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
    clear,
    on=[
        "exposure_mode",
        "trial",
    ],
    validate="many_to_one",
)

trial_df["rms_ratio"] = (
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


# ============================================================
# SUMMARY ACROSS FIVE TRIALS
# ============================================================

summary = (
    trial_df
    .groupby(
        [
            "exposure_mode",
            "severity",
        ],
        as_index=False,
    )
    .agg(
        mean_intensity_mean=(
            "mean_intensity",
            "mean",
        ),
        mean_intensity_std=(
            "mean_intensity",
            lambda x: np.std(x, ddof=1),
        ),
        rms_ratio_mean=(
            "rms_ratio",
            "mean",
        ),
        rms_ratio_std=(
            "rms_ratio",
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


# ============================================================
# FOG30 VS FOG45 — PAIRED
# ============================================================

paired_rows = []

for mode in ["automatic", "manual"]:

    a = trial_df[
        (trial_df["exposure_mode"] == mode)
        & (trial_df["severity"] == 30)
    ]

    b = trial_df[
        (trial_df["exposure_mode"] == mode)
        & (trial_df["severity"] == 45)
    ]

    merged = a.merge(
        b,
        on="trial",
        suffixes=("_30", "_45"),
        validate="one_to_one",
    )

    for _, r in merged.iterrows():

        paired_rows.append({
            "exposure_mode": mode,
            "trial": int(r["trial"]),
            "intensity_30":
                r["mean_intensity_30"],
            "intensity_45":
                r["mean_intensity_45"],
            "intensity_45_minus_30":
                r["mean_intensity_45"]
                - r["mean_intensity_30"],

            "rms_ratio_30":
                r["rms_ratio_30"],
            "rms_ratio_45":
                r["rms_ratio_45"],
            "rms_45_minus_30":
                r["rms_ratio_45"]
                - r["rms_ratio_30"],

            "gradient_ratio_30":
                r["gradient_ratio_30"],
            "gradient_ratio_45":
                r["gradient_ratio_45"],
            "gradient_45_minus_30":
                r["gradient_ratio_45"]
                - r["gradient_ratio_30"],
        })


paired = pd.DataFrame(
    paired_rows
)

paired.to_csv(
    OUT / "fog30_vs_fog45_paired.csv",
    index=False,
)


# ============================================================
# FIGURE 1 — BRIGHTNESS
# ============================================================

plt.figure(figsize=(7, 5))

for mode in ["automatic", "manual"]:

    q = summary[
        summary["exposure_mode"] == mode
    ]

    plt.plot(
        q["severity"],
        q["mean_intensity_mean"],
        marker="o",
        linewidth=2,
        label=mode.capitalize(),
    )

plt.xlabel("Fog severity")
plt.ylabel("Mean grayscale intensity")
plt.title(
    "Fog30–Fog45 Brightness: Automatic vs Manual Exposure"
)
plt.xticks(SEVERITIES)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "fig_mean_intensity_auto_vs_manual.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 2 — RMS CONTRAST
# ============================================================

plt.figure(figsize=(7, 5))

for mode in ["automatic", "manual"]:

    q = summary[
        summary["exposure_mode"] == mode
    ]

    plt.errorbar(
        q["severity"],
        q["rms_ratio_mean"],
        yerr=q["rms_ratio_std"],
        marker="o",
        linewidth=2,
        capsize=4,
        label=mode.capitalize(),
    )

plt.xlabel("Fog severity")
plt.ylabel("RMS contrast remaining relative to clear")
plt.title(
    "Contrast Loss: Automatic vs Manual Exposure"
)
plt.xticks(SEVERITIES)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "fig_rms_auto_vs_manual.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 3 — GRADIENT DETAIL
# ============================================================

plt.figure(figsize=(7, 5))

for mode in ["automatic", "manual"]:

    q = summary[
        summary["exposure_mode"] == mode
    ]

    plt.errorbar(
        q["severity"],
        q["gradient_ratio_mean"],
        yerr=q["gradient_ratio_std"],
        marker="o",
        linewidth=2,
        capsize=4,
        label=mode.capitalize(),
    )

plt.xlabel("Fog severity")
plt.ylabel("Gradient detail remaining relative to clear")
plt.title(
    "Edge/Detail Loss: Automatic vs Manual Exposure"
)
plt.xticks(SEVERITIES)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "fig_gradient_auto_vs_manual.png",
    dpi=300,
)

plt.close()


# ============================================================
# PRINT
# ============================================================

print()
print("=" * 90)
print("AUTO VS MANUAL — SUMMARY")
print("=" * 90)

print(
    summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("=" * 90)
print("FOG30 VS FOG45 — PAIRED")
print("=" * 90)

print(
    paired.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print()
print("=" * 90)
print("SAVED")
print("=" * 90)

for p in sorted(OUT.iterdir()):
    print(p)
