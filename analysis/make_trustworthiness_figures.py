#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

FOURIER = ROOT / "results" / "final" / "fourier_monitor"
OUT = ROOT / "results" / "final" / "figures_trustworthiness"
OUT.mkdir(parents=True, exist_ok=True)

SEVERITIES = [0, 15, 30, 45, 60, 75, 90]
TRIALS = [1, 2, 3, 4, 5]

# Detector effect of interest happened around here.
WINDOW_MIN_M = 20.0
WINDOW_MAX_M = 26.0


# ============================================================
# LOAD FINAL FOURIER RESULTS
# ============================================================

severity = pd.read_csv(
    FOURIER / "severity_summary.csv"
)

trial = pd.read_csv(
    FOURIER / "trial_distances.csv"
)

position = pd.read_csv(
    FOURIER / "position_distances.csv"
)

threshold_df = pd.read_csv(
    FOURIER / "threshold.csv"
)

threshold = float(
    threshold_df[
        "threshold_mu0_plus_3sigma0"
    ].iloc[0]
)


# ============================================================
# FIGURE A
# GLOBAL FOURIER DEGRADATION
# ============================================================

# Use x10^-6 units so the figure is readable.
x = severity["severity"].to_numpy()

y = (
    severity["mean_spectral_distance"].to_numpy()
    * 1e6
)

yerr = (
    severity["std_spectral_distance"].to_numpy()
    * 1e6
)

plt.figure(figsize=(7.5, 5.0))

plt.errorbar(
    x,
    y,
    yerr=yerr,
    marker="o",
    linewidth=2,
    capsize=4,
)

plt.axhline(
    threshold * 1e6,
    linestyle="--",
    label="Clear threshold μ₀ + 3σ₀",
)

plt.xlabel("Fog severity")
plt.ylabel("Global spectral distance (×10⁻⁶)")
plt.title("Global Image Degradation vs Fog Severity")
plt.xticks(SEVERITIES)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "fig_global_fourier.png",
    dpi=300,
)

plt.close()


# ============================================================
# DEFINE EXACT MATCHED 20–26 m WINDOW
# ============================================================

# Define the sample window from CLEAR data.
#
# Require a sample ID to lie inside 20–26 m
# in ALL five clear trials.
#
# This avoids slightly changing the selected positions
# from trial to trial due to tiny geometric differences.

clear = position[
    position["severity"] == 0
].copy()

window_sets = []

for t in TRIALS:

    q = clear[
        (clear["trial"] == t)
        & (
            clear["distance_to_sign_m"]
            >= WINDOW_MIN_M
        )
        & (
            clear["distance_to_sign_m"]
            <= WINDOW_MAX_M
        )
    ]

    window_sets.append(
        set(
            q["sample_id"]
            .astype(int)
            .tolist()
        )
    )

window_ids = sorted(
    set.intersection(
        *window_sets
    )
)

if not window_ids:
    raise RuntimeError(
        "No common 20–26 m sample IDs."
    )

print("\n" + "=" * 80)
print("POST-HOC 20–26 m WINDOW")
print("=" * 80)
print(f"Common sample IDs: {len(window_ids)}")
print(
    f"sample_id range: "
    f"{window_ids[0]} -> {window_ids[-1]}"
)


# ============================================================
# FILTER ALL CONDITIONS TO SAME IDS
# ============================================================

local = position[
    position["sample_id"].isin(
        window_ids
    )
].copy()

expected = len(window_ids)

counts = (
    local
    .groupby(
        ["severity", "trial"]
    )["sample_id"]
    .nunique()
)

if not (counts == expected).all():
    print(counts)
    raise RuntimeError(
        "20–26 m sample matching failed."
    )


# ============================================================
# POSITION FIRST -> ONE VALUE PER TRIAL
# ============================================================

local_trial = (
    local
    .groupby(
        ["severity", "trial"],
        as_index=False,
    )
    .agg(
        spectral_distance=(
            "spectral_distance_l1",
            "mean",
        ),
        positions_n=(
            "sample_id",
            "nunique",
        ),
    )
    .sort_values(
        ["severity", "trial"]
    )
)

local_trial.to_csv(
    OUT / "fourier_20_26m_per_trial.csv",
    index=False,
)


# Across-trial summary
local_summary = (
    local_trial
    .groupby(
        "severity",
        as_index=False,
    )
    .agg(
        mean_spectral_distance=(
            "spectral_distance",
            "mean",
        ),
        std_spectral_distance=(
            "spectral_distance",
            lambda z:
                np.std(z, ddof=1),
        ),
        trials_n=(
            "trial",
            "count",
        ),
    )
)

local_summary.to_csv(
    OUT / "fourier_20_26m_summary.csv",
    index=False,
)


# ============================================================
# FIGURE B
# LOCAL FOURIER DEGRADATION AROUND SIGN
# ============================================================

plt.figure(figsize=(7.5, 5.0))

plt.errorbar(
    local_summary["severity"],
    (
        local_summary[
            "mean_spectral_distance"
        ]
        * 1e6
    ),
    yerr=(
        local_summary[
            "std_spectral_distance"
        ]
        * 1e6
    ),
    marker="o",
    linewidth=2,
    capsize=4,
)

plt.xlabel("Fog severity")
plt.ylabel(
    "20–26 m spectral distance (×10⁻⁶)"
)
plt.title(
    "Post-hoc Local Fourier Analysis Near STOP Detection"
)
plt.xticks(SEVERITIES)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUT / "fig_fourier_20_26m.png",
    dpi=300,
)

plt.close()


# ============================================================
# FOG30 VS FOG45: PAIRED BY TRIAL
# ============================================================

d30 = (
    local_trial[
        local_trial["severity"] == 30
    ][
        ["trial", "spectral_distance"]
    ]
    .rename(
        columns={
            "spectral_distance":
                "fog30_distance"
        }
    )
)

d45 = (
    local_trial[
        local_trial["severity"] == 45
    ][
        ["trial", "spectral_distance"]
    ]
    .rename(
        columns={
            "spectral_distance":
                "fog45_distance"
        }
    )
)

paired = d30.merge(
    d45,
    on="trial",
    validate="one_to_one",
)

paired[
    "fog45_minus_fog30"
] = (
    paired["fog45_distance"]
    - paired["fog30_distance"]
)

paired.to_csv(
    OUT
    / "fourier_20_26m_fog30_vs_fog45.csv",
    index=False,
)


# ============================================================
# FIGURE C
# PAIRED FOG30 -> FOG45
# ============================================================

plt.figure(figsize=(6.0, 5.0))

for _, row in paired.iterrows():

    plt.plot(
        [30, 45],
        [
            row["fog30_distance"] * 1e6,
            row["fog45_distance"] * 1e6,
        ],
        marker="o",
        linewidth=1.5,
        label=f"Trial {int(row['trial'])}",
    )

plt.xlabel("Fog severity")
plt.ylabel(
    "20–26 m spectral distance (×10⁻⁶)"
)
plt.title(
    "Fog30 → Fog45: Paired Local Spectral Change"
)
plt.xticks([30, 45])
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUT / "fig_fog30_fog45_paired_local.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE D
# MONITOR WARNING VS STOP-TASK FAILURE
#
# These are the final threshold crossings already observed:
#
# Fourier monitor: fog15
# Template STOP R < 0.50: fog15
# HSV STOP R < 0.50: fog30
# YOLO: no crossing through fog90
# RF-DETR: no crossing through fog90
#
# This figure is about WARNING TIMING, not comparing
# metric magnitudes.
# ============================================================

labels = [
    "Fourier degradation alert",
    "Template STOP reliability",
    "HSV STOP reliability",
    "YOLO STOP reliability",
    "RF-DETR STOP reliability",
]

values = [
    15,
    15,
    30,
    90,
    90,
]

no_cross = [
    False,
    False,
    False,
    True,
    True,
]

ypos = np.arange(
    len(labels)
)

plt.figure(figsize=(8.5, 4.8))

for i, (value, nc) in enumerate(
    zip(values, no_cross)
):
    plt.scatter(
        value,
        i,
        s=70,
    )

    if nc:
        plt.annotate(
            ">90 (no failure crossing)",
            (value, i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
        )
    else:
        plt.annotate(
            f"Fog {value}",
            (value, i),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
        )

plt.yticks(
    ypos,
    labels,
)

plt.xticks(
    SEVERITIES
)

plt.xlim(
    -3,
    108,
)

plt.xlabel(
    "Fog severity at first threshold crossing"
)

plt.title(
    "Input Degradation Warning Does Not Equal Task Failure"
)

plt.grid(
    axis="x",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    OUT / "fig_monitor_vs_task_thresholds.png",
    dpi=300,
)

plt.close()


# ============================================================
# EXACT NUMBERS
# ============================================================

print("\n" + "=" * 80)
print("GLOBAL FOURIER — EXACT")
print("=" * 80)

with pd.option_context(
    "display.float_format",
    lambda x: f"{x:.10e}",
):
    print(
        severity[
            [
                "severity",
                "mean_spectral_distance",
                "std_spectral_distance",
            ]
        ].to_string(
            index=False
        )
    )


print("\n" + "=" * 80)
print("20–26 m FOURIER — EXACT")
print("=" * 80)

with pd.option_context(
    "display.float_format",
    lambda x: f"{x:.10e}",
):
    print(
        local_summary.to_string(
            index=False
        )
    )


print("\n" + "=" * 80)
print("20–26 m FOG30 VS FOG45")
print("=" * 80)

with pd.option_context(
    "display.float_format",
    lambda x: f"{x:.10e}",
):
    print(
        paired.to_string(
            index=False
        )
    )


delta = paired[
    "fog45_minus_fog30"
].to_numpy()

print(
    "\nMean paired Fog45 - Fog30 = "
    f"{delta.mean():.10e}"
)

print(
    "Fog45 more degraded: "
    f"{int((delta > 0).sum())}/5"
)

print(
    "Fog45 less degraded: "
    f"{int((delta < 0).sum())}/5"
)


print("\n" + "=" * 80)
print("FIGURES SAVED")
print("=" * 80)

for f in sorted(
    OUT.glob("*.png")
):
    print(f)
