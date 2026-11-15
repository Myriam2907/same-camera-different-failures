#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "results"
    / "final"
    / "paper_figures"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEVERITIES = [
    0, 15, 30, 45, 60, 75, 90
]


# ============================================================
# INPUT FILES
# ============================================================

STOP_FILES = {
    "HSV": (
        ROOT
        / "results"
        / "final"
        / "stop_hsv"
        / "final_summary.csv"
    ),

    "Template": (
        ROOT
        / "results"
        / "final"
        / "stop_template"
        / "final_summary.csv"
    ),

    "YOLO": (
        ROOT
        / "results"
        / "final"
        / "stop_yolo"
        / "final_summary.csv"
    ),

    "RF-DETR": (
        ROOT
        / "results"
        / "final"
        / "stop_rfdetr"
        / "final_summary.csv"
    ),
}


STOP_TRIAL_FILES = {
    "HSV": (
        ROOT
        / "results"
        / "final"
        / "stop_hsv"
        / "final_per_trial.csv"
    ),
}


LANE_SUMMARY = (
    ROOT
    / "results"
    / "final"
    / "lane_ufld_range"
    / "final_summary.csv"
)


MAGNITUDE_SUMMARY = (
    ROOT
    / "results"
    / "final"
    / "local_magnitude_audit"
    / "summary.csv"
)


FOURIER_POSITIONS = (
    ROOT
    / "results"
    / "final"
    / "fourier_monitor"
    / "position_distances.csv"
)


MANUAL_EXPOSURE_SUMMARY = (
    ROOT
    / "results"
    / "final"
    / "manual_exposure_control"
    / "summary.csv"
)


MANUAL_HSV_SUMMARY = (
    ROOT
    / "results"
    / "final"
    / "stop_hsv_manualexp"
    / "manual_summary.csv"
)


MANUAL_HSV_TRIALS = (
    ROOT
    / "results"
    / "final"
    / "stop_hsv_manualexp"
    / "manual_per_trial.csv"
)


# ============================================================
# PLOT SETTINGS
# ============================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 400,
})


def require(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file missing:\n{path}"
        )


def save_figure(fig, stem):
    png = OUT / f"{stem}.png"
    pdf = OUT / f"{stem}.pdf"

    fig.savefig(
        png,
        bbox_inches="tight",
        dpi=400,
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
    )

    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


# ============================================================
# LOAD STOP RESULTS
# ============================================================

stop = {}

for name, path in STOP_FILES.items():

    require(path)

    df = pd.read_csv(path)

    df = df.sort_values(
        "severity"
    ).reset_index(drop=True)

    stop[name] = df


# ============================================================
# LOAD LANE SUMMARY
# ============================================================

require(LANE_SUMMARY)

lane = pd.read_csv(
    LANE_SUMMARY
)


def get_lane_row(name):

    if "row_label" not in lane.columns:
        raise RuntimeError(
            "lane final_summary.csv does not contain "
            "'row_label'. Columns are:\n"
            + str(list(lane.columns))
        )

    q = lane[
        lane["row_label"] == name
    ].copy()

    if q.empty:
        raise RuntimeError(
            f"No lane rows found for {name}"
        )

    return (
        q.sort_values("severity")
        .reset_index(drop=True)
    )


lane_near = get_lane_row(
    "primary_504"
)

lane_25 = get_lane_row(
    "target_25m"
)


# ============================================================
# FIGURE 1
#
# RELIABILITY HEATMAP
#
# This is intentionally a task-specific normalized reliability
# comparison. R_lane and R_sign have different task definitions,
# but both equal 1 at clear and decrease toward failure.
# ============================================================

rows = [
    (
        "Lane UFLD\n8.9 m",
        lane_near["R_lane_mean"].to_numpy()
    ),
    (
        "Lane UFLD\n25.1 m",
        lane_25["R_lane_mean"].to_numpy()
    ),
]

for name in [
    "HSV",
    "Template",
    "YOLO",
    "RF-DETR",
]:

    df = stop[name]

    rows.append(
        (
            f"STOP\n{name}",
            df["R_sign_mean"].to_numpy()
        )
    )


labels = [
    x[0]
    for x in rows
]

matrix = np.vstack([
    x[1]
    for x in rows
])


fig, ax = plt.subplots(
    figsize=(9.5, 5.1)
)

im = ax.imshow(
    matrix,
    aspect="auto",
    vmin=0,
    vmax=1,
    cmap="viridis",
)

ax.set_xticks(
    np.arange(
        len(SEVERITIES)
    )
)

ax.set_xticklabels(
    SEVERITIES
)

ax.set_yticks(
    np.arange(
        len(labels)
    )
)

ax.set_yticklabels(
    labels
)

ax.set_xlabel(
    "Fog density"
)

ax.set_title(
    "Same Camera, Different Reliability"
)

for i in range(
    matrix.shape[0]
):
    for j in range(
        matrix.shape[1]
    ):

        value = matrix[i, j]

        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8,
            color=(
                "white"
                if value < 0.45
                else "black"
            ),
        )


cbar = fig.colorbar(
    im,
    ax=ax,
    pad=0.02,
)

cbar.set_label(
    "Task-specific normalized reliability"
)

fig.text(
    0.5,
    -0.01,
    (
        "R=1 denotes clear-condition reliability. "
        "Lane and STOP use task-specific reliability definitions."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout()

save_figure(
    fig,
    "fig1_reliability_heatmap",
)

plt.close(fig)


# ============================================================
# FIGURE 2
#
# STOP RELIABILITY CURVES
# ============================================================

fig, ax = plt.subplots(
    figsize=(8.5, 5.3)
)

markers = [
    "o",
    "s",
    "^",
    "D",
]

for marker, name in zip(
    markers,
    [
        "HSV",
        "Template",
        "YOLO",
        "RF-DETR",
    ],
):

    df = stop[name]

    ax.errorbar(
        df["severity"],
        df["R_sign_mean"],
        yerr=df["R_sign_std"],
        marker=marker,
        linewidth=2,
        markersize=6,
        capsize=3,
        label=name,
    )


ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.5,
    label="Reliability threshold = 0.50",
)


ax.set_xlim(
    -3,
    93,
)

ax.set_ylim(
    -0.03,
    1.08,
)

ax.set_xticks(
    SEVERITIES
)

ax.set_xlabel(
    "Fog density"
)

ax.set_ylabel(
    r"$R_{\mathrm{sign}}$"
)

ax.set_title(
    "STOP-Sign Reliability Depends Strongly on Detector"
)

ax.grid(
    alpha=0.25,
)

ax.legend(
    ncol=2,
    frameon=False,
)


# Mark first threshold crossing.

for name in [
    "HSV",
    "Template",
    "YOLO",
    "RF-DETR",
]:

    df = stop[name]

    bad = df[
        df["R_sign_mean"] < 0.5
    ]

    if not bad.empty:

        first = bad.iloc[0]

        ax.annotate(
            f"{name}: Fog {int(first['severity'])}",
            xy=(
                first["severity"],
                first["R_sign_mean"],
            ),
            xytext=(5, -18),
            textcoords="offset points",
            fontsize=8,
        )


fig.tight_layout()

save_figure(
    fig,
    "fig2_stop_reliability_curves",
)

plt.close(fig)


# ============================================================
# FIGURE 3
#
# CORE TRUSTWORTHINESS FIGURE:
#
# A. Direct image information remaining
# B. Fourier spectral-shape distance
# C. Task reliability / first-detection distance
#
# Same severity axis.
# ============================================================

require(
    MAGNITUDE_SUMMARY
)

mag = pd.read_csv(
    MAGNITUDE_SUMMARY
).sort_values(
    "severity"
)


# ------------------------------------------------------------
# Local Fourier 20-26 m
# Use the already-frozen matched route positions 104-115.
# ------------------------------------------------------------

require(
    FOURIER_POSITIONS
)

fourier = pd.read_csv(
    FOURIER_POSITIONS
)


required = {
    "severity",
    "trial",
    "sample_id",
}

missing = (
    required
    - set(fourier.columns)
)

if missing:
    raise RuntimeError(
        "Fourier position CSV missing columns: "
        + str(sorted(missing))
        + "\nAvailable: "
        + str(list(fourier.columns))
    )


spectral_candidates = [
    c
    for c in fourier.columns
    if (
        "spectral" in c.lower()
        and "distance" in c.lower()
    )
]

if len(
    spectral_candidates
) == 0:

    # Fallback: any non-sign distance column
    spectral_candidates = [
        c
        for c in fourier.columns
        if (
            "distance" in c.lower()
            and c
            not in {
                "distance_to_sign_m",
            }
        )
    ]


if len(
    spectral_candidates
) != 1:

    raise RuntimeError(
        "Could not uniquely identify "
        "Fourier spectral-distance column.\n"
        f"Candidates: {spectral_candidates}\n"
        f"Columns: {list(fourier.columns)}"
    )


spectral_col = (
    spectral_candidates[0]
)


LOCAL_IDS = list(
    range(
        104,
        116,
    )
)


local_fourier = fourier[
    fourier["sample_id"].isin(
        LOCAL_IDS
    )
].copy()


local_trial = (
    local_fourier
    .groupby(
        [
            "severity",
            "trial",
        ],
        as_index=False,
    )[spectral_col]
    .mean()
)


local_summary = (
    local_trial
    .groupby(
        "severity",
        as_index=False,
    )
    .agg(
        spectral_mean=(
            spectral_col,
            "mean",
        ),
        spectral_std=(
            spectral_col,
            lambda x:
                np.std(
                    x,
                    ddof=1,
                ),
        ),
    )
)


hsv = stop["HSV"]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(15.2, 4.7),
)


# ------------------------------------------------------------
# PANEL A
# ------------------------------------------------------------

ax = axes[0]

ax.errorbar(
    mag["severity"],
    mag[
        "rms_contrast_ratio_mean"
    ],
    yerr=mag[
        "rms_contrast_ratio_std"
    ],
    marker="o",
    linewidth=2,
    capsize=3,
    label="RMS contrast",
)

ax.errorbar(
    mag["severity"],
    mag[
        "gradient_ratio_mean"
    ],
    yerr=mag[
        "gradient_ratio_std"
    ],
    marker="s",
    linewidth=2,
    capsize=3,
    label="Gradient detail",
)

ax.axvspan(
    30,
    45,
    alpha=0.08,
)

ax.set_xticks(
    SEVERITIES
)

ax.set_ylim(
    0,
    1.08,
)

ax.set_xlabel(
    "Fog density"
)

ax.set_ylabel(
    "Information remaining\n(relative to clear)"
)

ax.set_title(
    "(a) Direct image measures"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)


# ------------------------------------------------------------
# PANEL B
# ------------------------------------------------------------

ax = axes[1]

scale = 1e5

ax.errorbar(
    local_summary["severity"],
    local_summary[
        "spectral_mean"
    ] * scale,
    yerr=(
        local_summary[
            "spectral_std"
        ] * scale
    ),
    marker="o",
    linewidth=2,
    capsize=3,
)

ax.axvspan(
    30,
    45,
    alpha=0.08,
)

ax.set_xticks(
    SEVERITIES
)

ax.set_xlabel(
    "Fog density"
)

ax.set_ylabel(
    r"Spectral-shape distance $\times 10^{5}$"
)

ax.set_title(
    "(b) Fourier monitor, 20-26 m"
)

ax.grid(
    alpha=0.25
)


# ------------------------------------------------------------
# PANEL C
# ------------------------------------------------------------

ax = axes[2]

ax.errorbar(
    hsv["severity"],
    hsv[
        "first_correct_distance_mean_m"
    ],
    yerr=hsv[
        "first_correct_distance_std_m"
    ],
    marker="o",
    linewidth=2,
    capsize=3,
)

ax.axvspan(
    30,
    45,
    alpha=0.08,
)

ax.set_xticks(
    SEVERITIES
)

ax.set_xlabel(
    "Fog density"
)

ax.set_ylabel(
    "First correct STOP detection (m)"
)

ax.set_title(
    "(c) HSV task behavior"
)

ax.grid(
    alpha=0.25
)


fig.suptitle(
    (
        "Environmental Severity, Generic Image Monitoring, "
        "and Task Reliability Do Not Tell the Same Story"
    ),
    fontsize=14,
    y=1.03,
)

fig.tight_layout()

save_figure(
    fig,
    "fig3_monitor_task_mismatch",
)

plt.close(fig)


# ============================================================
# FIGURE 4
#
# CAMERA PIPELINE SENSITIVITY
#
# IMPORTANT:
# The current manual exposure setting is NOT luminance-matched.
# Therefore this is a DIAGNOSTIC figure, not causal proof.
# ============================================================

require(
    MANUAL_HSV_SUMMARY
)

manual_hsv = pd.read_csv(
    MANUAL_HSV_SUMMARY
)


auto_hsv = stop["HSV"]


auto_clear = auto_hsv[
    auto_hsv["severity"] == 0
].iloc[0]

manual_clear = manual_hsv[
    manual_hsv["severity"] == 0
].iloc[0]


auto_30 = auto_hsv[
    auto_hsv["severity"] == 30
].iloc[0]

auto_45 = auto_hsv[
    auto_hsv["severity"] == 45
].iloc[0]

manual_30 = manual_hsv[
    manual_hsv["severity"] == 30
].iloc[0]

manual_45 = manual_hsv[
    manual_hsv["severity"] == 45
].iloc[0]


auto_gap = (
    auto_45[
        "first_correct_distance_mean_m"
    ]
    -
    auto_30[
        "first_correct_distance_mean_m"
    ]
)

manual_gap = (
    manual_45[
        "first_correct_distance_mean_m"
    ]
    -
    manual_30[
        "first_correct_distance_mean_m"
    ]
)


fig, axes = plt.subplots(
    1,
    3,
    figsize=(13.7, 4.4),
)


labels = [
    "Histogram",
    "Tested manual",
]


# ------------------------------------------------------------
# CLEAR CORRECT RATE
# ------------------------------------------------------------

ax = axes[0]

vals = [
    auto_clear[
        "correct_detection_rate_mean_pct"
    ],
    manual_clear[
        "correct_detection_rate_mean_pct"
    ],
]

bars = ax.bar(
    labels,
    vals,
)

ax.set_ylabel(
    "Correct detections (%)"
)

ax.set_title(
    "(a) Clear-weather HSV"
)

ax.set_ylim(
    0,
    max(vals) * 1.25
)

for bar, value in zip(
    bars,
    vals,
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}%",
        ha="center",
        fontsize=10,
    )


# ------------------------------------------------------------
# FALSE POSITIVES
# ------------------------------------------------------------

ax = axes[1]

vals_fp = [
    auto_clear[
        "false_positive_mean"
    ],
    manual_clear[
        "false_positive_mean"
    ],
]

bars = ax.bar(
    labels,
    vals_fp,
)

ax.set_ylabel(
    "False positives / trial"
)

ax.set_title(
    "(b) Clear false positives"
)

ax.set_ylim(
    0,
    max(vals_fp) * 1.25
)

for bar, value in zip(
    bars,
    vals_fp,
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 1,
        f"{value:.1f}",
        ha="center",
        fontsize=10,
    )


# ------------------------------------------------------------
# FOG30 -> FOG45 GAP
# ------------------------------------------------------------

ax = axes[2]

gap_vals = [
    auto_gap,
    manual_gap,
]

bars = ax.bar(
    labels,
    gap_vals,
)

ax.axhline(
    0,
    linewidth=1,
)

ax.set_ylabel(
    r"$D_{45}-D_{30}$ (m)"
)

ax.set_title(
    "(c) Fog30→Fog45 rebound"
)

for bar, value in zip(
    bars,
    gap_vals,
):
    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.08,
        f"{value:.2f} m",
        ha="center",
        fontsize=10,
    )


survival_pct = (
    100.0
    * manual_gap
    / auto_gap
)

reduction_pct = (
    100.0
    - survival_pct
)


fig.suptitle(
    (
        "HSV Is Highly Sensitive to the Camera Imaging Pipeline\n"
        "Diagnostic manual exposure is not luminance-matched"
    ),
    fontsize=13,
    y=1.06,
)

fig.text(
    0.5,
    -0.04,
    (
        f"Observed Fog30→Fog45 gap reduction = "
        f"{reduction_pct:.1f}% "
        f"(residual {survival_pct:.1f}%). "
        "This is descriptive, not causal."
    ),
    ha="center",
    fontsize=9,
)

fig.tight_layout()

save_figure(
    fig,
    "fig4_camera_pipeline_sensitivity",
)

plt.close(fig)


# ============================================================
# FIGURE 5
#
# LANE RANGE DEPENDENCE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.6),
)


# ------------------------------------------------------------
# AVAILABILITY
# ------------------------------------------------------------

ax = axes[0]

ax.errorbar(
    lane_near[
        "severity"
    ],
    lane_near[
        "availability_mean_pct"
    ],
    yerr=lane_near[
        "availability_std_pct"
    ],
    marker="o",
    linewidth=2,
    capsize=3,
    label="Lane @ 8.9 m",
)

ax.errorbar(
    lane_25[
        "severity"
    ],
    lane_25[
        "availability_mean_pct"
    ],
    yerr=lane_25[
        "availability_std_pct"
    ],
    marker="s",
    linewidth=2,
    capsize=3,
    label="Lane @ 25.1 m",
)

ax.set_xticks(
    SEVERITIES
)

ax.set_ylim(
    -3,
    105,
)

ax.set_xlabel(
    "Fog density"
)

ax.set_ylabel(
    "Lane output availability (%)"
)

ax.set_title(
    "(a) Availability depends on range"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)


# ------------------------------------------------------------
# RELIABILITY
# ------------------------------------------------------------

ax = axes[1]

ax.errorbar(
    lane_near[
        "severity"
    ],
    lane_near[
        "R_lane_mean"
    ],
    yerr=lane_near[
        "R_lane_std"
    ],
    marker="o",
    linewidth=2,
    capsize=3,
    label="Lane @ 8.9 m",
)

ax.errorbar(
    lane_25[
        "severity"
    ],
    lane_25[
        "R_lane_mean"
    ],
    yerr=lane_25[
        "R_lane_std"
    ],
    marker="s",
    linewidth=2,
    capsize=3,
    label="Lane @ 25.1 m",
)

ax.axhline(
    0.5,
    linestyle="--",
    linewidth=1.5,
    label="Reliability threshold",
)

ax.set_xticks(
    SEVERITIES
)

ax.set_ylim(
    -0.04,
    1.08,
)

ax.set_xlabel(
    "Fog density"
)

ax.set_ylabel(
    r"$R_{\mathrm{lane}}$"
)

ax.set_title(
    "(b) Reliability depends on range"
)

ax.grid(
    alpha=0.25
)

ax.legend(
    frameon=False
)


fig.suptitle(
    "A Single Lane-Robustness Number Can Hide Strong Range Dependence",
    fontsize=13,
)

fig.tight_layout()

save_figure(
    fig,
    "fig5_lane_range_dependence",
)

plt.close(fig)


# ============================================================
# BONUS FIGURE
#
# PAIRED HSV FOG30 -> FOG45
#
# Shows that the direction occurs across individual trials,
# rather than being created only by averaging.
# ============================================================

require(
    STOP_TRIAL_FILES["HSV"]
)

require(
    MANUAL_HSV_TRIALS
)


auto_trials = pd.read_csv(
    STOP_TRIAL_FILES["HSV"]
)

manual_trials = pd.read_csv(
    MANUAL_HSV_TRIALS
)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.8, 4.6),
    sharey=True,
)


def paired_plot(
    ax,
    df,
    title,
):

    fog30 = (
        df[
            df["severity"] == 30
        ]
        .sort_values("trial")
    )

    fog45 = (
        df[
            df["severity"] == 45
        ]
        .sort_values("trial")
    )

    merged = fog30[
        [
            "trial",
            "first_correct_detection_distance_m",
        ]
    ].merge(
        fog45[
            [
                "trial",
                "first_correct_detection_distance_m",
            ]
        ],
        on="trial",
        suffixes=(
            "_30",
            "_45",
        ),
    )

    for _, row in merged.iterrows():

        ax.plot(
            [
                30,
                45,
            ],
            [
                row[
                    "first_correct_detection_distance_m_30"
                ],
                row[
                    "first_correct_detection_distance_m_45"
                ],
            ],
            marker="o",
            linewidth=1.5,
            alpha=0.75,
        )

    mean30 = merged[
        "first_correct_detection_distance_m_30"
    ].mean()

    mean45 = merged[
        "first_correct_detection_distance_m_45"
    ].mean()

    ax.plot(
        [
            30,
            45,
        ],
        [
            mean30,
            mean45,
        ],
        marker="s",
        linewidth=4,
        color="black",
        label="Mean",
    )

    delta = (
        mean45
        - mean30
    )

    ax.set_xticks(
        [
            30,
            45,
        ]
    )

    ax.set_xticklabels(
        [
            "Fog30",
            "Fog45",
        ]
    )

    ax.set_title(
        title
        + f"\nMean Δ = {delta:+.2f} m"
    )

    ax.grid(
        alpha=0.25
    )

    ax.legend(
        frameon=False
    )


paired_plot(
    axes[0],
    auto_trials,
    "Histogram exposure",
)

paired_plot(
    axes[1],
    manual_trials,
    "Tested manual exposure",
)

axes[0].set_ylabel(
    "First correct detection distance (m)"
)

fig.suptitle(
    (
        "HSV Fog30→Fog45 Rebound Across Individual Trials\n"
        "Manual configuration is a diagnostic control, not luminance-matched"
    ),
    fontsize=13,
)

fig.tight_layout()

save_figure(
    fig,
    "fig6_hsv_fog30_fog45_paired",
)

plt.close(fig)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 90)
print("PAPER FIGURES COMPLETE")
print("=" * 90)

print()
print("Main-paper candidates:")
print("  Fig 1: reliability heatmap")
print("  Fig 2: STOP reliability curves")
print("  Fig 3: image-monitor-task mismatch")
print("  Fig 5: lane range dependence")

print()
print("Diagnostic / supplementary:")
print("  Fig 4: camera-pipeline sensitivity")
print("  Fig 6: paired Fog30/Fog45 exposure diagnostic")

print()
print(f"Output directory: {OUT}")
