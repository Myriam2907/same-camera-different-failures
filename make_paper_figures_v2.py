#!/usr/bin/env python3
"""
Rebuild the paper figures using the FINAL result CSVs.

Place this file at:
    analysis/make_paper_figures_v2.py

Run from the repository root:
    PYTHONPATH="$PWD" python -u analysis/make_paper_figures_v2.py

Outputs:
    results/final/paper_figures_v2/
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "final" / "paper_figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

SEVERITIES = [0, 15, 30, 45, 60, 75, 90]
TRIALS = [1, 2, 3, 4, 5]

STOP_DIRS = {
    "HSV": ROOT / "results" / "final" / "stop_hsv",
    "Template": ROOT / "results" / "final" / "stop_template",
    "YOLO26s": ROOT / "results" / "final" / "stop_yolo",
    "RF-DETR-L": ROOT / "results" / "final" / "stop_rfdetr",
}

SUMMARY_FILES = {
    name: path / "final_summary.csv"
    for name, path in STOP_DIRS.items()
}

LANE_SUMMARY = ROOT / "results" / "final" / "lane_ufld_range" / "final_summary.csv"
MAG_SUMMARY = ROOT / "results" / "final" / "local_magnitude_audit" / "summary.csv"
FOURIER_POS = ROOT / "results" / "final" / "fourier_monitor" / "position_distances.csv"


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def rec_name(sev, trial):
    if sev == 0:
        return f"stop3_clear_0_drift_0p75_trial_{trial}"
    return f"stop3_fog_{sev}_drift_0p75_trial_{trial}"


def result_prefix(sev, trial):
    if sev == 0:
        return f"clear_trial_{trial}"
    return f"fog{sev}_trial_{trial}"


def save(fig, stem):
    for ext in ("png", "pdf"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, dpi=400 if ext == "png" else None, bbox_inches="tight")
        print("Saved:", path)


plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})


# ============================================================
# 1. STOP reliability: all four detectors
# ============================================================

stop_summary = {}
for name, path in SUMMARY_FILES.items():
    require(path)
    stop_summary[name] = pd.read_csv(path).sort_values("severity")

fig, ax = plt.subplots(figsize=(8.4, 5.2))
markers = ["o", "s", "^", "D"]

for marker, (name, df) in zip(markers, stop_summary.items()):
    ax.errorbar(
        df["severity"],
        df["R_sign_mean"],
        yerr=df["R_sign_std"],
        marker=marker,
        linewidth=2,
        capsize=3,
        label=name,
    )

ax.axhline(0.5, linestyle="--", linewidth=1.4, label="Reliability threshold = 0.50")
ax.set_xticks(SEVERITIES)
ax.set_ylim(-0.03, 1.08)
ax.set_xlabel("Fog density")
ax.set_ylabel(r"$R_{\mathrm{sign}}$")
ax.set_title("STOP-Sign Reliability: Same Camera, Different Detectors")
ax.grid(alpha=0.25)
ax.legend(ncol=2, frameon=False)
fig.tight_layout()
save(fig, "fig_stop_reliability_all")
plt.close(fig)


# ============================================================
# 2. Deep-detector trade-off: retained range vs false positives
# ============================================================

yolo = stop_summary["YOLO26s"]
rf = stop_summary["RF-DETR-L"]

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

ax = axes[0]
for name, df, marker in [
    ("YOLO26s", yolo, "o"),
    ("RF-DETR-L", rf, "s"),
]:
    ax.errorbar(
        df["severity"],
        df["R_sign_mean"],
        yerr=df["R_sign_std"],
        marker=marker,
        linewidth=2,
        capsize=3,
        label=name,
    )
ax.axhline(0.5, linestyle="--", linewidth=1.2)
ax.set_xticks(SEVERITIES)
ax.set_ylim(0, 1.08)
ax.set_xlabel("Fog density")
ax.set_ylabel(r"$R_{\mathrm{sign}}$")
ax.set_title("(a) Retained first-detection range")
ax.grid(alpha=0.25)
ax.legend(frameon=False)

ax = axes[1]
for name, df, marker in [
    ("YOLO26s", yolo, "o"),
    ("RF-DETR-L", rf, "s"),
]:
    ax.plot(
        df["severity"],
        df["false_positive_boxes_mean"],
        marker=marker,
        linewidth=2,
        label=name,
    )
ax.set_yscale("log")
ax.set_xticks(SEVERITIES)
ax.set_xlabel("Fog density")
ax.set_ylabel("False-positive boxes / trial (log scale)")
ax.set_title("(b) False-positive cost")
ax.grid(alpha=0.25, which="both")
ax.legend(frameon=False)

fig.suptitle(
    "A Detector Can Keep Long Range While Producing Many False Alarms",
    fontsize=13,
)
fig.tight_layout()
save(fig, "fig_deep_detector_tradeoff")
plt.close(fig)


# ============================================================
# 3. Lane figure: NO penalty-derived 25 m reliability.
#    Show what is directly measured:
#    availability + conditional error.
# ============================================================

require(LANE_SUMMARY)
lane = pd.read_csv(LANE_SUMMARY)

near = lane[lane["row_label"] == "primary_504"].sort_values("severity")
far = lane[lane["row_label"] == "target_25m"].sort_values("severity")

if near.empty or far.empty:
    raise RuntimeError("Could not find primary_504 and target_25m rows.")

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

ax = axes[0]
for name, df, marker in [
    ("8.9 m", near, "o"),
    ("25.1 m", far, "s"),
]:
    ax.errorbar(
        df["severity"],
        df["availability_mean_pct"],
        yerr=df["availability_std_pct"],
        marker=marker,
        linewidth=2,
        capsize=3,
        label=name,
    )
ax.set_xticks(SEVERITIES)
ax.set_ylim(-2, 104)
ax.set_xlabel("Fog density")
ax.set_ylabel("Lane output availability (%)")
ax.set_title("(a) Does the detector return a lane?")
ax.grid(alpha=0.25)
ax.legend(frameon=False)

ax = axes[1]
for name, df, marker in [
    ("8.9 m", near, "o"),
    ("25.1 m", far, "s"),
]:
    ax.errorbar(
        df["severity"],
        100.0 * df["conditional_mae_m_mean"],
        yerr=100.0 * df["conditional_mae_m_std"],
        marker=marker,
        linewidth=2,
        capsize=3,
        label=name,
    )
ax.set_xticks(SEVERITIES)
ax.set_xlabel("Fog density")
ax.set_ylabel("Center MAE when output exists (cm)")
ax.set_title("(b) How accurate is the returned lane?")
ax.grid(alpha=0.25)
ax.legend(frameon=False)

fig.suptitle(
    "Lane Range Changes Availability More Than Conditional Accuracy",
    fontsize=13,
)
fig.tight_layout()
save(fig, "fig_lane_availability_accuracy")
plt.close(fig)


# ============================================================
# Helper: Fourier spectral-distance column
# ============================================================

require(FOURIER_POS)
fourier = pd.read_csv(FOURIER_POS)

spectral_candidates = [
    c for c in fourier.columns
    if "spectral" in c.lower() and "distance" in c.lower()
]
if len(spectral_candidates) != 1:
    spectral_candidates = [
        c for c in fourier.columns
        if "distance" in c.lower()
        and c not in {"distance_to_sign_m"}
    ]
if len(spectral_candidates) != 1:
    raise RuntimeError(
        "Could not uniquely identify the Fourier distance column. "
        f"Candidates: {spectral_candidates}"
    )
SPECTRAL = spectral_candidates[0]


def summarize_fourier(df):
    per_trial = (
        df.groupby(["severity", "trial"], as_index=False)[SPECTRAL]
        .mean()
    )
    return (
        per_trial.groupby("severity", as_index=False)
        .agg(
            mean=(SPECTRAL, "mean"),
            std=(SPECTRAL, lambda x: np.std(x, ddof=1)),
        )
    )


global_fourier = summarize_fourier(fourier)
local_ids = list(range(104, 116))
local_fourier = summarize_fourier(
    fourier[fourier["sample_id"].isin(local_ids)].copy()
)


# ============================================================
# 4. Global vs local Fourier
# ============================================================

fig, ax = plt.subplots(figsize=(8.2, 5.0))
for name, df, marker in [
    ("Whole route", global_fourier, "o"),
    ("20-26 m window", local_fourier, "s"),
]:
    ax.errorbar(
        df["severity"],
        1e5 * df["mean"],
        yerr=1e5 * df["std"],
        marker=marker,
        linewidth=2,
        capsize=3,
        label=name,
    )

ax.set_xticks(SEVERITIES)
ax.set_xlabel("Fog density")
ax.set_ylabel(r"Normalized spectral-shape distance $\times 10^5$")
ax.set_title("Route Averaging Can Hide a Local Spectral Effect")
ax.grid(alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
save(fig, "fig_fourier_global_vs_local")
plt.close(fig)


# ============================================================
# 5. Matched 20-26 m monitor/task figure.
#    All three panels now use the SAME route window.
# ============================================================

require(MAG_SUMMARY)
mag = pd.read_csv(MAG_SUMMARY).sort_values("severity")

# Compute HSV correctness on exactly samples 104-115, per trial.
local_hsv_rows = []

for sev in SEVERITIES:
    trial_values = []

    for trial in TRIALS:
        prefix = result_prefix(sev, trial)
        csv_path = STOP_DIRS["HSV"] / f"{prefix}_per_frame.csv"
        require(csv_path)

        df = pd.read_csv(csv_path)
        q = df[df["sample_id"].isin(local_ids)].copy()

        if len(q) != len(local_ids):
            raise RuntimeError(
                f"{csv_path}: expected {len(local_ids)} local frames, "
                f"found {len(q)}."
            )

        trial_values.append(100.0 * q["correct_detection"].astype(float).mean())

    local_hsv_rows.append({
        "severity": sev,
        "mean": float(np.mean(trial_values)),
        "std": float(np.std(trial_values, ddof=1)),
    })

local_hsv = pd.DataFrame(local_hsv_rows)

fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

ax = axes[0]
ax.errorbar(
    mag["severity"],
    mag["rms_contrast_ratio_mean"],
    yerr=mag["rms_contrast_ratio_std"],
    marker="o",
    linewidth=2,
    capsize=3,
    label="RMS contrast",
)
ax.errorbar(
    mag["severity"],
    mag["gradient_ratio_mean"],
    yerr=mag["gradient_ratio_std"],
    marker="s",
    linewidth=2,
    capsize=3,
    label="Gradient detail",
)
ax.set_xticks(SEVERITIES)
ax.set_ylim(0, 1.06)
ax.set_xlabel("Fog density")
ax.set_ylabel("Normalized value relative to clear")
ax.set_title("(a) Direct image measures")
ax.grid(alpha=0.25)
ax.legend(frameon=False)

ax = axes[1]
ax.errorbar(
    local_fourier["severity"],
    1e5 * local_fourier["mean"],
    yerr=1e5 * local_fourier["std"],
    marker="o",
    linewidth=2,
    capsize=3,
)
ax.set_xticks(SEVERITIES)
ax.set_xlabel("Fog density")
ax.set_ylabel(r"Spectral-shape distance $\times 10^5$")
ax.set_title("(b) Fourier monitor")
ax.grid(alpha=0.25)

ax = axes[2]
ax.errorbar(
    local_hsv["severity"],
    local_hsv["mean"],
    yerr=local_hsv["std"],
    marker="o",
    linewidth=2,
    capsize=3,
)
ax.set_xticks(SEVERITIES)
ax.set_ylim(-3, 103)
ax.set_xlabel("Fog density")
ax.set_ylabel("HSV correct detections (%)")
ax.set_title("(c) Task result on the same frames")
ax.grid(alpha=0.25)

fig.suptitle(
    "Same 20-26 m Frames: Image Change, Spectral Change, and Task Success",
    fontsize=13,
)
fig.tight_layout()
save(fig, "fig_monitor_task_same_window")
plt.close(fig)


# ============================================================
# 6. Raw CARLA fog progression
# ============================================================

raw_specs = [
    ("Clear", 0),
    ("Fog30", 30),
    ("Fog45", 45),
    ("Fog90", 90),
]

fig, axes = plt.subplots(1, 4, figsize=(14.5, 3.1))
for ax, (label, sev) in zip(axes, raw_specs):
    frame = (
        ROOT / "data" / "recordings" / rec_name(sev, 3)
        / "frames" / "0105.png"
    )
    require(frame)
    img = cv2.cvtColor(cv2.imread(str(frame)), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(label)
    ax.axis("off")

fig.suptitle("Same CARLA Camera Pose Under Increasing Fog", fontsize=13)
fig.tight_layout()
save(fig, "fig_carla_fog_progression")
plt.close(fig)


# ============================================================
# 7. Detector-output qualitative panel
#    Trial 1, sample 101.
# ============================================================

def draw_gt(img, row, x1="gt_bbox_x1", y1="gt_bbox_y1",
            x2="gt_bbox_x2", y2="gt_bbox_y2"):
    if all(c in row.index for c in [x1, y1, x2, y2]):
        vals = [row[x1], row[y1], row[x2], row[y2]]
        if all(pd.notna(v) for v in vals):
            p1 = (int(round(vals[0])), int(round(vals[1])))
            p2 = (int(round(vals[2])), int(round(vals[3])))
            cv2.rectangle(img, p1, p2, (255, 0, 255), 2)


def draw_hsv(img, sev, trial, sid):
    prefix = result_prefix(sev, trial)
    df = pd.read_csv(STOP_DIRS["HSV"] / f"{prefix}_per_frame.csv")
    row = df[df["sample_id"] == sid].iloc[0]
    draw_gt(img, row)

    if int(row.get("raw_detected", 0)) == 1:
        vals = [row.get("bbox_x"), row.get("bbox_y"),
                row.get("bbox_w"), row.get("bbox_h")]
        if all(pd.notna(v) for v in vals):
            x, y, w, h = [int(round(v)) for v in vals]
            color = (0, 200, 0) if int(row.get("correct_detection", 0)) == 1 else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img


def boolish(v):
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return bool(v)


def draw_deep(img, detector_key, sev, trial, sid):
    prefix = result_prefix(sev, trial)
    ddir = STOP_DIRS[detector_key]

    frames = pd.read_csv(ddir / f"{prefix}_frames.csv")
    row = frames[frames["sample_id"] == sid].iloc[0]
    draw_gt(img, row, "gt_x1", "gt_y1", "gt_x2", "gt_y2")

    boxes = pd.read_csv(ddir / f"{prefix}_raw_boxes.csv")
    q = boxes[boxes["sample_id"] == sid].copy()

    if "passes_operating_threshold" in q.columns:
        q = q[q["passes_operating_threshold"].map(boolish)]

    for _, b in q.iterrows():
        vals = [b["x1"], b["y1"], b["x2"], b["y2"]]
        if not all(pd.notna(v) for v in vals):
            continue
        p1 = (int(round(b["x1"])), int(round(b["y1"])))
        p2 = (int(round(b["x2"])), int(round(b["y2"])))
        correct = boolish(b.get("correct_by_iou", False))
        color = (0, 200, 0) if correct else (0, 0, 255)
        cv2.rectangle(img, p1, p2, color, 2)

    return img


conditions = [("Clear", 0), ("Fog30", 30), ("Fog45", 45), ("Fog90", 90)]
rows = ["Raw RGB", "HSV", "YOLO26s", "RF-DETR-L"]

fig, axes = plt.subplots(4, 4, figsize=(14.5, 10.0))

for col, (label, sev) in enumerate(conditions):
    frame = (
        ROOT / "data" / "recordings" / rec_name(sev, 1)
        / "frames" / "0101.png"
    )
    require(frame)
    base = cv2.imread(str(frame))

    versions = [
        base.copy(),
        draw_hsv(base.copy(), sev, 1, 101),
        draw_deep(base.copy(), "YOLO26s", sev, 1, 101),
        draw_deep(base.copy(), "RF-DETR-L", sev, 1, 101),
    ]

    for row_idx, img in enumerate(versions):
        axes[row_idx, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[row_idx, col].axis("off")

        if row_idx == 0:
            axes[row_idx, col].set_title(label)

        if col == 0:
            axes[row_idx, col].text(
                -0.04, 0.5, rows[row_idx],
                transform=axes[row_idx, col].transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontsize=11,
            )

fig.suptitle(
    "Same Frame Position, Different STOP-Detector Behavior\n"
    "Magenta = CARLA ground truth; green = correct; red = accepted false positive",
    fontsize=13,
)
fig.tight_layout()
save(fig, "fig_stop_detector_examples")
plt.close(fig)


# ============================================================
# 8. Generate detailed STOP table for LaTeX
# ============================================================

# Key severities keep the table readable; the curve contains all seven.
KEY = [0, 30, 45, 90]

lines = []
lines.append(r"\begin{table*}[t]")
lines.append(r"\centering")
lines.append(r"\small")
lines.append(
    r"\caption{Detailed STOP-sign results at representative fog levels. "
    r"$D$ is mean first correct detection distance. "
    r"Correct is the mean percentage of evaluable approach frames with IoU$\geq0.5$. "
    r"For YOLO26s and RF-DETR-L, FP is the mean number of false-positive boxes; "
    r"for HSV and Template, FP is their detector-specific false-positive count, "
    r"so FP values should be compared directly only within the same detector family.}"
)
lines.append(r"\label{tab:stop-detailed}")
lines.append(r"\resizebox{\textwidth}{!}{%")
lines.append(r"\begin{tabular}{@{}llrrrr@{}}")
lines.append(r"\toprule")
lines.append(r"\textbf{Detector} & \textbf{Fog} & \textbf{$D$ (m)} & \textbf{$R_{\rm sign}$} & \textbf{Correct (\%)} & \textbf{FP} \\")
lines.append(r"\midrule")

for name, df in stop_summary.items():
    for sev in KEY:
        row = df[df["severity"] == sev].iloc[0]

        if name in {"YOLO26s", "RF-DETR-L"}:
            fp = row["false_positive_boxes_mean"]
        else:
            fp = row["false_positive_mean"]

        lines.append(
            f"{name} & {sev} & "
            f"{row['first_correct_distance_mean_m']:.2f} $\\pm$ {row['first_correct_distance_std_m']:.2f} & "
            f"{row['R_sign_mean']:.2f} $\\pm$ {row['R_sign_std']:.2f} & "
            f"{row['correct_detection_rate_mean_pct']:.1f} & "
            f"{fp:.1f} \\\\"
        )
    lines.append(r"\addlinespace")

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}}")
lines.append(r"\end{table*}")

table_path = OUT / "stop_detailed_table.tex"
table_path.write_text("\n".join(lines) + "\n")
print("Saved:", table_path)

print("\nDone.")
print("Output directory:", OUT)
