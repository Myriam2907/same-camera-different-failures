#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "final"
    / "lane_ufld_range"
    / "final_per_trial.csv"
)

OUT = (
    ROOT
    / "results"
    / "final"
    / "lane_ufld_range"
)

EPSILON_M = 0.013552

FORMAL_ROWS = [
    "primary_504",
    "target_25m",
]

EXPLORATORY_ROWS = [
    "target_30m",
    "target_35m",
    "target_40m",
]


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(INPUT)

required = {
    "severity",
    "trial",
    "row_label",
    "row_y",
    "preregister_status",
    "n_eligible",
    "availability_pct",
    "actual_median_range_m",
    "conditional_center_mae_px",
    "conditional_center_mae_m",
}

missing = required - set(df.columns)

if missing:
    raise RuntimeError(
        f"Missing columns: {sorted(missing)}"
    )


# ============================================================
# PAIRED CLEAR BASELINES
# ============================================================

clear = (
    df[
        df["severity"] == 0
    ][
        [
            "trial",
            "row_label",
            "availability_pct",
            "conditional_center_mae_px",
            "conditional_center_mae_m",
        ]
    ]
    .rename(
        columns={
            "availability_pct":
                "clear_availability_pct",

            "conditional_center_mae_px":
                "clear_conditional_mae_px",

            "conditional_center_mae_m":
                "clear_conditional_mae_m",
        }
    )
)

paired = df.merge(
    clear,
    on=[
        "trial",
        "row_label",
    ],
    how="left",
)

if paired[
    "clear_availability_pct"
].isna().any():

    raise RuntimeError(
        "Missing paired Clear baseline."
    )


# ============================================================
# POST-HOC SPLIT METRICS
#
# IMPORTANT:
# These were defined AFTER inspecting the range-confound
# behavior. They are diagnostic/post-hoc, NOT preregistered.
#
# No min(1, ...) clipping is used.
# ============================================================

paired[
    "R_avail_unclipped"
] = (
    paired[
        "availability_pct"
    ]
    /
    paired[
        "clear_availability_pct"
    ]
)


paired[
    "R_acc_unclipped"
] = (
    paired[
        "clear_conditional_mae_m"
    ]
    + EPSILON_M
) / (
    paired[
        "conditional_center_mae_m"
    ]
    + EPSILON_M
)


# Diagnostic only:
# quantify exactly how much the old clipping would change
# the post-hoc R_acc value.

paired[
    "R_acc_clipped_diagnostic"
] = np.minimum(
    1.0,
    paired[
        "R_acc_unclipped"
    ],
)

paired[
    "clip_effect"
] = (
    paired[
        "R_acc_unclipped"
    ]
    -
    paired[
        "R_acc_clipped_diagnostic"
    ]
)


# Paired changes in raw conditional localization error.

paired[
    "delta_conditional_mae_px"
] = (
    paired[
        "conditional_center_mae_px"
    ]
    -
    paired[
        "clear_conditional_mae_px"
    ]
)

paired[
    "delta_conditional_mae_m"
] = (
    paired[
        "conditional_center_mae_m"
    ]
    -
    paired[
        "clear_conditional_mae_m"
    ]
)


# ============================================================
# SAVE PER-CONFIGURATION DIAGNOSTICS
# ============================================================

paired_path = (
    OUT
    / "posthoc_split_metrics_per_trial.csv"
)

paired.to_csv(
    paired_path,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for (
    row_label,
    severity,
), g in paired.groupby(
    [
        "row_label",
        "severity",
    ]
):

    cond_m = g[
        "conditional_center_mae_m"
    ].dropna()

    cond_px = g[
        "conditional_center_mae_px"
    ].dropna()

    r_acc = g[
        "R_acc_unclipped"
    ].dropna()

    delta_m = g[
        "delta_conditional_mae_m"
    ].dropna()

    delta_px = g[
        "delta_conditional_mae_px"
    ].dropna()

    summary_rows.append(
        {
            "row_label":
                row_label,

            "severity":
                int(severity),

            "row_y":
                int(
                    g["row_y"].iloc[0]
                ),

            "actual_range_m":
                float(
                    g[
                        "actual_median_range_m"
                    ].mean()
                ),

            "eligible_frames_min":
                int(
                    g[
                        "n_eligible"
                    ].min()
                ),

            "availability_mean_pct":
                float(
                    g[
                        "availability_pct"
                    ].mean()
                ),

            "availability_std_pct":
                float(
                    g[
                        "availability_pct"
                    ].std(ddof=1)
                ),

            "R_avail_mean":
                float(
                    g[
                        "R_avail_unclipped"
                    ].mean()
                ),

            "R_avail_std":
                float(
                    g[
                        "R_avail_unclipped"
                    ].std(ddof=1)
                ),

            "conditional_mae_px_mean":
                float(
                    cond_px.mean()
                )
                if len(cond_px)
                else np.nan,

            "conditional_mae_px_std":
                float(
                    cond_px.std(ddof=1)
                )
                if len(cond_px) > 1
                else np.nan,

            "conditional_mae_m_mean":
                float(
                    cond_m.mean()
                )
                if len(cond_m)
                else np.nan,

            "conditional_mae_m_std":
                float(
                    cond_m.std(ddof=1)
                )
                if len(cond_m) > 1
                else np.nan,

            "conditional_trials_n":
                int(
                    len(cond_m)
                ),

            "R_acc_unclipped_mean":
                float(
                    r_acc.mean()
                )
                if len(r_acc)
                else np.nan,

            "R_acc_unclipped_std":
                float(
                    r_acc.std(ddof=1)
                )
                if len(r_acc) > 1
                else np.nan,

            "R_acc_clipped_mean_DIAGNOSTIC":
                float(
                    g[
                        "R_acc_clipped_diagnostic"
                    ].mean()
                ),

            "mean_clip_effect":
                float(
                    g[
                        "clip_effect"
                    ].mean()
                ),

            "delta_conditional_mae_px_mean":
                float(
                    delta_px.mean()
                )
                if len(delta_px)
                else np.nan,

            "delta_conditional_mae_px_std":
                float(
                    delta_px.std(ddof=1)
                )
                if len(delta_px) > 1
                else np.nan,

            "delta_conditional_mae_m_mean":
                float(
                    delta_m.mean()
                )
                if len(delta_m)
                else np.nan,

            "delta_conditional_mae_m_std":
                float(
                    delta_m.std(ddof=1)
                )
                if len(delta_m) > 1
                else np.nan,
        }
    )


summary = pd.DataFrame(
    summary_rows
)

order = {
    "primary_504": 0,
    "target_25m": 1,
    "target_30m": 2,
    "target_35m": 3,
    "target_40m": 4,
}

summary["_order"] = (
    summary[
        "row_label"
    ].map(order)
)

summary = (
    summary
    .sort_values(
        [
            "_order",
            "severity",
        ]
    )
    .drop(
        columns="_order"
    )
    .reset_index(drop=True)
)

summary_path = (
    OUT
    / "posthoc_split_metrics_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False,
)


# ============================================================
# PRINT 1:
# NEAR-RANGE PAIRED LOCALIZATION
# ============================================================

print()
print("=" * 110)
print(
    "8.9 m CONDITIONAL LOCALIZATION — "
    "PAIRED CONFIGURATION CHECK"
)
print("=" * 110)

near = paired[
    paired[
        "row_label"
    ]
    == "primary_504"
].copy()

for severity in sorted(
    near[
        "severity"
    ].unique()
):

    g = near[
        near[
            "severity"
        ]
        == severity
    ].sort_values(
        "trial"
    )

    print()
    print(
        f"FOG {severity}"
        if severity != 0
        else "CLEAR"
    )

    print(
        g[
            [
                "trial",
                "conditional_center_mae_px",
                "clear_conditional_mae_px",
                "delta_conditional_mae_px",
                "R_acc_unclipped",
                "R_acc_clipped_diagnostic",
                "clip_effect",
            ]
        ].to_string(
            index=False
        )
    )


print()
print("=" * 110)
print(
    "8.9 m CONDITIONAL MAE SUMMARY"
)
print("=" * 110)

near_summary = summary[
    summary[
        "row_label"
    ]
    == "primary_504"
]

print(
    near_summary[
        [
            "severity",
            "conditional_mae_px_mean",
            "conditional_mae_px_std",
            "delta_conditional_mae_px_mean",
            "delta_conditional_mae_px_std",
            "R_acc_unclipped_mean",
            "R_acc_unclipped_std",
            "R_acc_clipped_mean_DIAGNOSTIC",
            "mean_clip_effect",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# PRINT 2:
# FORMAL RANGE-CONTROL ROW
# ============================================================

print()
print("=" * 110)
print(
    "25.1 m POST-HOC SPLIT METRICS"
)
print("=" * 110)

formal25 = summary[
    summary[
        "row_label"
    ]
    == "target_25m"
]

print(
    formal25[
        [
            "severity",
            "availability_mean_pct",
            "availability_std_pct",
            "R_avail_mean",
            "R_avail_std",
            "conditional_mae_m_mean",
            "conditional_mae_m_std",
            "R_acc_unclipped_mean",
            "R_acc_unclipped_std",
            "R_acc_clipped_mean_DIAGNOSTIC",
            "mean_clip_effect",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# PRINT 3:
# EXPLORATORY LONG-RANGE ROWS
#
# Deliberately NO normalized R values here.
# ============================================================

print()
print("=" * 110)
print(
    "30–40 m EXPLORATORY RAW METRICS "
    "(NO R VALUES)"
)
print("=" * 110)

exploratory = summary[
    summary[
        "row_label"
    ].isin(
        EXPLORATORY_ROWS
    )
]

print(
    exploratory[
        [
            "row_label",
            "severity",
            "actual_range_m",
            "eligible_frames_min",
            "availability_mean_pct",
            "availability_std_pct",
            "conditional_mae_m_mean",
            "conditional_mae_m_std",
            "conditional_trials_n",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SAVE A PUBLICATION-FRIENDLY FORMAL TABLE
# ============================================================

formal = summary[
    summary[
        "row_label"
    ].isin(
        FORMAL_ROWS
    )
].copy()

formal_table = formal[
    [
        "row_label",
        "actual_range_m",
        "severity",
        "eligible_frames_min",
        "availability_mean_pct",
        "availability_std_pct",
        "conditional_mae_m_mean",
        "conditional_mae_m_std",
        "R_avail_mean",
        "R_avail_std",
        "R_acc_unclipped_mean",
        "R_acc_unclipped_std",
    ]
]

formal_path = (
    OUT
    / "posthoc_formal_split_table.csv"
)

formal_table.to_csv(
    formal_path,
    index=False,
)


print()
print("=" * 110)
print("SAVED")
print("=" * 110)

print(paired_path)
print(summary_path)
print(formal_path)

print()
print(
    "IMPORTANT:"
)

print(
    "- R_avail and R_acc are POST-HOC diagnostic metrics."
)

print(
    "- Neither metric uses min(1, ...)."
)

print(
    "- Values > 1 are preserved."
)

print(
    "- The original preregistered 8.9 m "
    "failure-aware R_lane remains unchanged."
)

print(
    "- No R values should be reported for "
    "30/35/40 m exploratory rows."
)
