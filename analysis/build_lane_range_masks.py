#!/usr/bin/env python3

from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from common.projection import interpolate_x_at_row


ROOT = Path(__file__).resolve().parents[1]

PREREG = (
    ROOT
    / "results"
    / "final"
    / "lane"
    / "range_extension_preregister.csv"
)

OUT = (
    ROOT
    / "results"
    / "final"
    / "lane"
    / "range_geometry"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

# ------------------------------------------------------------
# Load the already-validated geometry implementation.
# This reruns geometry only. It does NOT inspect UFLD results.
# ------------------------------------------------------------

ns = runpy.run_path(
    str(
        ROOT
        / "analysis"
        / "select_lane_range_rows.py"
    )
)

project_future_route = ns[
    "project_future_route"
]

crossing_at_row = ns[
    "crossing_at_row"
]

clear_trials = ns[
    "clear_trials"
]


# ------------------------------------------------------------
# Read the FROZEN preregistration rows.
# Do not derive or change them here.
# ------------------------------------------------------------

pre = pd.read_csv(PREREG)

required = {
    "label",
    "frozen_row_y",
    "status",
    "actual_median_camera_range_m",
    "median_m_per_px",
}

missing = required - set(pre.columns)

if missing:
    raise RuntimeError(
        f"Preregistration missing columns: {sorted(missing)}"
    )


print()
print("=" * 92)
print("BUILDING FROZEN LANE-RANGE GEOMETRY MASKS")
print("=" * 92)

print(
    pre[
        [
            "label",
            "frozen_row_y",
            "status",
            "actual_median_camera_range_m",
        ]
    ].to_string(index=False)
)


all_counts = []


for trial in range(1, 6):

    df = clear_trials[trial]

    rows = []

    for _, frame in df.iterrows():

        sid = int(
            frame["sample_id"]
        )

        route_s = float(
            frame["route_s_m"]
        )

        ego_junction = int(
            frame["is_junction"]
        )

        projected = None
        camera_xyz = None

        if ego_junction == 0:

            (
                projected,
                camera_xyz,
                _,
            ) = project_future_route(
                frame
            )

        for _, spec in pre.iterrows():

            label = str(
                spec["label"]
            )

            row_y = int(
                spec["frozen_row_y"]
            )

            status = str(
                spec["status"]
            )

            nominal_target = (
                float(spec["nominal_target_m"])
                if (
                    "nominal_target_m"
                    in pre.columns
                    and not pd.isna(
                        spec["nominal_target_m"]
                    )
                )
                else np.nan
            )

            eligible = 0
            gt_x = np.nan
            camera_range = np.nan
            depth = np.nan
            m_per_px = np.nan

            if (
                ego_junction == 0
                and projected is not None
            ):

                crossing = crossing_at_row(
                    projected,
                    camera_xyz,
                    float(row_y),
                )

                if (
                    crossing is not None
                    and crossing["eligible"]
                ):

                    points = [
                        (
                            float(p["u"]),
                            float(p["v"]),
                        )
                        for p in projected
                    ]

                    gt = interpolate_x_at_row(
                        points,
                        float(row_y),
                    )

                    if (
                        gt is not None
                        and np.isfinite(gt)
                        and 0.0 <= gt < 1280.0
                    ):

                        eligible = 1
                        gt_x = float(gt)

                        camera_range = float(
                            crossing[
                                "camera_range_m"
                            ]
                        )

                        depth = float(
                            crossing[
                                "depth_m"
                            ]
                        )

                        m_per_px = float(
                            crossing[
                                "m_per_px"
                            ]
                        )

            rows.append(
                {
                    "trial":
                        trial,

                    "sample_id":
                        sid,

                    "route_s_m":
                        route_s,

                    "row_label":
                        label,

                    "row_y":
                        row_y,

                    "nominal_target_m":
                        nominal_target,

                    "preregister_status":
                        status,

                    "eligible":
                        eligible,

                    "lane_gt_x":
                        gt_x,

                    "camera_range_m":
                        camera_range,

                    "depth_m":
                        depth,

                    "m_per_px":
                        m_per_px,
                }
            )

    out = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # y=504 hard validation against original saved GT
    # --------------------------------------------------------

    primary = out[
        (
            out["row_label"]
            == "primary_504"
        )
        &
        (
            out["eligible"]
            == 1
        )
    ].copy()

    original = (
        df[
            [
                "sample_id",
                "lane_gt_x",
            ]
        ]
        .rename(
            columns={
                "lane_gt_x":
                    "saved_lane_gt_x"
            }
        )
    )

    check = primary.merge(
        original,
        on="sample_id",
        how="left",
    )

    check = check[
        check[
            "saved_lane_gt_x"
        ].notna()
    ]

    diff = np.abs(
        check["lane_gt_x"]
        - check["saved_lane_gt_x"]
    )

    if (
        len(diff) == 0
        or float(diff.max()) > 0.05
    ):
        raise RuntimeError(
            f"Trial {trial}: y=504 GT "
            "reprojection validation failed."
        )

    path = (
        OUT
        / f"trial_{trial}_range_geometry.csv"
    )

    out.to_csv(
        path,
        index=False,
    )

    counts = (
        out[
            out["eligible"] == 1
        ]
        .groupby(
            [
                "row_label",
                "row_y",
                "preregister_status",
            ]
        )
        .size()
        .reset_index(
            name="eligible_frames"
        )
    )

    counts[
        "trial"
    ] = trial

    all_counts.append(
        counts
    )

    print()
    print(
        f"TRIAL {trial}"
    )

    print(
        counts.to_string(
            index=False
        )
    )

    print(
        "y=504 GT max reprojection error:",
        f"{float(diff.max()):.6f} px",
    )

    print(
        "Saved:",
        path,
    )


counts = pd.concat(
    all_counts,
    ignore_index=True,
)

counts_path = (
    OUT
    / "eligibility_counts.csv"
)

counts.to_csv(
    counts_path,
    index=False,
)


print()
print("=" * 92)
print("FINAL ELIGIBILITY CHECK")
print("=" * 92)

summary = (
    counts
    .groupby(
        [
            "row_label",
            "row_y",
            "preregister_status",
        ]
    )["eligible_frames"]
    .agg(
        [
            "min",
            "max",
            "mean",
        ]
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False
    )
)

print()
print("Expected frozen counts:")
print("  primary_504 : 88")
print("  target_25m  : 55")
print("  target_30m  : 47")
print("  target_35m  : 47")
print("  target_40m  : 46-47")

print()
print(
    "PASS only if these counts match the preregistration."
)
