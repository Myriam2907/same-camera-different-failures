#!/usr/bin/env python3

import sys
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

UFLD_ROOT = (
    Path.home()
    / "Ultra-Fast-Lane-Detection-v2"
)

sys.path.insert(
    0,
    str(UFLD_ROOT),
)

from utils.config import Config
from utils.common import get_model


CONFIG = (
    UFLD_ROOT
    / "configs"
    / "culane_res18.py"
)

WEIGHTS = (
    UFLD_ROOT
    / "weights"
    / "culane_res18.pth"
)

GEOM_ROOT = (
    ROOT
    / "results"
    / "final"
    / "lane"
    / "range_geometry"
)

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
    / "lane_ufld_range"
)


# ============================================================
# FROZEN SETTINGS
# ============================================================

IMAGE_W = 1280
IMAGE_H = 720

FAILURE_ERROR_PX = 640.0

# Frozen BEFORE range outcomes.
EPSILON_M = 0.013552

SEVERITIES = [
    0,
    15,
    30,
    45,
    60,
    75,
    90,
]

TRIALS = [
    1,
    2,
    3,
    4,
    5,
]


# ============================================================
# DATASET PATHS
# ============================================================

def dataset_root(
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


# ============================================================
# UFLD MODEL
# ============================================================

def load_model():

    cfg = Config.fromfile(
        str(CONFIG)
    )

    # Exactly the same anchors as the
    # already validated primary UFLD experiment.
    cfg.row_anchor = np.linspace(
        0.42,
        1.0,
        cfg.num_row,
    )

    cfg.col_anchor = np.linspace(
        0.0,
        1.0,
        cfg.num_col,
    )

    net = get_model(
        cfg
    )

    checkpoint = torch.load(
        str(WEIGHTS),
        map_location="cpu",
    )

    state_dict = checkpoint[
        "model"
    ]

    compatible = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):
            key = key[7:]

        compatible[
            key
        ] = value

    missing, unexpected = (
        net.load_state_dict(
            compatible,
            strict=False,
        )
    )

    if missing or unexpected:

        print("WARNING")
        print(
            "Missing keys:",
            missing,
        )

        print(
            "Unexpected keys:",
            unexpected,
        )

    net.eval().cuda()

    return (
        net,
        cfg,
    )


def make_transform(
    cfg,
):

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    int(
                        cfg.train_height
                        / cfg.crop_ratio
                    ),
                    cfg.train_width,
                )
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                (
                    0.485,
                    0.456,
                    0.406,
                ),
                (
                    0.229,
                    0.224,
                    0.225,
                ),
            ),
        ]
    )


# ============================================================
# UFLD DECODING
# EXACTLY SAME STYLE AS PRIMARY EXPERIMENT
# ============================================================

def decode_row_lane(
    pred,
    lane_id,
    row_anchor,
    original_w,
    original_h,
    local_width=1,
):

    (
        _,
        num_grid_row,
        num_cls_row,
        _,
    ) = pred[
        "loc_row"
    ].shape

    max_indices_row = (
        pred[
            "loc_row"
        ]
        .argmax(1)
        .cpu()
    )

    valid_row = (
        pred[
            "exist_row"
        ]
        .argmax(1)
        .cpu()
    )

    loc_row = (
        pred[
            "loc_row"
        ]
        .detach()
        .cpu()
    )

    points = []

    if (
        valid_row[
            0,
            :,
            lane_id,
        ].sum()
        <= num_cls_row / 2
    ):
        return points

    for k in range(
        valid_row.shape[1]
    ):

        if not valid_row[
            0,
            k,
            lane_id,
        ]:
            continue

        center = int(
            max_indices_row[
                0,
                k,
                lane_id,
            ]
        )

        inds = torch.tensor(
            list(
                range(
                    max(
                        0,
                        center
                        - local_width,
                    ),
                    min(
                        num_grid_row - 1,
                        center
                        + local_width,
                    )
                    + 1,
                )
            )
        )

        out_tmp = (
            loc_row[
                0,
                inds,
                k,
                lane_id,
            ].softmax(0)
            * inds.float()
        ).sum() + 0.5

        x = (
            float(out_tmp)
            / (
                num_grid_row
                - 1
            )
            * original_w
        )

        y = (
            float(
                row_anchor[k]
            )
            * original_h
        )

        points.append(
            (
                float(x),
                float(y),
            )
        )

    return points


def interpolate_x_at_y(
    points,
    target_y,
):

    if len(points) < 2:
        return None

    points = sorted(
        points,
        key=lambda p: p[1],
    )

    ys = np.asarray(
        [
            p[1]
            for p in points
        ],
        dtype=float,
    )

    xs = np.asarray(
        [
            p[0]
            for p in points
        ],
        dtype=float,
    )

    if (
        target_y < ys.min()
        or target_y > ys.max()
    ):
        return None

    unique_y, idx = np.unique(
        ys,
        return_index=True,
    )

    unique_x = xs[
        idx
    ]

    if len(unique_y) < 2:
        return None

    return float(
        np.interp(
            target_y,
            unique_y,
            unique_x,
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Clean old range output only.
    # Never touch primary lane results.
    # --------------------------------------------------------

    if OUT.exists():

        print(
            "Deleting old range-extension results:"
        )

        print(
            OUT
        )

        shutil.rmtree(
            OUT
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not PREREG.exists():

        raise RuntimeError(
            f"Missing preregistration: {PREREG}"
        )

    prereg = pd.read_csv(
        PREREG
    )

    expected_rows = {
        "primary_504": 504,
        "target_25m": 410,
        "target_30m": 402,
        "target_35m": 396,
        "target_40m": 391,
    }

    for label, expected_y in (
        expected_rows.items()
    ):

        match = prereg[
            prereg[
                "label"
            ]
            == label
        ]

        if len(match) != 1:

            raise RuntimeError(
                f"Preregistration missing "
                f"unique row: {label}"
            )

        actual_y = int(
            match.iloc[0][
                "frozen_row_y"
            ]
        )

        if actual_y != expected_y:

            raise RuntimeError(
                f"Frozen row changed for "
                f"{label}: "
                f"{actual_y} != {expected_y}"
            )

    print()
    print("=" * 100)
    print(
        "FINAL UFLD-v2 PHYSICAL-RANGE SWEEP"
    )
    print("=" * 100)

    print(
        "Config:",
        CONFIG,
    )

    print(
        "Weights:",
        WEIGHTS,
    )

    print(
        "Frozen epsilon:",
        EPSILON_M,
        "m",
    )

    print()
    print(
        prereg[
            [
                "label",
                "frozen_row_y",
                "status",
                "actual_median_camera_range_m",
            ]
        ].to_string(
            index=False
        )
    )

    net, cfg = load_model()

    transform = make_transform(
        cfg
    )

    per_frame_rows = []
    per_trial_rows = []


    # ========================================================
    # ALL 35 CLEAR + FOG DATASETS
    # ========================================================

    for severity in SEVERITIES:

        for trial in TRIALS:

            dataset = dataset_root(
                severity,
                trial,
            )

            csv_path = (
                dataset
                / "frames.csv"
            )

            frames_dir = (
                dataset
                / "frames"
            )

            geom_path = (
                GEOM_ROOT
                / (
                    f"trial_{trial}_"
                    f"range_geometry.csv"
                )
            )

            if not csv_path.exists():

                raise RuntimeError(
                    f"Missing recording CSV: "
                    f"{csv_path}"
                )

            if not frames_dir.exists():

                raise RuntimeError(
                    f"Missing frame directory: "
                    f"{frames_dir}"
                )

            if not geom_path.exists():

                raise RuntimeError(
                    f"Missing frozen geometry: "
                    f"{geom_path}"
                )

            recording = pd.read_csv(
                csv_path
            )

            geometry = pd.read_csv(
                geom_path
            )

            geometry = geometry[
                geometry[
                    "eligible"
                ]
                == 1
            ].copy()

            sample_ids = sorted(
                geometry[
                    "sample_id"
                ]
                .astype(int)
                .unique()
                .tolist()
            )

            recording_by_sid = (
                recording
                .set_index(
                    "sample_id"
                )
            )

            # ------------------------------------------------
            # Verify same route positions across weather.
            # ------------------------------------------------

            for sid in sample_ids:

                if sid not in (
                    recording_by_sid.index
                ):

                    raise RuntimeError(
                        f"Sample {sid} "
                        f"missing in "
                        f"{dataset.name}"
                    )

                geom_s = float(
                    geometry[
                        geometry[
                            "sample_id"
                        ]
                        == sid
                    ][
                        "route_s_m"
                    ].iloc[0]
                )

                data_s = float(
                    recording_by_sid.loc[
                        sid,
                        "route_s_m",
                    ]
                )

                if abs(
                    geom_s - data_s
                ) > 1e-6:

                    raise RuntimeError(
                        f"Route mismatch "
                        f"{dataset.name}, "
                        f"sample {sid}: "
                        f"{geom_s} vs "
                        f"{data_s}"
                    )

            print()
            print("=" * 82)

            condition_name = (
                "CLEAR"
                if severity == 0
                else f"FOG{severity}"
            )

            print(
                f"{condition_name} "
                f"| trial {trial}"
            )

            print("=" * 82)

            dataset_output = []


            # =================================================
            # ONE MODEL INFERENCE PER IMAGE
            # ALL FIVE ROWS EXTRACTED FROM SAME PREDICTION
            # =================================================

            for sid in sample_ids:

                image_path = (
                    frames_dir
                    / f"{sid:04d}.png"
                )

                if not image_path.exists():

                    raise RuntimeError(
                        f"Missing image: "
                        f"{image_path}"
                    )

                with Image.open(
                    image_path
                ) as pil:

                    pil = pil.convert(
                        "RGB"
                    )

                    (
                        original_w,
                        original_h,
                    ) = pil.size

                    tensor = transform(
                        pil
                    )

                tensor = tensor[
                    :,
                    -cfg.train_height:,
                    :,
                ]

                tensor = (
                    tensor
                    .unsqueeze(0)
                    .cuda()
                )

                with torch.no_grad():

                    pred = net(
                        tensor
                    )

                # Frozen ego lanes:
                # lane slot 1 = left
                # lane slot 2 = right

                left_points = decode_row_lane(
                    pred,
                    1,
                    cfg.row_anchor,
                    original_w,
                    original_h,
                )

                right_points = decode_row_lane(
                    pred,
                    2,
                    cfg.row_anchor,
                    original_w,
                    original_h,
                )

                sample_geometry = geometry[
                    geometry[
                        "sample_id"
                    ]
                    == sid
                ]

                for _, g in (
                    sample_geometry.iterrows()
                ):

                    row_y = float(
                        g[
                            "row_y"
                        ]
                    )

                    left_x = interpolate_x_at_y(
                        left_points,
                        row_y,
                    )

                    right_x = interpolate_x_at_y(
                        right_points,
                        row_y,
                    )

                    success = (
                        left_x is not None
                        and right_x is not None
                        and left_x
                        < right_x
                    )

                    gt_x = float(
                        g[
                            "lane_gt_x"
                        ]
                    )

                    m_per_px = float(
                        g[
                            "m_per_px"
                        ]
                    )

                    if success:

                        pred_center = (
                            0.5
                            * (
                                left_x
                                + right_x
                            )
                        )

                        signed_error_px = (
                            pred_center
                            - gt_x
                        )

                        error_px = abs(
                            signed_error_px
                        )

                        conditional_error_px = (
                            error_px
                        )

                        conditional_error_m = (
                            error_px
                            * m_per_px
                        )

                        fail_aware_error_px = (
                            error_px
                        )

                        fail_aware_error_m = (
                            error_px
                            * m_per_px
                        )

                        lane_width_px = (
                            right_x
                            - left_x
                        )

                    else:

                        pred_center = np.nan
                        signed_error_px = np.nan

                        conditional_error_px = (
                            np.nan
                        )

                        conditional_error_m = (
                            np.nan
                        )

                        fail_aware_error_px = (
                            FAILURE_ERROR_PX
                        )

                        fail_aware_error_m = (
                            FAILURE_ERROR_PX
                            * m_per_px
                        )

                        lane_width_px = np.nan

                    output_row = {

                        "severity":
                            severity,

                        "trial":
                            trial,

                        "sample_id":
                            sid,

                        "route_s_m":
                            float(
                                g[
                                    "route_s_m"
                                ]
                            ),

                        "row_label":
                            str(
                                g[
                                    "row_label"
                                ]
                            ),

                        "row_y":
                            int(
                                g[
                                    "row_y"
                                ]
                            ),

                        "preregister_status":
                            str(
                                g[
                                    "preregister_status"
                                ]
                            ),

                        "camera_range_m":
                            float(
                                g[
                                    "camera_range_m"
                                ]
                            ),

                        "m_per_px":
                            m_per_px,

                        "gt_x":
                            gt_x,

                        "left_x":
                            (
                                left_x
                                if left_x
                                is not None
                                else np.nan
                            ),

                        "right_x":
                            (
                                right_x
                                if right_x
                                is not None
                                else np.nan
                            ),

                        "pred_center_x":
                            pred_center,

                        "lane_width_px":
                            lane_width_px,

                        "success":
                            int(
                                success
                            ),

                        "signed_error_px":
                            signed_error_px,

                        "conditional_error_px":
                            conditional_error_px,

                        "conditional_error_m":
                            conditional_error_m,

                        "fail_aware_error_px":
                            fail_aware_error_px,

                        "fail_aware_error_m":
                            fail_aware_error_m,
                    }

                    dataset_output.append(
                        output_row
                    )

                    per_frame_rows.append(
                        output_row
                    )


            # =================================================
            # PER-TRIAL / PER-RANGE METRICS
            # =================================================

            dataset_df = pd.DataFrame(
                dataset_output
            )

            for (
                row_label,
                group,
            ) in dataset_df.groupby(
                "row_label"
            ):

                good = group[
                    group[
                        "success"
                    ]
                    == 1
                ]

                n_eligible = len(
                    group
                )

                n_success = int(
                    group[
                        "success"
                    ].sum()
                )

                availability_pct = (
                    100.0
                    * n_success
                    / n_eligible
                )

                if len(good) > 0:

                    conditional_px = float(
                        good[
                            "conditional_error_px"
                        ].mean()
                    )

                    conditional_m = float(
                        good[
                            "conditional_error_m"
                        ].mean()
                    )

                else:

                    conditional_px = np.nan
                    conditional_m = np.nan

                fail_aware_px = float(
                    group[
                        "fail_aware_error_px"
                    ].mean()
                )

                fail_aware_m = float(
                    group[
                        "fail_aware_error_m"
                    ].mean()
                )

                median_range = float(
                    group[
                        "camera_range_m"
                    ].median()
                )

                per_trial_rows.append(
                    {

                        "severity":
                            severity,

                        "trial":
                            trial,

                        "row_label":
                            row_label,

                        "row_y":
                            int(
                                group[
                                    "row_y"
                                ].iloc[0]
                            ),

                        "preregister_status":
                            str(
                                group[
                                    "preregister_status"
                                ].iloc[0]
                            ),

                        "n_eligible":
                            n_eligible,

                        "n_success":
                            n_success,

                        "availability_pct":
                            availability_pct,

                        "actual_median_range_m":
                            median_range,

                        "conditional_center_mae_px":
                            conditional_px,

                        "conditional_center_mae_m":
                            conditional_m,

                        "fail_aware_center_mae_px":
                            fail_aware_px,

                        "fail_aware_center_mae_m":
                            fail_aware_m,
                    }
                )

                print(
                    f"{row_label:12s} "
                    f"N={n_eligible:3d} | "
                    f"success={n_success:3d} | "
                    f"avail="
                    f"{availability_pct:6.2f}% | "
                    f"cond="
                    f"{conditional_m:8.4f} m | "
                    f"fail-aware="
                    f"{fail_aware_m:8.4f} m"
                )


    # ========================================================
    # BUILD DATAFRAMES
    # ========================================================

    per_frame = pd.DataFrame(
        per_frame_rows
    )

    per_trial = pd.DataFrame(
        per_trial_rows
    )


    # ========================================================
    # PAIRED CLEAR BASELINE PER CONFIGURATION
    # ========================================================

    clear_baseline = (
        per_trial[
            per_trial[
                "severity"
            ]
            == 0
        ][
            [
                "trial",
                "row_label",
                "fail_aware_center_mae_m",
            ]
        ]
        .rename(
            columns={
                "fail_aware_center_mae_m":
                    "clear_fail_aware_E_m"
            }
        )
    )

    per_trial = per_trial.merge(
        clear_baseline,
        on=[
            "trial",
            "row_label",
        ],
        how="left",
    )

    if per_trial[
        "clear_fail_aware_E_m"
    ].isna().any():

        raise RuntimeError(
            "Missing paired Clear "
            "baseline for R_lane."
        )


    # ========================================================
    # PHYSICAL R_lane
    # ========================================================

    per_trial[
        "R_lane"
    ] = np.minimum(
        1.0,
        (
            per_trial[
                "clear_fail_aware_E_m"
            ]
            + EPSILON_M
        )
        /
        (
            per_trial[
                "fail_aware_center_mae_m"
            ]
            + EPSILON_M
        ),
    )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary_rows = []

    for (
        row_label,
        severity,
    ), group in per_trial.groupby(
        [
            "row_label",
            "severity",
        ]
    ):

        row_y = int(
            group[
                "row_y"
            ].iloc[0]
        )

        status = str(
            group[
                "preregister_status"
            ].iloc[0]
        )

        if row_label == "primary_504":

            analysis_role = (
                "PRIMARY_EXISTING_ENDPOINT"
            )

        elif status == "KEEP":

            analysis_role = (
                "FORMAL_RANGE_CONTROL"
            )

        else:

            analysis_role = (
                "EXPLORATORY_INSUFFICIENT_COVERAGE"
            )

        summary_rows.append(
            {

                "row_label":
                    row_label,

                "row_y":
                    row_y,

                "analysis_role":
                    analysis_role,

                "preregister_status":
                    status,

                "severity":
                    int(
                        severity
                    ),

                "n_trials":
                    len(group),

                "eligible_frames_min":
                    int(
                        group[
                            "n_eligible"
                        ].min()
                    ),

                "eligible_frames_max":
                    int(
                        group[
                            "n_eligible"
                        ].max()
                    ),

                "actual_range_mean_m":
                    float(
                        group[
                            "actual_median_range_m"
                        ].mean()
                    ),

                "availability_mean_pct":
                    float(
                        group[
                            "availability_pct"
                        ].mean()
                    ),

                "availability_std_pct":
                    float(
                        group[
                            "availability_pct"
                        ].std(
                            ddof=1
                        )
                    ),

                "conditional_mae_px_mean":
                    float(
                        group[
                            "conditional_center_mae_px"
                        ].mean()
                    ),

                "conditional_mae_px_std":
                    float(
                        group[
                            "conditional_center_mae_px"
                        ].std(
                            ddof=1
                        )
                    ),

                "conditional_mae_m_mean":
                    float(
                        group[
                            "conditional_center_mae_m"
                        ].mean()
                    ),

                "conditional_mae_m_std":
                    float(
                        group[
                            "conditional_center_mae_m"
                        ].std(
                            ddof=1
                        )
                    ),

                "fail_aware_mae_px_mean":
                    float(
                        group[
                            "fail_aware_center_mae_px"
                        ].mean()
                    ),

                "fail_aware_mae_px_std":
                    float(
                        group[
                            "fail_aware_center_mae_px"
                        ].std(
                            ddof=1
                        )
                    ),

                "fail_aware_mae_m_mean":
                    float(
                        group[
                            "fail_aware_center_mae_m"
                        ].mean()
                    ),

                "fail_aware_mae_m_std":
                    float(
                        group[
                            "fail_aware_center_mae_m"
                        ].std(
                            ddof=1
                        )
                    ),

                "R_lane_mean":
                    float(
                        group[
                            "R_lane"
                        ].mean()
                    ),

                "R_lane_std":
                    float(
                        group[
                            "R_lane"
                        ].std(
                            ddof=1
                        )
                    ),
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

    summary[
        "_order"
    ] = summary[
        "row_label"
    ].map(
        order
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
            columns=[
                "_order"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    per_frame_path = (
        OUT
        / "final_per_frame.csv"
    )

    per_trial_path = (
        OUT
        / "final_per_trial.csv"
    )

    summary_path = (
        OUT
        / "final_summary.csv"
    )

    per_frame.to_csv(
        per_frame_path,
        index=False,
    )

    per_trial.to_csv(
        per_trial_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )


    # ========================================================
    # PRIMARY y=504 SANITY CHECK
    # ========================================================

    print()
    print("=" * 110)
    print(
        "PRIMARY y=504 SANITY CHECK"
    )
    print("=" * 110)

    primary = summary[
        summary[
            "row_label"
        ]
        == "primary_504"
    ]

    print(
        primary[
            [
                "severity",
                "availability_mean_pct",
                "conditional_mae_px_mean",
                "fail_aware_mae_px_mean",
                "R_lane_mean",
                "R_lane_std",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Reminder: the original pixel-based "
        "primary table remains authoritative."
    )

    print(
        "This new R_lane is the secondary "
        "meter-based range-control version."
    )


    # ========================================================
    # FINAL RANGE TABLES
    # ========================================================

    print()
    print("=" * 110)
    print(
        "FINAL UFLD-v2 RANGE RESULTS"
    )
    print("=" * 110)

    for row_label in [
        "primary_504",
        "target_25m",
        "target_30m",
        "target_35m",
        "target_40m",
    ]:

        table = summary[
            summary[
                "row_label"
            ]
            == row_label
        ].copy()

        if len(table) == 0:
            continue

        first = table.iloc[0]

        print()
        print("-" * 110)

        print(
            f"{row_label} | "
            f"y={int(first['row_y'])} | "
            f"range≈"
            f"{first['actual_range_mean_m']:.3f} m"
        )

        print(
            "Role:",
            first[
                "analysis_role"
            ],
        )

        print("-" * 110)

        print(
            table[
                [
                    "severity",
                    "eligible_frames_min",
                    "availability_mean_pct",
                    "availability_std_pct",
                    "conditional_mae_m_mean",
                    "fail_aware_mae_m_mean",
                    "R_lane_mean",
                    "R_lane_std",
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "Threshold crossings:"
        )

        degraded = table[
            table[
                "severity"
            ]
            > 0
        ]

        for threshold in [
            0.40,
            0.50,
            0.60,
        ]:

            crossed = degraded[
                degraded[
                    "R_lane_mean"
                ]
                < threshold
            ]

            if len(crossed) > 0:

                first_cross = int(
                    crossed.iloc[0][
                        "severity"
                    ]
                )

                print(
                    f"  R_lane < "
                    f"{threshold:.2f}: "
                    f"Fog{first_cross}"
                )

            else:

                print(
                    f"  R_lane < "
                    f"{threshold:.2f}: "
                    "not reached"
                )


    # ========================================================
    # COMPACT FOG90 RANGE COMPARISON
    # ========================================================

    fog90 = summary[
        summary[
            "severity"
        ]
        == 90
    ].copy()

    print()
    print("=" * 110)
    print(
        "FOG90 RANGE COMPARISON"
    )
    print("=" * 110)

    print(
        fog90[
            [
                "row_label",
                "actual_range_mean_m",
                "analysis_role",
                "eligible_frames_min",
                "availability_mean_pct",
                "fail_aware_mae_m_mean",
                "R_lane_mean",
                "R_lane_std",
            ]
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # OUTPUT LOCATIONS
    # ========================================================

    print()
    print("=" * 110)
    print("SAVED")
    print("=" * 110)

    print(
        per_frame_path
    )

    print(
        per_trial_path
    )

    print(
        summary_path
    )

    print()
    print(
        "Formal quantitative range control:"
    )

    print(
        "  primary_504 (~8.9 m)"
    )

    print(
        "  target_25m (~25.1 m)"
    )

    print()
    print(
        "Exploratory only because "
        "N < 50:"
    )

    print(
        "  target_30m"
    )

    print(
        "  target_35m"
    )

    print(
        "  target_40m"
    )


if __name__ == "__main__":
    main()
