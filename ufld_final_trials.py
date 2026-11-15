#!/usr/bin/env python3

"""
Run UFLD-v2 on all 35 FINAL fog experiment recordings.

Final datasets:
    Clear: 5 trials
    Fog15: 5 trials
    Fog30: 5 trials
    Fog45: 5 trials
    Fog60: 5 trials
    Fog75: 5 trials
    Fog90: 5 trials

Total = 35 recordings

Keeps the same validated UFLD setup:
    - CULane ResNet18
    - same preprocessing
    - lane slot 1 = left
    - lane slot 2 = right
    - y = 504
    - failure penalty = 640 px

IMPORTANT:
The results produced here are BEFORE the final lane-geometry eligibility mask.
We will apply that mask in the next analysis step.
"""

from pathlib import Path
import csv
import sys

import numpy as np
import torch
from PIL import Image


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

UFLD_ROOT = (
    Path.home()
    / "Ultra-Fast-Lane-Detection-v2"
)

if not UFLD_ROOT.exists():
    raise RuntimeError(
        f"Could not find UFLD repository:\n"
        f"{UFLD_ROOT}"
    )


# Make sure UFLD's own modules are imported
# instead of unrelated pip packages.
sys.path.insert(
    0,
    str(UFLD_ROOT),
)

sys.path.insert(
    1,
    str(ROOT),
)


# ============================================================
# IMPORT EXISTING VALIDATED UFLD FUNCTIONS
# ============================================================

import ufld_fog_sweep as base


# ============================================================
# FINAL OUTPUT
# ============================================================

OUTPUT_DIR = (
    ROOT
    / "results"
    / "final"
    / "lane_ufld"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FINAL EXPERIMENT SETTINGS
# ============================================================

TRIALS = [
    1,
    2,
    3,
    4,
    5,
]

SEVERITIES = [
    0,
    15,
    30,
    45,
    60,
    75,
    90,
]

LOOKAHEAD_Y = 504.0

FAILURE_ERROR_PX = 640.0


# ============================================================
# FIND MODEL FILES
# ============================================================

def find_model_files():

    config_candidates = [

        UFLD_ROOT
        / "configs"
        / "culane_res18.py",

        ROOT
        / "configs"
        / "culane_res18.py",
    ]

    weight_candidates = [

        UFLD_ROOT
        / "weights"
        / "culane_res18.pth",

        ROOT
        / "weights"
        / "culane_res18.pth",

        ROOT
        / "culane_res18.pth",
    ]

    config_path = None

    for path in config_candidates:

        if path.exists():

            config_path = path
            break


    weights_path = None

    for path in weight_candidates:

        if path.exists():

            weights_path = path
            break


    # Fallback search
    if config_path is None:

        matches = sorted(
            UFLD_ROOT.rglob(
                "culane_res18.py"
            )
        )

        if matches:

            config_path = matches[0]


    if weights_path is None:

        matches = sorted(
            UFLD_ROOT.rglob(
                "*culane*res18*.pth"
            )
        )

        if matches:

            weights_path = matches[0]


    if config_path is None:

        raise RuntimeError(
            "Could not find "
            "culane_res18.py"
        )


    if weights_path is None:

        raise RuntimeError(
            "Could not find "
            "culane_res18.pth"
        )


    return (
        config_path,
        weights_path,
    )


# ============================================================
# DATASET NAME
# ============================================================

def get_dataset_path(
    severity,
    trial,
):

    if severity == 0:

        name = (
            "stop3_clear_0_"
            "drift_0p75_"
            f"trial_{trial}"
        )

    else:

        name = (
            f"stop3_fog_{severity}_"
            "drift_0p75_"
            f"trial_{trial}"
        )


    return (
        ROOT
        / "data"
        / "recordings"
        / name
    )


# ============================================================
# CONDITION LABEL
# ============================================================

def get_condition_name(
    severity,
):

    if severity == 0:

        return "clear"

    return f"fog{severity}"


# ============================================================
# PROCESS ONE TRIAL
# ============================================================

def evaluate_trial(
    net,
    cfg,
    transform,
    severity,
    trial,
):

    dataset = get_dataset_path(
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


    if not csv_path.exists():

        raise RuntimeError(
            f"Missing CSV:\n"
            f"{csv_path}"
        )


    if not frames_dir.exists():

        raise RuntimeError(
            f"Missing frame directory:\n"
            f"{frames_dir}"
        )


    with csv_path.open(
        newline="",
    ) as f:

        input_rows = list(
            csv.DictReader(f)
        )


    output_rows = []

    total_frames = 0

    junction_excluded = 0

    evaluated = 0

    successes = 0

    failures = 0


    # ========================================================
    # FRAME LOOP
    # ========================================================

    for row in input_rows:

        total_frames += 1

        sample_id = int(
            float(
                row[
                    "sample_id"
                ]
            )
        )


        is_junction = int(
            float(
                row[
                    "is_junction"
                ]
            )
        )


        # ----------------------------------------------------
        # Same initial eligibility behavior as pilot.
        # Final lane-marking mask comes later.
        # ----------------------------------------------------

        if is_junction:

            junction_excluded += 1
            continue


        gt_text = row.get(
            "lane_gt_x",
            "",
        )


        if (
            gt_text is None
            or gt_text == ""
        ):

            continue


        gt_x = float(
            gt_text
        )

        evaluated += 1


        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_path = (
            frames_dir
            / f"{sample_id:04d}.png"
        )


        if not image_path.exists():

            raise RuntimeError(
                f"Missing image:\n"
                f"{image_path}"
            )


        pil_image = (
            Image
            .open(
                image_path
            )
            .convert(
                "RGB"
            )
        )


        (
            original_w,
            original_h,
        ) = pil_image.size


        # ----------------------------------------------------
        # SAME UFLD PREPROCESSING
        # ----------------------------------------------------

        tensor = transform(
            pil_image
        )


        tensor = tensor[
            :,
            -cfg.train_height:,
            :
        ]


        tensor = (
            tensor
            .unsqueeze(0)
            .cuda()
        )


        # ----------------------------------------------------
        # INFERENCE
        # ----------------------------------------------------

        with torch.no_grad():

            prediction = net(
                tensor
            )


        # ----------------------------------------------------
        # SAME FIXED LANE SLOTS
        #
        # slot 1 = left
        # slot 2 = right
        #
        # GT is NOT used for lane selection.
        # ----------------------------------------------------

        left_points = (
            base.decode_row_lane(
                prediction,
                1,
                cfg.row_anchor,
                original_w,
                original_h,
            )
        )


        right_points = (
            base.decode_row_lane(
                prediction,
                2,
                cfg.row_anchor,
                original_w,
                original_h,
            )
        )


        # ----------------------------------------------------
        # INTERPOLATE AT y = 504
        # ----------------------------------------------------

        left_x = (
            base.interpolate_x_at_y(
                left_points,
                LOOKAHEAD_Y,
            )
        )


        right_x = (
            base.interpolate_x_at_y(
                right_points,
                LOOKAHEAD_Y,
            )
        )


        # ----------------------------------------------------
        # SUCCESS RULE
        # ----------------------------------------------------

        success = (

            left_x is not None

            and

            right_x is not None

            and

            left_x < right_x
        )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        if success:

            successes += 1


            pred_center_x = (
                0.5
                * (
                    left_x
                    + right_x
                )
            )


            signed_error_px = (
                pred_center_x
                - gt_x
            )


            error_px = abs(
                signed_error_px
            )


            width_px = (
                right_x
                - left_x
            )


        else:

            failures += 1


            pred_center_x = None

            signed_error_px = None

            width_px = None


            # Frozen failure penalty
            error_px = (
                FAILURE_ERROR_PX
            )


        # ----------------------------------------------------
        # SAVE FRAME RESULT
        # ----------------------------------------------------

        output_rows.append({

            "trial":
                trial,

            "severity":
                severity,

            "condition":
                get_condition_name(
                    severity
                ),

            "sample_id":
                sample_id,

            "route_s_m":
                row.get(
                    "route_s_m",
                    "",
                ),

            "lane_gt_x":
                gt_x,

            "left_x_y504":
                (
                    ""
                    if left_x is None
                    else left_x
                ),

            "right_x_y504":
                (
                    ""
                    if right_x is None
                    else right_x
                ),

            "pred_center_x_y504":
                (
                    ""
                    if pred_center_x
                    is None
                    else pred_center_x
                ),

            "signed_error_px":
                (
                    ""
                    if signed_error_px
                    is None
                    else signed_error_px
                ),

            "error_px":
                error_px,

            "width_px":
                (
                    ""
                    if width_px is None
                    else width_px
                ),

            "success":
                int(
                    success
                ),
        })


    # ========================================================
    # CHECK
    # ========================================================

    if len(output_rows) == 0:

        raise RuntimeError(
            f"No evaluable frames:\n"
            f"{dataset}"
        )


    # ========================================================
    # SAVE PER-FRAME CSV
    # ========================================================

    condition = (
        get_condition_name(
            severity
        )
    )


    output_path = (

        OUTPUT_DIR

        / (
            f"{condition}_"
            f"trial_{trial}_"
            "per_frame.csv"
        )
    )


    with output_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=list(
                output_rows[0].keys()
            ),
        )


        writer.writeheader()

        writer.writerows(
            output_rows
        )


    # ========================================================
    # PRE-MASK SUMMARY
    # ========================================================

    errors = np.array(

        [
            float(
                row[
                    "error_px"
                ]
            )
            for row
            in output_rows
        ],

        dtype=float,
    )


    availability_pct = (

        100.0
        * successes
        / evaluated
    )


    print(

        f"{condition:>6} "
        f"trial {trial}: "

        f"evaluated={evaluated:3d} | "

        f"success={successes:3d} | "

        f"fail={failures:3d} | "

        f"availability="
        f"{availability_pct:6.2f}% | "

        f"pre-mask MAE="
        f"{errors.mean():7.2f}px"
    )


    return {

        "trial":
            trial,

        "severity":
            severity,

        "condition":
            condition,

        "total_frames":
            total_frames,

        "junction_excluded":
            junction_excluded,

        "evaluated_pre_mask":
            evaluated,

        "successes_pre_mask":
            successes,

        "failures_pre_mask":
            failures,

        "availability_pre_mask_pct":
            availability_pct,

        "failure_aware_mae_pre_mask_px":
            float(
                errors.mean()
            ),

        "per_frame_csv":
            str(
                output_path
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    (
        config_path,
        weights_path,
    ) = find_model_files()


    print()

    print(
        "=" * 90
    )

    print(
        "FINAL UFLD-v2 FOG SWEEP"
    )

    print(
        "=" * 90
    )


    print(
        "Config:",
        config_path,
    )

    print(
        "Weights:",
        weights_path,
    )

    print(
        "Primary row:",
        LOOKAHEAD_Y,
    )

    print(
        "Lane slots:"
        " 1=left, 2=right"
    )

    print(
        "Failure penalty:",
        FAILURE_ERROR_PX,
        "px",
    )

    print(
        "Expected datasets: 35"
    )

    print()


    # ========================================================
    # LOAD MODEL ONCE
    # ========================================================

    net, cfg = (
        base.load_model(
            str(
                config_path
            ),
            str(
                weights_path
            ),
        )
    )


    transform = (
        base.make_transform(
            cfg
        )
    )


    # ========================================================
    # RUN ALL 35 FINAL DATASETS
    # ========================================================

    summaries = []


    for severity in SEVERITIES:

        for trial in TRIALS:

            result = evaluate_trial(

                net,
                cfg,
                transform,
                severity,
                trial,
            )

            summaries.append(
                result
            )


    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary_path = (

        OUTPUT_DIR
        / "pre_mask_summary.csv"
    )


    with summary_path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(

            f,

            fieldnames=list(
                summaries[0].keys()
            ),
        )


        writer.writeheader()

        writer.writerows(
            summaries
        )


    # ========================================================
    # FINAL CHECK
    # ========================================================

    generated = list(

        OUTPUT_DIR.glob(
            "*_trial_*_per_frame.csv"
        )
    )


    print()

    print(
        "=" * 90
    )

    print(
        "UFLD FINAL INFERENCE COMPLETE"
    )

    print(
        "=" * 90
    )


    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )


    print()

    print(
        "Pre-mask summary:"
    )

    print(
        summary_path
    )


    print()

    print(
        "Expected per-frame files: 35"
    )

    print(
        "Actual:",
        len(
            generated
        ),
    )


    if len(generated) != 35:

        raise RuntimeError(

            "Expected 35 final UFLD "
            f"per-frame CSVs, got "
            f"{len(generated)}"
        )


    print()

    print(
        "PASS: all 35 UFLD "
        "final trial outputs generated."
    )


    print()

    print(
        "IMPORTANT:"
    )

    print(
        "These are PRE-MASK results."
    )

    print(
        "Do not use them as final paper "
        "R_lane values yet."
    )

    print(
        "Next step = apply the final "
        "row-ground lane eligibility mask "
        "and compute paired R_lane."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
