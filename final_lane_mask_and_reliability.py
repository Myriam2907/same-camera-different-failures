#!/usr/bin/env python3

from pathlib import Path
import sys

import carla
import numpy as np
import pandas as pd

from common.projection import (
    build_camera_matrix,
    world_to_image,
)


# ============================================================
# PATHS / CONSTANTS
# ============================================================

ROOT = Path(__file__).resolve().parent

ROUTE_CSV = (
    ROOT
    / "data"
    / "routes"
    / "stop3_route.csv"
)

UFLD_DIR = (
    ROOT
    / "results"
    / "final"
    / "lane_ufld"
)

OUT_DIR = (
    ROOT
    / "results"
    / "final"
    / "lane"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


IMAGE_W = 1280
IMAGE_H = 720
FOV = 90.0

LOOKAHEAD_ROW = 504.0

CAMERA_X = 1.5
CAMERA_Y = 0.0
CAMERA_Z = 1.7

LOOK_FORWARD_M = 60.0

EPSILON_PX = 1.0

FAILURE_ERROR_PX = 640.0

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


# ============================================================
# HELPERS
# ============================================================

def condition_name(severity):

    if severity == 0:
        return "clear"

    return f"fog{severity}"


def recording_dir(
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


def ufld_csv(
    severity,
    trial,
):

    return (
        UFLD_DIR
        / (
            f"{condition_name(severity)}_"
            f"trial_{trial}_"
            f"per_frame.csv"
        )
    )


def marking_name(
    marking,
):

    if marking is None:
        return "NONE"

    return str(
        marking.type
    )


def marking_exists(
    marking,
):

    if marking is None:
        return False

    text = str(
        marking.type
    ).upper()

    return not (
        text.endswith(".NONE")
        or text == "NONE"
        or "LANEMARKINGTYPE.NONE" in text
    )


# ============================================================
# CAMERA TRANSFORM
# ============================================================

def build_world_to_camera(
    row,
):

    vehicle_transform = carla.Transform(

        carla.Location(
            x=float(row["ego_x"]),
            y=float(row["ego_y"]),
            z=float(row["ego_z"]),
        ),

        carla.Rotation(
            pitch=float(row["ego_pitch"]),
            yaw=float(row["ego_yaw"]),
            roll=float(row["ego_roll"]),
        ),
    )


    camera_relative = carla.Transform(

        carla.Location(
            x=CAMERA_X,
            y=CAMERA_Y,
            z=CAMERA_Z,
        ),

        carla.Rotation(
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
        ),
    )


    vehicle_to_world = np.array(
        vehicle_transform.get_matrix(),
        dtype=float,
    )

    camera_to_vehicle = np.array(
        camera_relative.get_matrix(),
        dtype=float,
    )


    camera_to_world = (
        vehicle_to_world
        @ camera_to_vehicle
    )


    world_to_camera_matrix = (
        np.linalg.inv(
            camera_to_world
        )
    )


    camera_world_xyz = (
        camera_to_world[
            :3,
            3
        ].copy()
    )


    return (
        world_to_camera_matrix,
        camera_world_xyz,
    )


# ============================================================
# FIND PHYSICAL ROUTE POINT SEEN AT y=504
# ============================================================

def find_row_target(
    route,
    current_s,
    world_to_camera_matrix,
    K,
):

    candidate = route[
        (
            route["route_s_m"]
            >= current_s - 1e-6
        )
        &
        (
            route["route_s_m"]
            <= current_s + LOOK_FORWARD_M
        )
    ].copy()


    projected = []


    for _, point in candidate.iterrows():

        location = carla.Location(

            x=float(
                point["lane_x"]
            ),

            y=float(
                point["lane_y"]
            ),

            z=float(
                point["lane_z"]
            ),
        )


        result = world_to_image(

            location,
            world_to_camera_matrix,
            K,
        )


        if result is None:
            continue


        u, v, depth = result


        if not (
            np.isfinite(u)
            and np.isfinite(v)
            and np.isfinite(depth)
        ):
            continue


        if depth <= 0:
            continue


        projected.append({

            "u": float(u),
            "v": float(v),

            "x": float(
                point["lane_x"]
            ),

            "y": float(
                point["lane_y"]
            ),

            "z": float(
                point["lane_z"]
            ),

            "route_s_m": float(
                point["route_s_m"]
            ),
        })


    if len(projected) < 2:
        return None


    # Find the route segment crossing y=504.
    crossings = []


    for i in range(
        len(projected) - 1
    ):

        a = projected[i]
        b = projected[i + 1]


        va = a["v"]
        vb = b["v"]


        if abs(vb - va) < 1e-12:
            continue


        if (
            LOOKAHEAD_ROW
            < min(va, vb)
            or
            LOOKAHEAD_ROW
            > max(va, vb)
        ):
            continue


        t = (
            LOOKAHEAD_ROW
            - va
        ) / (
            vb - va
        )


        if (
            t < -1e-6
            or
            t > 1.0 + 1e-6
        ):
            continue


        t = float(
            np.clip(
                t,
                0.0,
                1.0,
            )
        )


        target = {

            "u":
                a["u"]
                + t
                * (
                    b["u"]
                    - a["u"]
                ),

            "x":
                a["x"]
                + t
                * (
                    b["x"]
                    - a["x"]
                ),

            "y":
                a["y"]
                + t
                * (
                    b["y"]
                    - a["y"]
                ),

            "z":
                a["z"]
                + t
                * (
                    b["z"]
                    - a["z"]
                ),

            "route_s_m":
                a["route_s_m"]
                + t
                * (
                    b["route_s_m"]
                    - a["route_s_m"]
                ),
        }


        crossings.append(
            target
        )


    if len(crossings) == 0:
        return None


    # The first physical crossing ahead is the
    # lane target associated with this image row.
    return crossings[0]


# ============================================================
# BUILD ONE TRIAL'S GEOMETRY MASK
# ============================================================

def build_trial_mask(
    trial,
    route,
    carla_map,
    K,
):

    clear_csv = (
        recording_dir(
            0,
            trial,
        )
        / "frames.csv"
    )


    metadata = pd.read_csv(
        clear_csv
    )


    ufld = pd.read_csv(
        ufld_csv(
            0,
            trial,
        )
    )


    ufld_ids = set(
        ufld["sample_id"]
        .astype(int)
        .tolist()
    )


    output = []

    self_check_errors = []


    for _, row in metadata.iterrows():

        sample_id = int(
            row["sample_id"]
        )


        # UFLD inference already excludes
        # ego-junction / missing-GT frames.
        if sample_id not in ufld_ids:
            continue


        if pd.isna(
            row["lane_gt_x"]
        ):
            continue


        (
            world_to_camera_matrix,
            camera_xyz,
        ) = build_world_to_camera(
            row
        )


        target = find_row_target(

            route,

            float(
                row["route_s_m"]
            ),

            world_to_camera_matrix,

            K,
        )


        if target is None:

            output.append({

                "trial":
                    trial,

                "sample_id":
                    sample_id,

                "route_s_m":
                    float(
                        row["route_s_m"]
                    ),

                "eligible":
                    0,

                "reason":
                    "NO_ROW_CROSSING",
            })

            continue


        projected_x = float(
            target["u"]
        )


        recorded_gt_x = float(
            row["lane_gt_x"]
        )


        gt_difference = abs(
            projected_x
            - recorded_gt_x
        )


        self_check_errors.append(
            gt_difference
        )


        target_location = carla.Location(

            x=float(
                target["x"]
            ),

            y=float(
                target["y"]
            ),

            z=float(
                target["z"]
            ),
        )


        waypoint = carla_map.get_waypoint(

            target_location,

            project_to_road=True,

            lane_type=(
                carla.LaneType.Driving
            ),
        )


        if waypoint is None:

            eligible = False
            reason = (
                "NO_TARGET_WAYPOINT"
            )

            left_marking = "NONE"
            right_marking = "NONE"

            target_junction = None
            lane_width_m = np.nan


        else:

            target_junction = bool(
                waypoint.is_junction
            )


            left_marking = (
                marking_name(
                    waypoint.left_lane_marking
                )
            )


            right_marking = (
                marking_name(
                    waypoint.right_lane_marking
                )
            )


            left_exists = (
                marking_exists(
                    waypoint.left_lane_marking
                )
            )


            right_exists = (
                marking_exists(
                    waypoint.right_lane_marking
                )
            )


            lane_width_m = float(
                waypoint.lane_width
            )


            eligible = (

                not target_junction

                and

                left_exists

                and

                right_exists
            )


            if target_junction:

                reason = (
                    "TARGET_JUNCTION"
                )

            elif not left_exists:

                reason = (
                    "LEFT_MARKING_NONE"
                )

            elif not right_exists:

                reason = (
                    "RIGHT_MARKING_NONE"
                )

            else:

                reason = "ELIGIBLE"


        target_xyz = np.array([

            target["x"],
            target["y"],
            target["z"],

        ], dtype=float)


        distance_m = float(
            np.linalg.norm(
                target_xyz
                - camera_xyz
            )
        )


        output.append({

            "trial":
                trial,

            "sample_id":
                sample_id,

            "route_s_m":
                float(
                    row["route_s_m"]
                ),

            "lane_gt_x_recorded":
                recorded_gt_x,

            "lane_gt_x_reprojected":
                projected_x,

            "gt_selfcheck_abs_px":
                gt_difference,

            "target_route_s_m":
                float(
                    target[
                        "route_s_m"
                    ]
                ),

            "target_x":
                float(
                    target["x"]
                ),

            "target_y":
                float(
                    target["y"]
                ),

            "target_z":
                float(
                    target["z"]
                ),

            "camera_to_target_m":
                distance_m,

            "target_is_junction":
                (
                    ""
                    if target_junction
                    is None
                    else int(
                        target_junction
                    )
                ),

            "left_marking_type":
                left_marking,

            "right_marking_type":
                right_marking,

            "lane_width_m":
                lane_width_m,

            "eligible":
                int(
                    eligible
                ),

            "reason":
                reason,
        })


    mask = pd.DataFrame(
        output
    )


    if len(
        self_check_errors
    ) == 0:

        raise RuntimeError(
            f"Trial {trial}: "
            f"no lane GT self-checks."
        )


    mean_check = float(
        np.mean(
            self_check_errors
        )
    )


    max_check = float(
        np.max(
            self_check_errors
        )
    )


    print()

    print(
        "=" * 72
    )

    print(
        f"TRIAL {trial} GEOMETRY"
    )

    print(
        "=" * 72
    )


    print(
        "GT reprojection mean abs error:",
        f"{mean_check:.6f} px"
    )


    print(
        "GT reprojection max abs error:",
        f"{max_check:.6f} px"
    )


    # Strong guard against using the wrong
    # camera transform or route geometry.
    if max_check > 0.20:

        raise RuntimeError(

            f"Trial {trial}: "
            f"GT geometry self-check FAILED. "

            f"Max difference = "
            f"{max_check:.6f} px"
        )


    eligible_count = int(
        mask["eligible"].sum()
    )


    print(
        "UFLD candidate rows:",
        len(mask)
    )


    print(
        "Final eligible rows:",
        eligible_count
    )


    eligible_rows = mask[
        mask["eligible"] == 1
    ]


    if len(
        eligible_rows
    ) == 0:

        raise RuntimeError(
            f"Trial {trial}: "
            f"zero eligible rows."
        )


    print(
        "Physical lookahead median:",
        f"{eligible_rows['camera_to_target_m'].median():.3f} m"
    )


    print(
        "Physical lookahead range:",
        f"{eligible_rows['camera_to_target_m'].min():.3f}"
        " to "
        f"{eligible_rows['camera_to_target_m'].max():.3f} m"
    )


    mask_path = (
        OUT_DIR
        / f"trial_{trial}_y504_geometry_mask.csv"
    )


    mask.to_csv(
        mask_path,
        index=False,
    )


    print(
        "Saved:",
        mask_path
    )


    return mask


# ============================================================
# APPLY MASK + COMPUTE LANE METRICS
# ============================================================

def compute_metrics(
    trial,
    severity,
    mask,
):

    path = ufld_csv(
        severity,
        trial,
    )


    data = pd.read_csv(
        path
    )


    eligible_ids = set(

        mask.loc[
            mask["eligible"] == 1,
            "sample_id",
        ]
        .astype(int)
        .tolist()
    )


    selected = data[
        data["sample_id"]
        .astype(int)
        .isin(
            eligible_ids
        )
    ].copy()


    if len(
        selected
    ) != len(
        eligible_ids
    ):

        missing = sorted(

            eligible_ids

            - set(
                selected[
                    "sample_id"
                ]
                .astype(int)
                .tolist()
            )
        )


        raise RuntimeError(

            f"Trial {trial}, "
            f"severity {severity}: "

            f"missing eligible "
            f"UFLD IDs: {missing}"
        )


    success = (
        selected[
            "success"
        ].astype(int)
        == 1
    )


    n = len(
        selected
    )


    n_success = int(
        success.sum()
    )


    n_fail = (
        n
        - n_success
    )


    availability_pct = (
        100.0
        * n_success
        / n
    )


    # --------------------------------------------
    # SUCCESS-ONLY CENTER MAE
    # --------------------------------------------

    if n_success > 0:

        conditional_mae_px = float(

            selected.loc[
                success,
                "error_px",
            ]
            .astype(float)
            .mean()
        )

    else:

        conditional_mae_px = np.nan


    # --------------------------------------------
    # FAILURE-AWARE CENTER MAE
    #
    # UFLD output already stores 640 px for
    # detector failures.
    # --------------------------------------------

    failure_aware_mae_px = float(

        selected[
            "error_px"
        ]
        .astype(float)
        .mean()
    )


    return {

        "trial":
            trial,

        "severity":
            severity,

        "condition":
            condition_name(
                severity
            ),

        "n_eligible":
            n,

        "n_success":
            n_success,

        "n_failure":
            n_fail,

        "availability_pct":
            availability_pct,

        "conditional_center_mae_px":
            conditional_mae_px,

        "failure_aware_center_mae_px":
            failure_aware_mae_px,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL LANE MASK + R_lane"
    )

    print(
        "=" * 80
    )


    # --------------------------------------------------------
    # Load frozen route
    # --------------------------------------------------------

    route = pd.read_csv(
        ROUTE_CSV
    )


    required_route_cols = {

        "route_s_m",

        "lane_x",
        "lane_y",
        "lane_z",
    }


    missing = (
        required_route_cols
        - set(
            route.columns
        )
    )


    if missing:

        raise RuntimeError(

            "Frozen route missing columns: "
            + str(
                sorted(
                    missing
                )
            )
        )


    # --------------------------------------------------------
    # Connect to CARLA map only.
    # No actors are created.
    # --------------------------------------------------------

    client = carla.Client(
        "127.0.0.1",
        2000,
    )


    client.set_timeout(
        10.0
    )


    world = client.get_world()

    carla_map = (
        world.get_map()
    )


    print(
        "Connected map:",
        carla_map.name
    )


    if (
        "Town10HD_Opt"
        not in carla_map.name
    ):

        raise RuntimeError(

            "Expected Town10HD_Opt, "
            f"got {carla_map.name}"
        )


    K = build_camera_matrix(

        IMAGE_W,
        IMAGE_H,
        FOV,
    )


    # --------------------------------------------------------
    # Build one fixed geometry mask per trial.
    #
    # Same mask is used for Clear and every Fog
    # severity of that matched trial.
    # --------------------------------------------------------

    masks = {}


    for trial in TRIALS:

        masks[trial] = (
            build_trial_mask(

                trial,
                route,
                carla_map,
                K,
            )
        )


    # --------------------------------------------------------
    # Compute final lane metrics
    # --------------------------------------------------------

    records = []


    for severity in SEVERITIES:

        for trial in TRIALS:

            result = compute_metrics(

                trial,
                severity,
                masks[trial],
            )

            records.append(
                result
            )


    metrics = pd.DataFrame(
        records
    )


    # --------------------------------------------------------
    # Paired R_lane per trial
    #
    # R_lane,j(s) =
    # min(1, (E_j(0)+1)/(E_j(s)+1))
    #
    # E = failure-aware center MAE
    # --------------------------------------------------------

    metrics[
        "R_lane"
    ] = np.nan


    for trial in TRIALS:

        clear_rows = metrics[
            (
                metrics["trial"]
                == trial
            )
            &
            (
                metrics["severity"]
                == 0
            )
        ]


        if len(
            clear_rows
        ) != 1:

            raise RuntimeError(

                f"Trial {trial}: "
                f"missing unique clear baseline."
            )


        clear_error = float(

            clear_rows.iloc[0][
                "failure_aware_center_mae_px"
            ]
        )


        idx = metrics[
            metrics["trial"]
            == trial
        ].index


        for i in idx:

            error = float(

                metrics.loc[
                    i,
                    "failure_aware_center_mae_px",
                ]
            )


            reliability = min(

                1.0,

                (
                    clear_error
                    + EPSILON_PX
                )
                /
                (
                    error
                    + EPSILON_PX
                ),
            )


            metrics.loc[
                i,
                "R_lane",
            ] = reliability


    # --------------------------------------------------------
    # Save per-trial final metrics
    # --------------------------------------------------------

    trial_path = (
        OUT_DIR
        / "ufld_final_per_trial.csv"
    )


    metrics.to_csv(
        trial_path,
        index=False,
    )


    # --------------------------------------------------------
    # Aggregate across 5 controlled configurations
    # --------------------------------------------------------

    summary_rows = []


    for severity in SEVERITIES:

        block = metrics[
            metrics["severity"]
            == severity
        ]


        summary_rows.append({

            "severity":
                severity,

            "n_trials":
                len(block),

            "eligible_frames_mean":
                float(
                    block[
                        "n_eligible"
                    ].mean()
                ),

            "availability_mean_pct":
                float(
                    block[
                        "availability_pct"
                    ].mean()
                ),

            "availability_std_pct":
                float(
                    block[
                        "availability_pct"
                    ].std(
                        ddof=1
                    )
                ),

            "conditional_center_mae_mean_px":
                float(
                    block[
                        "conditional_center_mae_px"
                    ].mean()
                ),

            "conditional_center_mae_std_px":
                float(
                    block[
                        "conditional_center_mae_px"
                    ].std(
                        ddof=1
                    )
                ),

            "failure_aware_center_mae_mean_px":
                float(
                    block[
                        "failure_aware_center_mae_px"
                    ].mean()
                ),

            "failure_aware_center_mae_std_px":
                float(
                    block[
                        "failure_aware_center_mae_px"
                    ].std(
                        ddof=1
                    )
                ),

            "R_lane_mean":
                float(
                    block[
                        "R_lane"
                    ].mean()
                ),

            "R_lane_std":
                float(
                    block[
                        "R_lane"
                    ].std(
                        ddof=1
                    )
                ),
        })


    summary = pd.DataFrame(
        summary_rows
    )


    summary_path = (
        OUT_DIR
        / "ufld_final_summary.csv"
    )


    summary.to_csv(
        summary_path,
        index=False,
    )


    # --------------------------------------------------------
    # Print final table
    # --------------------------------------------------------

    print()

    print(
        "=" * 100
    )

    print(
        "FINAL UFLD-v2 LANE RESULTS"
    )

    print(
        "=" * 100
    )


    display_cols = [

        "severity",

        "availability_mean_pct",

        "conditional_center_mae_mean_px",

        "failure_aware_center_mae_mean_px",

        "R_lane_mean",

        "R_lane_std",
    ]


    print(

        summary[
            display_cols
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Reliability crossing
    # --------------------------------------------------------

    print()

    print(
        "Reliability thresholds:"
    )


    for threshold in [

        0.40,
        0.50,
        0.60,

    ]:

        failed = summary[
            summary[
                "R_lane_mean"
            ]
            < threshold
        ]


        if len(
            failed
        ) == 0:

            print(

                f"  R_lane < "
                f"{threshold:.2f}: "
                f"not reached"
            )

        else:

            first = int(

                failed.iloc[0][
                    "severity"
                ]
            )


            print(

                f"  R_lane < "
                f"{threshold:.2f}: "
                f"Fog {first}"
            )


    print()

    print(
        "Saved:"
    )

    print(
        trial_path
    )

    print(
        summary_path
    )

    print()

    print(
        "PASS: final lane eligibility "
        "and paired R_lane complete."
    )


if __name__ == "__main__":

    main()
