#!/usr/bin/env python3

import sys
from pathlib import Path

import carla
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.projection import (
    build_camera_matrix,
    world_to_image,
    interpolate_x_at_row,
)


# ============================================================
# FROZEN SETTINGS
# ============================================================

ROUTE_PATH = (
    ROOT
    / "data"
    / "routes"
    / "stop3_route.csv"
)

IMAGE_W = 1280
IMAGE_H = 720
FOV_DEG = 90.0

CAMERA_X = 1.5
CAMERA_Y = 0.0
CAMERA_Z = 1.7

PRIMARY_ROW = 504.0

TARGET_DISTANCES_M = [
    25.0,
    30.0,
    35.0,
    40.0,
]

TRIALS = [1, 2, 3, 4, 5]

LOOK_FORWARD_M = 60.0

# UFLD CULane row-anchor domain already verified.
UFLD_MIN_Y = 302.4
UFLD_MAX_Y = 720.0

# PRE-REGISTERED BEFORE UFLD RANGE RESULTS.
MIN_ELIGIBLE_FRAMES_PER_TRIAL = 50

# Secondary range-analysis epsilon.
# Primary y=504 result itself remains unchanged:
# epsilon = 1 pixel.
EPSILON_M = 0.0139


K = build_camera_matrix(
    IMAGE_W,
    IMAGE_H,
    FOV_DEG,
)

FX = float(K[0, 0])


# ============================================================
# ROUTE
# ============================================================

route = pd.read_csv(
    ROUTE_PATH
)

needed_route = {
    "route_s_m",
    "lane_x",
    "lane_y",
    "lane_z",
    "is_junction",
}

missing = (
    needed_route
    - set(route.columns)
)

if missing:
    raise RuntimeError(
        f"Route missing columns: {sorted(missing)}"
    )

route = (
    route
    .sort_values("route_s_m")
    .reset_index(drop=True)
)


# ============================================================
# CAMERA
# ============================================================

def camera_transform(frame):

    ego = carla.Transform(
        carla.Location(
            x=float(frame["ego_x"]),
            y=float(frame["ego_y"]),
            z=float(frame["ego_z"]),
        ),
        carla.Rotation(
            pitch=float(frame["ego_pitch"]),
            yaw=float(frame["ego_yaw"]),
            roll=float(frame["ego_roll"]),
        ),
    )

    camera_location = ego.transform(
        carla.Location(
            x=CAMERA_X,
            y=CAMERA_Y,
            z=CAMERA_Z,
        )
    )

    return carla.Transform(
        camera_location,
        carla.Rotation(
            pitch=float(frame["ego_pitch"]),
            yaw=float(frame["ego_yaw"]),
            roll=float(frame["ego_roll"]),
        ),
    )


def xyz_from_route_row(row):

    return np.array(
        [
            float(row["lane_x"]),
            float(row["lane_y"]),
            float(row["lane_z"]),
        ],
        dtype=float,
    )


def carla_location_from_xyz(xyz):

    return carla.Location(
        x=float(xyz[0]),
        y=float(xyz[1]),
        z=float(xyz[2]),
    )


# ============================================================
# PROJECT FUTURE LANE CENTER
# ============================================================

def project_future_route(frame):

    current_s = float(
        frame["route_s_m"]
    )

    cam = camera_transform(frame)

    camera_xyz = np.array(
        [
            cam.location.x,
            cam.location.y,
            cam.location.z,
        ],
        dtype=float,
    )

    world_to_camera = np.asarray(
        cam.get_inverse_matrix(),
        dtype=float,
    )

    future = route[
        (
            route["route_s_m"]
            >= current_s
        )
        &
        (
            route["route_s_m"]
            <= current_s + LOOK_FORWARD_M
        )
    ]

    projected = []

    for _, rr in future.iterrows():

        xyz = xyz_from_route_row(rr)

        result = world_to_image(
            carla_location_from_xyz(xyz),
            world_to_camera,
            K,
        )

        if result is None:
            continue

        u, v, depth = result

        if not (
            np.isfinite(u)
            and np.isfinite(v)
            and np.isfinite(depth)
            and depth > 0.0
        ):
            continue

        camera_range = float(
            np.linalg.norm(
                xyz - camera_xyz
            )
        )

        projected.append(
            {
                "xyz": xyz,
                "u": float(u),
                "v": float(v),
                "depth": float(depth),
                "camera_range_m":
                    camera_range,
                "route_forward_m":
                    float(
                        rr["route_s_m"]
                        - current_s
                    ),
                "is_junction":
                    int(rr["is_junction"]),
            }
        )

    return (
        projected,
        camera_xyz,
        world_to_camera,
    )


# ============================================================
# EXACT TARGET CAMERA RANGE -> IMAGE ROW
# ============================================================

def row_at_camera_range(
    frame,
    target_range_m,
):

    current_s = float(
        frame["route_s_m"]
    )

    cam = camera_transform(frame)

    camera_xyz = np.array(
        [
            cam.location.x,
            cam.location.y,
            cam.location.z,
        ],
        dtype=float,
    )

    world_to_camera = np.asarray(
        cam.get_inverse_matrix(),
        dtype=float,
    )

    future = route[
        (
            route["route_s_m"]
            >= current_s
        )
        &
        (
            route["route_s_m"]
            <= current_s + LOOK_FORWARD_M
        )
    ].reset_index(drop=True)

    if len(future) < 2:
        return None

    for i in range(
        len(future) - 1
    ):

        p0 = xyz_from_route_row(
            future.iloc[i]
        )

        p1 = xyz_from_route_row(
            future.iloc[i + 1]
        )

        segment = p1 - p0
        q = p0 - camera_xyz

        a = float(
            np.dot(
                segment,
                segment,
            )
        )

        b = float(
            2.0
            * np.dot(
                q,
                segment,
            )
        )

        c = float(
            np.dot(q, q)
            - target_range_m**2
        )

        if a <= 0.0:
            continue

        disc = (
            b * b
            - 4.0 * a * c
        )

        if disc < 0.0:
            continue

        root = np.sqrt(
            max(
                0.0,
                disc,
            )
        )

        candidates = [
            (-b - root) / (2.0 * a),
            (-b + root) / (2.0 * a),
        ]

        for t in sorted(candidates):

            if not (
                0.0 <= t <= 1.0
            ):
                continue

            xyz = (
                p0
                + t * segment
            )

            result = world_to_image(
                carla_location_from_xyz(xyz),
                world_to_camera,
                K,
            )

            if result is None:
                continue

            u, v, depth = result

            if not (
                np.isfinite(u)
                and np.isfinite(v)
                and np.isfinite(depth)
                and depth > 0.0
            ):
                continue

            if not (
                0.0 <= u < IMAGE_W
                and 0.0 <= v < IMAGE_H
            ):
                continue

            # Check exact physical distance.
            actual_range = float(
                np.linalg.norm(
                    xyz
                    - camera_xyz
                )
            )

            if abs(
                actual_range
                - target_range_m
            ) > 1e-5:
                raise RuntimeError(
                    "Camera-range intersection "
                    "calculation failed."
                )

            return float(v)

    return None


# ============================================================
# ROW -> PHYSICAL CAMERA RANGE
# ============================================================

def crossing_at_row(
    projected,
    camera_xyz,
    target_y,
):

    candidates = []

    for a, b in zip(
        projected[:-1],
        projected[1:],
    ):

        v1 = float(a["v"])
        v2 = float(b["v"])

        if abs(
            v2 - v1
        ) < 1e-12:
            continue

        if not (
            min(v1, v2)
            <= target_y
            <= max(v1, v2)
        ):
            continue

        t = (
            (target_y - v1)
            / (v2 - v1)
        )

        if not (
            0.0 <= t <= 1.0
        ):
            continue

        xyz = (
            a["xyz"]
            + t
            * (
                b["xyz"]
                - a["xyz"]
            )
        )

        u = (
            float(a["u"])
            + t
            * (
                float(b["u"])
                - float(a["u"])
            )
        )

        depth = (
            float(a["depth"])
            + t
            * (
                float(b["depth"])
                - float(a["depth"])
            )
        )

        route_forward = (
            float(a["route_forward_m"])
            + t
            * (
                float(b["route_forward_m"])
                - float(a["route_forward_m"])
            )
        )

        camera_range = float(
            np.linalg.norm(
                xyz
                - camera_xyz
            )
        )

        eligible = (
            int(a["is_junction"]) == 0
            and int(b["is_junction"]) == 0
            and 0.0 <= u < IMAGE_W
        )

        candidates.append(
            {
                "route_forward_m":
                    route_forward,
                "camera_range_m":
                    camera_range,
                "depth_m":
                    depth,
                "m_per_px":
                    depth / FX,
                "eligible":
                    bool(eligible),
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x:
            x["route_forward_m"]
    )

    return candidates[0]


# ============================================================
# LOAD CLEAR TRIALS
# ============================================================

clear_trials = {}

for trial in TRIALS:

    path = (
        ROOT
        / "data"
        / "recordings"
        / f"stop3_clear_0_drift_0p75_trial_{trial}"
        / "frames.csv"
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing: {path}"
        )

    df = pd.read_csv(path)

    needed = {
        "sample_id",
        "route_s_m",
        "ego_x",
        "ego_y",
        "ego_z",
        "ego_pitch",
        "ego_yaw",
        "ego_roll",
        "is_junction",
        "lane_gt_x",
    }

    missing = (
        needed
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            f"{path} missing: "
            f"{sorted(missing)}"
        )

    clear_trials[trial] = df


# ============================================================
# 1. REPROJECTION SELF-CHECK
# ============================================================

reprojection_errors = []

for trial, df in clear_trials.items():

    for _, frame in df.iterrows():

        saved = frame["lane_gt_x"]

        if pd.isna(saved):
            continue

        projected, _, _ = (
            project_future_route(
                frame
            )
        )

        points = [
            (p["u"], p["v"])
            for p in projected
        ]

        recomputed = (
            interpolate_x_at_row(
                points,
                PRIMARY_ROW,
            )
        )

        if recomputed is None:
            continue

        reprojection_errors.append(
            abs(
                float(recomputed)
                - float(saved)
            )
        )


reprojection_errors = np.asarray(
    reprojection_errors,
    dtype=float,
)

print("=" * 92)
print(
    "CORRECTED GEOMETRY-ONLY LANE RANGE PREREGISTRATION"
)
print("=" * 92)

print()
print(
    "Distance definition: "
    "3D Euclidean CAMERA -> lane-center point"
)

print(
    "This matches the physical-distance "
    "convention used for STOP-sign range."
)

print()
print("PRIMARY GT REPROJECTION CHECK")

print(
    "mean abs error:",
    f"{reprojection_errors.mean():.6f} px",
)

print(
    "max abs error: ",
    f"{reprojection_errors.max():.6f} px",
)

if (
    len(reprojection_errors) == 0
    or reprojection_errors.max() > 0.05
):
    raise RuntimeError(
        "Reprojection check failed."
    )

print(
    "PASS: camera reconstruction "
    "matches saved lane GT."
)


# ============================================================
# 2. CAMERA RANGE -> ROW
# ============================================================

row_measurements = {
    d: {
        trial: []
        for trial in TRIALS
    }
    for d in TARGET_DISTANCES_M
}


for trial, df in clear_trials.items():

    for _, frame in df.iterrows():

        if int(
            frame["is_junction"]
        ) != 0:
            continue

        for d in TARGET_DISTANCES_M:

            y = row_at_camera_range(
                frame,
                d,
            )

            if y is None:
                continue

            row_measurements[
                d
            ][trial].append(y)


frozen_rows = {}


print()
print("=" * 92)
print(
    "CAMERA DISTANCE -> IMAGE ROW"
)
print("=" * 92)


for d in TARGET_DISTANCES_M:

    combined = []

    print()
    print(
        f"TARGET CAMERA RANGE = {d:.0f} m"
    )

    for trial in TRIALS:

        vals = np.asarray(
            row_measurements[
                d
            ][trial],
            dtype=float,
        )

        combined.extend(
            vals.tolist()
        )

        if len(vals) == 0:

            print(
                f"  trial {trial}: N=0"
            )

        else:

            print(
                f"  trial {trial}: "
                f"N={len(vals):3d}, "
                f"median y="
                f"{np.median(vals):.3f}, "
                f"std="
                f"{np.std(vals,ddof=1):.3f}"
            )

    combined = np.asarray(
        combined,
        dtype=float,
    )

    if len(combined) == 0:
        raise RuntimeError(
            f"No mapping for {d} m"
        )

    median_y = float(
        np.median(combined)
    )

    frozen_y = int(
        np.rint(median_y)
    )

    frozen_rows[d] = frozen_y

    print(
        f"  OVERALL median y="
        f"{median_y:.3f}"
    )

    print(
        f"  FROZEN ROW: y={frozen_y}"
    )

    supported = (
        UFLD_MIN_Y
        <= frozen_y
        <= UFLD_MAX_Y
    )

    print(
        "  UFLD support:",
        supported,
    )

    if not supported:
        raise RuntimeError(
            f"{d} m outside UFLD domain."
        )


# ============================================================
# 3. ROW -> ACTUAL CAMERA RANGE + ELIGIBILITY
# ============================================================

rows_to_check = {
    "primary_504":
        int(PRIMARY_ROW),
}

for d, y in frozen_rows.items():

    rows_to_check[
        f"target_{int(d)}m"
    ] = y


summary_rows = []


print()
print("=" * 92)
print(
    "FROZEN ROW -> ACTUAL CAMERA RANGE"
)
print("=" * 92)


for label, target_y in rows_to_check.items():

    all_ranges = []
    all_scales = []
    counts = []

    print()
    print(
        f"{label}: y={target_y}"
    )

    for trial, df in clear_trials.items():

        trial_ranges = []
        trial_scales = []

        for _, frame in df.iterrows():

            if int(
                frame["is_junction"]
            ) != 0:
                continue

            (
                projected,
                camera_xyz,
                _,
            ) = project_future_route(
                frame
            )

            crossing = crossing_at_row(
                projected,
                camera_xyz,
                float(target_y),
            )

            if crossing is None:
                continue

            if not crossing[
                "eligible"
            ]:
                continue

            trial_ranges.append(
                crossing[
                    "camera_range_m"
                ]
            )

            trial_scales.append(
                crossing[
                    "m_per_px"
                ]
            )

        trial_ranges = np.asarray(
            trial_ranges,
            dtype=float,
        )

        trial_scales = np.asarray(
            trial_scales,
            dtype=float,
        )

        counts.append(
            len(trial_ranges)
        )

        all_ranges.extend(
            trial_ranges.tolist()
        )

        all_scales.extend(
            trial_scales.tolist()
        )

        if len(trial_ranges) == 0:

            print(
                f"  trial {trial}: "
                "eligible N=0"
            )

        else:

            print(
                f"  trial {trial}: "
                f"eligible N="
                f"{len(trial_ranges):3d}, "
                f"median range="
                f"{np.median(trial_ranges):.3f} m"
            )

    all_ranges = np.asarray(
        all_ranges,
        dtype=float,
    )

    all_scales = np.asarray(
        all_scales,
        dtype=float,
    )

    min_count = int(
        min(counts)
    )

    keep = (
        min_count
        >= MIN_ELIGIBLE_FRAMES_PER_TRIAL
    )

    median_range = (
        float(
            np.median(all_ranges)
        )
        if len(all_ranges)
        else np.nan
    )

    median_scale = (
        float(
            np.median(all_scales)
        )
        if len(all_scales)
        else np.nan
    )

    print(
        "  minimum eligible count:",
        min_count,
    )

    print(
        "  status:",
        (
            "KEEP"
            if keep
            else
            "INSUFFICIENT COVERAGE"
        ),
    )

    print(
        "  overall median camera range:",
        f"{median_range:.3f} m",
    )

    print(
        "  geometric lateral scale:",
        f"{median_scale:.6f} m/px",
    )

    nominal = np.nan

    if label.startswith(
        "target_"
    ):

        nominal = float(
            label
            .replace("target_", "")
            .replace("m", "")
        )

    summary_rows.append(
        {
            "label":
                label,

            "nominal_target_m":
                nominal,

            "distance_definition":
                "camera_to_lane_3d_euclidean",

            "frozen_row_y":
                target_y,

            "min_eligible_frames_per_trial":
                min_count,

            "minimum_required":
                MIN_ELIGIBLE_FRAMES_PER_TRIAL,

            "status":
                (
                    "KEEP"
                    if keep
                    else
                    "INSUFFICIENT_COVERAGE"
                ),

            "actual_median_camera_range_m":
                median_range,

            "median_m_per_px":
                median_scale,
        }
    )


# ============================================================
# SAVE / OVERWRITE OBSOLETE PREREGISTRATION
# ============================================================

OUT = (
    ROOT
    / "results"
    / "final"
    / "lane"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

summary = pd.DataFrame(
    summary_rows
)

csv_path = (
    OUT
    / "range_extension_preregister.csv"
)

txt_path = (
    OUT
    / "range_extension_preregister.txt"
)

summary.to_csv(
    csv_path,
    index=False,
)


with txt_path.open("w") as f:

    f.write(
        "CORRECTED LANE RANGE "
        "EXTENSION PREREGISTRATION\n\n"
    )

    f.write(
        "Distance definition: "
        "3D Euclidean distance from RGB camera "
        "to CARLA lane-center point.\n"
    )

    f.write(
        "This matches the physical range "
        "definition used for STOP signs.\n"
    )

    f.write(
        "The earlier route-distance mapping "
        "was superseded before any UFLD "
        "range-extension outcomes were inspected.\n\n"
    )

    f.write(
        "Targets: 25, 30, 35, 40 m\n"
    )

    f.write(
        "Minimum eligibility: "
        "50 frames in every controlled configuration\n"
    )

    f.write(
        "Primary y=504 result remains unchanged "
        "with epsilon=1 px.\n"
    )

    f.write(
        "Secondary physical epsilon: "
        "0.0139 m.\n\n"
    )

    f.write(
        summary.to_string(
            index=False
        )
    )

    f.write("\n")


print()
print("=" * 92)
print(
    "CORRECTED PREREGISTRATION SAVED"
)
print("=" * 92)

print(
    summary.to_string(
        index=False
    )
)

print()
print("Saved:")
print(csv_path)
print(txt_path)

print()
print(
    "No UFLD range outcomes have been used "
    "to choose these rows."
)
