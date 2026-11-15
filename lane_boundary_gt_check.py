#!/usr/bin/env python3
"""
Project CARLA lane-edge ground truth to the existing lookahead row y=504.

Purpose:
- Resolve whether LSD (~390 px pair width) or UFLD-v2 (~250 px pair width)
  matches the simulator lane geometry.
- Uses CARLA map waypoint.lane_width as physical ground truth.
- Uses the same frozen route and the same camera model as collect_data.py.
- Does NOT change any detector or weather result.

Run from:
    ~/same-camera-different-failures

Requirements:
    1) CARLA server running with Town10HD_Opt
    2) conda activate carla
    3) python lane_boundary_gt_check.py
"""

import csv
import math
from pathlib import Path

import carla
import numpy as np

from common.projection import (
    build_camera_matrix,
    world_to_image,
    interpolate_x_at_row,
)

HOST = "127.0.0.1"
PORT = 2000

IMAGE_W = 1280
IMAGE_H = 720
FOV = 90.0
LOOKAHEAD_ROW = 504
LOOK_FORWARD_M = 60.0

CAMERA_X = 1.5
CAMERA_Y = 0.0
CAMERA_Z = 1.7

ROOT = Path(__file__).resolve().parent
ROUTE_CSV = ROOT / "data" / "routes" / "stop3_route.csv"
CLEAR_CSV = (
    ROOT
    / "data"
    / "recordings"
    / "stop3_clear_0_drift_0p75"
    / "frames.csv"
)
OUT_CSV = ROOT / "results" / "lane_boundary_gt_clear.csv"


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def camera_transform_from_row(row):
    ego = carla.Transform(
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

    cam_world_location = ego.transform(
        carla.Location(
            x=CAMERA_X,
            y=CAMERA_Y,
            z=CAMERA_Z,
        )
    )

    # Relative camera rotation is exactly zero in collect_data.py.
    return carla.Transform(
        cam_world_location,
        carla.Rotation(
            pitch=float(row["ego_pitch"]),
            yaw=float(row["ego_yaw"]),
            roll=float(row["ego_roll"]),
        ),
    )


def interp_scalar_at_row(samples, target_row):
    """
    samples = [(value, image_y), ...]
    """
    if len(samples) < 2:
        return None

    arr = sorted(samples, key=lambda t: t[1])
    ys = np.array([x[1] for x in arr], dtype=float)
    vals = np.array([x[0] for x in arr], dtype=float)

    unique_y, idx = np.unique(ys, return_index=True)
    vals = vals[idx]

    if len(unique_y) < 2:
        return None

    if target_row < unique_y.min() or target_row > unique_y.max():
        return None

    return float(np.interp(target_row, unique_y, vals))


def main():
    route = read_csv(ROUTE_CSV)
    clear = read_csv(CLEAR_CSV)

    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    world = client.get_world()
    carla_map = world.get_map()

    print("Connected to:", carla_map.name)

    K = build_camera_matrix(
        IMAGE_W,
        IMAGE_H,
        FOV,
    )

    route_s = np.array(
        [float(r["route_s_m"]) for r in route],
        dtype=float,
    )

    output = []

    for frame_row in clear:
        if int(float(frame_row["is_junction"])) != 0:
            continue

        gt_text = frame_row.get("lane_gt_x", "")
        if gt_text == "":
            continue

        current_s = float(frame_row["route_s_m"])
        route_index = int(np.argmin(np.abs(route_s - current_s)))

        if abs(route_s[route_index] - current_s) > 1e-4:
            raise RuntimeError(
                f"Could not match route_s={current_s} to frozen route."
            )

        cam_tf = camera_transform_from_row(frame_row)
        world_to_camera = np.array(
            cam_tf.get_inverse_matrix(),
            dtype=np.float64,
        )

        center_proj = []
        left_proj = []
        right_proj = []
        width_m_samples = []

        for j in range(route_index, len(route)):
            rr = route[j]
            forward_s = float(rr["route_s_m"]) - current_s

            if forward_s > LOOK_FORWARD_M:
                break

            center = carla.Location(
                x=float(rr["lane_x"]),
                y=float(rr["lane_y"]),
                z=float(rr["lane_z"]),
            )

            wp = carla_map.get_waypoint(
                center,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if wp is None:
                continue

            # Reject an accidental snap to a different road/lane.
            if (
                int(wp.road_id) != int(rr["road_id"])
                or int(wp.lane_id) != int(rr["lane_id"])
            ):
                continue

            lane_width_m = float(wp.lane_width)
            half = 0.5 * lane_width_m
            rv = wp.transform.get_right_vector()

            edge_a = carla.Location(
                x=center.x - half * rv.x,
                y=center.y - half * rv.y,
                z=center.z - half * rv.z,
            )
            edge_b = carla.Location(
                x=center.x + half * rv.x,
                y=center.y + half * rv.y,
                z=center.z + half * rv.z,
            )

            pc = world_to_image(center, world_to_camera, K)
            pa = world_to_image(edge_a, world_to_camera, K)
            pb = world_to_image(edge_b, world_to_camera, K)

            if pc is not None:
                uc, vc, dc = pc
                if np.all(np.isfinite([uc, vc, dc])):
                    center_proj.append((float(uc), float(vc)))
                    width_m_samples.append((lane_width_m, float(vc)))

            if pa is not None:
                ua, va, da = pa
                if np.all(np.isfinite([ua, va, da])):
                    left_proj.append((float(ua), float(va)))

            if pb is not None:
                ub, vb, db = pb
                if np.all(np.isfinite([ub, vb, db])):
                    right_proj.append((float(ub), float(vb)))

        center_x = interpolate_x_at_row(
            center_proj,
            LOOKAHEAD_ROW,
        )
        edge_a_x = interpolate_x_at_row(
            left_proj,
            LOOKAHEAD_ROW,
        )
        edge_b_x = interpolate_x_at_row(
            right_proj,
            LOOKAHEAD_ROW,
        )
        lane_width_m_at_row = interp_scalar_at_row(
            width_m_samples,
            LOOKAHEAD_ROW,
        )

        if center_x is None or edge_a_x is None or edge_b_x is None:
            continue

        image_left_x = min(edge_a_x, edge_b_x)
        image_right_x = max(edge_a_x, edge_b_x)
        width_px = image_right_x - image_left_x
        edge_mid_x = 0.5 * (image_left_x + image_right_x)

        saved_center_x = float(gt_text)

        output.append({
            "sample_id": int(float(frame_row["sample_id"])),
            "route_s_m": current_s,
            "saved_lane_gt_x": saved_center_x,
            "reprojected_center_x": center_x,
            "center_reprojection_error_px": abs(center_x - saved_center_x),
            "gt_left_edge_x": image_left_x,
            "gt_right_edge_x": image_right_x,
            "gt_edge_mid_x": edge_mid_x,
            "edge_mid_vs_center_px": edge_mid_x - center_x,
            "gt_lane_width_px": width_px,
            "carla_lane_width_m": lane_width_m_at_row,
        })

    if not output:
        raise RuntimeError("No valid boundary projections produced.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(output[0].keys()),
        )
        writer.writeheader()
        writer.writerows(output)

    width_px = np.array(
        [r["gt_lane_width_px"] for r in output],
        dtype=float,
    )
    width_m = np.array(
        [r["carla_lane_width_m"] for r in output],
        dtype=float,
    )
    reproj = np.array(
        [r["center_reprojection_error_px"] for r in output],
        dtype=float,
    )
    edge_mid_delta = np.array(
        [r["edge_mid_vs_center_px"] for r in output],
        dtype=float,
    )

    print()
    print("=" * 68)
    print("CARLA LANE-EDGE GROUND TRUTH @ y=504")
    print("=" * 68)
    print("Frames projected:", len(output))
    print()
    print("PHYSICAL LANE WIDTH")
    print(
        "mean / median / std: "
        f"{width_m.mean():.4f} / "
        f"{np.median(width_m):.4f} / "
        f"{width_m.std():.4f} m"
    )
    print()
    print("PROJECTED TRUE LANE WIDTH")
    print(
        "mean / median / std: "
        f"{width_px.mean():.3f} / "
        f"{np.median(width_px):.3f} / "
        f"{width_px.std():.3f} px"
    )
    print(
        "min / max: "
        f"{width_px.min():.3f} / "
        f"{width_px.max():.3f} px"
    )
    print()
    print("CENTER REPROJECTION CHECK")
    print(
        "mean / max |reprojected center - saved lane_gt_x|: "
        f"{reproj.mean():.6f} / {reproj.max():.6f} px"
    )
    print()
    print("EDGE-MIDPOINT VS CENTERLINE")
    print(
        "mean signed / mean absolute: "
        f"{edge_mid_delta.mean():+.6f} / "
        f"{np.abs(edge_mid_delta).mean():.6f} px"
    )
    print()
    print("REFERENCE DETECTOR WIDTHS")
    print("UFLD-v2 successful Clear mean: ~250.218 px")
    print("LSD successful Clear mean:     ~390.398 px")
    print()
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
