#!/usr/bin/env python3
"""
Audit CARLA lane-marking metadata at the *actual lookahead ground point*
used by the lane metric (image row y=504), not merely at the ego vehicle.

Why:
- UFLD-v2 fails on Clear samples 144-159.
- The lane metric evaluates a future road point that projects to y=504.
- We need to know whether left/right lane markings actually exist at that
  evaluated ground point.

Run from:
    ~/same-camera-different-failures

Requirements:
    CARLA server running on Town10HD_Opt
    conda activate carla
    python lane_marking_lookahead_audit.py

Outputs:
    results/lane_marking_lookahead_audit.csv
"""

import csv
from pathlib import Path

import carla
import numpy as np

from common.projection import build_camera_matrix, world_to_image


HOST = "127.0.0.1"
PORT = 2000

IMAGE_W = 1280
IMAGE_H = 720
FOV = 90.0
LOOKAHEAD_ROW = 504.0
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
UFLD_CSV = ROOT / "results" / "lane_ufld" / "clear_per_frame.csv"
OUT_CSV = ROOT / "results" / "lane_marking_lookahead_audit.csv"


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

    camera_world_location = ego.transform(
        carla.Location(
            x=CAMERA_X,
            y=CAMERA_Y,
            z=CAMERA_Z,
        )
    )

    return carla.Transform(
        camera_world_location,
        carla.Rotation(
            pitch=float(row["ego_pitch"]),
            yaw=float(row["ego_yaw"]),
            roll=float(row["ego_roll"]),
        ),
    )


def enum_name(value):
    # CARLA enums stringify differently across builds.
    s = str(value)
    if "." in s:
        s = s.split(".")[-1]
    return s


def marking_info(marking):
    if marking is None:
        return {
            "type": "MISSING_OBJECT",
            "color": "",
            "width": "",
            "none": True,
        }

    t = enum_name(marking.type)
    color = enum_name(marking.color)

    width = getattr(marking, "width", "")
    try:
        width = float(width)
    except Exception:
        pass

    try:
        is_none = marking.type == carla.LaneMarkingType.NONE
    except Exception:
        is_none = t.upper() == "NONE"

    return {
        "type": t,
        "color": color,
        "width": width,
        "none": bool(is_none),
    }


def find_lookahead_ground_point(
    route,
    route_index,
    current_s,
    world_to_camera,
    K,
):
    """
    Find the route segment whose projected centerline crosses y=504,
    then interpolate the 3D center point and route s at that crossing.
    """

    projected = []

    for j in range(route_index, len(route)):
        rr = route[j]
        s = float(rr["route_s_m"])
        ahead = s - current_s

        if ahead > LOOK_FORWARD_M:
            break

        loc = carla.Location(
            x=float(rr["lane_x"]),
            y=float(rr["lane_y"]),
            z=float(rr["lane_z"]),
        )

        out = world_to_image(
            loc,
            world_to_camera,
            K,
        )

        if out is None:
            continue

        u, v, depth = out

        if not np.all(np.isfinite([u, v, depth])):
            continue

        if depth <= 0:
            continue

        projected.append({
            "route_index": j,
            "s": s,
            "x": loc.x,
            "y": loc.y,
            "z": loc.z,
            "u": float(u),
            "v": float(v),
            "depth": float(depth),
        })

    if len(projected) < 2:
        return None

    candidates = []

    for a, b in zip(projected[:-1], projected[1:]):
        va = a["v"]
        vb = b["v"]

        # Target row lies on or between these projected points.
        if (va - LOOKAHEAD_ROW) * (vb - LOOKAHEAD_ROW) > 0:
            continue

        if abs(vb - va) < 1e-12:
            continue

        t = (LOOKAHEAD_ROW - va) / (vb - va)

        if t < -1e-6 or t > 1.0 + 1e-6:
            continue

        t = min(1.0, max(0.0, t))

        x = a["x"] + t * (b["x"] - a["x"])
        y = a["y"] + t * (b["y"] - a["y"])
        z = a["z"] + t * (b["z"] - a["z"])
        s = a["s"] + t * (b["s"] - a["s"])

        candidates.append({
            "location": carla.Location(x=x, y=y, z=z),
            "lookahead_s": float(s),
            "ahead_m": float(s - current_s),
            "segment_start_s": a["s"],
            "segment_end_s": b["s"],
        })

    if not candidates:
        return None

    # On this route there should be one crossing. If more than one exists,
    # choose the nearest positive lookahead.
    candidates.sort(key=lambda x: x["ahead_m"])
    return candidates[0]


def main():
    route = read_csv(ROUTE_CSV)
    clear = read_csv(CLEAR_CSV)
    ufld = read_csv(UFLD_CSV)

    ufld_by_id = {
        int(float(r["sample_id"])): int(float(r["success"]))
        for r in ufld
    }

    route_s = np.array(
        [float(r["route_s_m"]) for r in route],
        dtype=float,
    )

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

    rows_out = []

    for frame_row in clear:
        if int(float(frame_row["is_junction"])) != 0:
            continue

        gt = frame_row.get("lane_gt_x", "")
        if gt is None or str(gt).strip() == "":
            continue

        sid = int(float(frame_row["sample_id"]))
        current_s = float(frame_row["route_s_m"])

        route_index = int(
            np.argmin(np.abs(route_s - current_s))
        )

        if abs(route_s[route_index] - current_s) > 1e-4:
            raise RuntimeError(
                f"Could not match sample {sid} route_s={current_s}"
            )

        cam_tf = camera_transform_from_row(frame_row)
        world_to_camera = np.array(
            cam_tf.get_inverse_matrix(),
            dtype=np.float64,
        )

        look = find_lookahead_ground_point(
            route,
            route_index,
            current_s,
            world_to_camera,
            K,
        )

        if look is None:
            rows_out.append({
                "sample_id": sid,
                "ego_route_s_m": current_s,
                "ufld_success": ufld_by_id.get(sid, ""),
                "lookahead_route_s_m": "",
                "lookahead_ahead_m": "",
                "lookahead_road_id": "",
                "lookahead_lane_id": "",
                "lookahead_is_junction": "",
                "left_marking_type": "NO_LOOKAHEAD",
                "left_marking_color": "",
                "left_marking_width_m": "",
                "left_marking_none": "",
                "right_marking_type": "NO_LOOKAHEAD",
                "right_marking_color": "",
                "right_marking_width_m": "",
                "right_marking_none": "",
                "either_marking_none": "",
            })
            continue

        wp = carla_map.get_waypoint(
            look["location"],
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )

        if wp is None:
            raise RuntimeError(
                f"No CARLA waypoint at lookahead for sample {sid}"
            )

        left = marking_info(wp.left_lane_marking)
        right = marking_info(wp.right_lane_marking)

        rows_out.append({
            "sample_id": sid,
            "ego_route_s_m": current_s,
            "ufld_success": ufld_by_id.get(sid, ""),
            "lookahead_route_s_m": look["lookahead_s"],
            "lookahead_ahead_m": look["ahead_m"],
            "lookahead_road_id": int(wp.road_id),
            "lookahead_lane_id": int(wp.lane_id),
            "lookahead_is_junction": int(bool(wp.is_junction)),
            "left_marking_type": left["type"],
            "left_marking_color": left["color"],
            "left_marking_width_m": left["width"],
            "left_marking_none": int(left["none"]),
            "right_marking_type": right["type"],
            "right_marking_color": right["color"],
            "right_marking_width_m": right["width"],
            "right_marking_none": int(right["none"]),
            "either_marking_none": int(
                left["none"] or right["none"]
            ),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows_out[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print()
    print("=" * 106)
    print("LOOKAHEAD MARKING AUDIT — transition and UFLD failure region")
    print("=" * 106)
    print(
        f"{'sid':>4} {'ego_s':>7} {'look_s':>8} {'ahead':>7} "
        f"{'UFLD':>5} {'junc':>5} "
        f"{'left_type':>16} {'right_type':>16}"
    )

    for r in rows_out:
        sid = int(r["sample_id"])
        if 135 <= sid <= 159:
            print(
                f"{sid:>4} "
                f"{float(r['ego_route_s_m']):>7.1f} "
                f"{float(r['lookahead_route_s_m']):>8.3f} "
                f"{float(r['lookahead_ahead_m']):>7.3f} "
                f"{str(r['ufld_success']):>5} "
                f"{str(r['lookahead_is_junction']):>5} "
                f"{str(r['left_marking_type']):>16} "
                f"{str(r['right_marking_type']):>16}"
            )

    valid = [
        r for r in rows_out
        if r["lookahead_route_s_m"] != ""
    ]

    failures = [
        r for r in valid
        if str(r["ufld_success"]) == "0"
    ]

    successes = [
        r for r in valid
        if str(r["ufld_success"]) == "1"
    ]

    def count(rows, key, value=1):
        return sum(
            1 for r in rows
            if str(r[key]) == str(value)
        )

    print()
    print("=" * 106)
    print("SUMMARY")
    print("=" * 106)
    print("Evaluable Clear frames:", len(valid))
    print("Lookahead in junction:", count(valid, "lookahead_is_junction"))
    print("Either lookahead marking NONE:", count(valid, "either_marking_none"))
    print()
    print("UFLD successes:", len(successes))
    print(
        "  lookahead in junction:",
        count(successes, "lookahead_is_junction"),
    )
    print(
        "  either marking NONE:",
        count(successes, "either_marking_none"),
    )
    print()
    print("UFLD failures:", len(failures))
    print(
        "  lookahead in junction:",
        count(failures, "lookahead_is_junction"),
    )
    print(
        "  either marking NONE:",
        count(failures, "either_marking_none"),
    )
    print()
    print("Saved:", OUT_CSV)


if __name__ == "__main__":
    main()
