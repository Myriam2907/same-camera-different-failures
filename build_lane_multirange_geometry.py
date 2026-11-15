#!/usr/bin/env python3
"""
Build row-specific CARLA lane geometry for matched-distance analysis.

Run from the project root in the `carla` conda environment while CARLA is running.

Creates:
  results/pilot/lane_multirange_geometry.csv
  results/pilot/lane_multirange_geometry_summary.csv

Rows evaluated: 504, 480, 460, 440, 420.

For each sample/row this records:
- projected lane-center GT x
- 3-D Euclidean camera -> road target distance
- CARLA lane width and marking/junction eligibility
- local road plane/right-vector geometry
- camera-to-world rotation/origin needed for exact pixel-ray back-projection

The script also checks row 504 against the already-recorded lane_gt_x.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import carla

ROOT = Path(__file__).resolve().parent
ROUTE_CSV = ROOT / "data" / "routes" / "stop3_route.csv"
CLEAR_CSV = (
    ROOT / "data" / "recordings" / "stop3_clear_0_drift_0p75" / "frames.csv"
)
OUT_DIR = ROOT / "results" / "pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUT_DIR / "lane_multirange_geometry.csv"
SUMMARY_CSV = OUT_DIR / "lane_multirange_geometry_summary.csv"

ROWS_Y = [504, 480, 460, 440, 420]
IMAGE_W = 1280
IMAGE_H = 720
FOV_DEG = 90.0
FX = IMAGE_W / (2.0 * math.tan(math.radians(FOV_DEG) / 2.0))
FY = FX
CX = IMAGE_W / 2.0
CY = IMAGE_H / 2.0

CAMERA_X = 1.5
CAMERA_Y = 0.0
CAMERA_Z = 1.7
LOOK_FORWARD_M = 60.0

def project_world_xyz(xyz, world_to_camera):
    p = np.array([xyz[0], xyz[1], xyz[2], 1.0], dtype=float)
    q = world_to_camera @ p

    forward = float(q[0])
    right = float(q[1])
    up = float(q[2])

    if forward <= 1e-9:
        return None

    u = FX * (right / forward) + CX
    v = FY * (-up / forward) + CY

    if not (np.isfinite(u) and np.isfinite(v)):
        return None

    return float(u), float(v), forward

def camera_matrix_from_recorded_pose(row):
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

    ego_m = np.array(ego.get_matrix(), dtype=float)
    cam_local = np.array([CAMERA_X, CAMERA_Y, CAMERA_Z, 1.0], dtype=float)
    cam_world = ego_m @ cam_local

    cam_tf = carla.Transform(
        carla.Location(
            x=float(cam_world[0]),
            y=float(cam_world[1]),
            z=float(cam_world[2]),
        ),
        ego.rotation,
    )

    c2w = np.array(cam_tf.get_matrix(), dtype=float)
    w2c = np.array(cam_tf.get_inverse_matrix(), dtype=float)
    return cam_tf, c2w, w2c

def find_row_crossing(route, start_idx, current_s, target_y, w2c):
    projected = []

    for idx in range(start_idx, len(route)):
        rr = route.iloc[idx]
        route_s = float(rr["route_s_m"])
        if route_s - current_s > LOOK_FORWARD_M:
            break

        xyz = np.array(
            [float(rr["lane_x"]), float(rr["lane_y"]), float(rr["lane_z"])],
            dtype=float,
        )
        uv = project_world_xyz(xyz, w2c)
        if uv is None:
            continue

        u, v, depth = uv
        projected.append(
            {
                "idx": idx,
                "s": route_s,
                "xyz": xyz,
                "u": u,
                "v": v,
                "depth": depth,
            }
        )

    candidates = []
    for a, b in zip(projected[:-1], projected[1:]):
        va, vb = a["v"], b["v"]
        if abs(vb - va) < 1e-12:
            continue

        if (target_y - va) * (target_y - vb) > 0:
            continue

        t = (target_y - va) / (vb - va)
        if t < -1e-9 or t > 1.0 + 1e-9:
            continue

        t = float(np.clip(t, 0.0, 1.0))
        xyz = a["xyz"] + t * (b["xyz"] - a["xyz"])
        route_s = a["s"] + t * (b["s"] - a["s"])
        route_ahead = route_s - current_s

        if route_ahead < -1e-6:
            continue

        uv = project_world_xyz(xyz, w2c)
        if uv is None:
            continue

        candidates.append(
            {
                "xyz": xyz,
                "route_s": float(route_s),
                "route_ahead_m": float(route_ahead),
                "u": float(uv[0]),
                "v": float(uv[1]),
                "depth": float(uv[2]),
            }
        )

    if not candidates:
        return None

    # The frozen route should produce one physically relevant crossing.
    candidates.sort(key=lambda z: z["route_ahead_m"])
    return candidates[0]

def marking_name(marking):
    try:
        return str(marking.type).split(".")[-1]
    except Exception:
        return "UNKNOWN"

def is_none_marking(marking):
    name = marking_name(marking).upper()
    return name == "NONE" or name.endswith("NONE")

def vec3(v):
    return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)

def main():
    if not ROUTE_CSV.exists():
        raise RuntimeError(f"Missing frozen route: {ROUTE_CSV}")
    if not CLEAR_CSV.exists():
        raise RuntimeError(f"Missing Clear metadata: {CLEAR_CSV}")

    route = pd.read_csv(ROUTE_CSV)
    meta = pd.read_csv(CLEAR_CSV)

    needed_route = {"route_s_m", "lane_x", "lane_y", "lane_z"}
    needed_meta = {
        "sample_id", "route_s_m",
        "ego_x", "ego_y", "ego_z",
        "ego_pitch", "ego_yaw", "ego_roll",
    }
    if not needed_route.issubset(route.columns):
        raise RuntimeError(
            f"Route missing columns: {sorted(needed_route - set(route.columns))}"
        )
    if not needed_meta.issubset(meta.columns):
        raise RuntimeError(
            f"frames.csv missing columns: {sorted(needed_meta - set(meta.columns))}"
        )

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    print("=" * 88)
    print("LANE MULTI-RANGE GEOMETRY")
    print("=" * 88)
    print("Map:", carla_map.name)
    print("Rows:", ROWS_Y)

    route_s_values = route["route_s_m"].to_numpy(dtype=float)
    records = []

    for _, mr in meta.iterrows():
        sid = int(mr["sample_id"])
        current_s = float(mr["route_s_m"])

        start_idx = int(np.argmin(np.abs(route_s_values - current_s)))

        cam_tf, c2w, w2c = camera_matrix_from_recorded_pose(mr)
        cam_origin = np.array(
            [
                float(cam_tf.location.x),
                float(cam_tf.location.y),
                float(cam_tf.location.z),
            ],
            dtype=float,
        )

        for row_y in ROWS_Y:
            cross = find_row_crossing(
                route, start_idx, current_s, float(row_y), w2c
            )

            if cross is None:
                records.append(
                    {
                        "sample_id": sid,
                        "route_s_m": current_s,
                        "row_y": row_y,
                        "geometry_valid": 0,
                        "eligible": 0,
                    }
                )
                continue

            center = cross["xyz"]
            wp = carla_map.get_waypoint(
                carla.Location(
                    x=float(center[0]),
                    y=float(center[1]),
                    z=float(center[2]),
                ),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )

            if wp is None:
                records.append(
                    {
                        "sample_id": sid,
                        "route_s_m": current_s,
                        "row_y": row_y,
                        "geometry_valid": 0,
                        "eligible": 0,
                    }
                )
                continue

            right = vec3(wp.transform.get_right_vector())
            right /= max(np.linalg.norm(right), 1e-12)

            up = vec3(wp.transform.get_up_vector())
            up /= max(np.linalg.norm(up), 1e-12)

            lane_width = float(wp.lane_width)
            left_name = marking_name(wp.left_lane_marking)
            right_name = marking_name(wp.right_lane_marking)

            eligible = (
                (not bool(wp.is_junction))
                and (not is_none_marking(wp.left_lane_marking))
                and (not is_none_marking(wp.right_lane_marking))
            )

            # GT lane boundaries from CARLA lane width at the target cross-section.
            left_world = center - 0.5 * lane_width * right
            right_world = center + 0.5 * lane_width * right

            center_uv = project_world_xyz(center, w2c)
            left_uv = project_world_xyz(left_world, w2c)
            right_uv = project_world_xyz(right_world, w2c)

            if center_uv is None or left_uv is None or right_uv is None:
                geometry_valid = 0
                eligible = False
                gt_x = np.nan
                gt_left_x = np.nan
                gt_right_x = np.nan
            else:
                geometry_valid = 1
                gt_x = float(center_uv[0])
                gt_left_x = float(left_uv[0])
                gt_right_x = float(right_uv[0])

            distance_3d = float(np.linalg.norm(center - cam_origin))

            rec = {
                "sample_id": sid,
                "route_s_m": current_s,
                "row_y": row_y,
                "geometry_valid": geometry_valid,
                "eligible": int(bool(eligible)),
                "target_route_s_m": cross["route_s"],
                "route_ahead_m": cross["route_ahead_m"],
                "camera_to_ground_3d_m": distance_3d,
                "gt_center_x_px": gt_x,
                "gt_left_x_px": gt_left_x,
                "gt_right_x_px": gt_right_x,
                "carla_lane_width_m": lane_width,
                "target_is_junction": int(bool(wp.is_junction)),
                "left_marking": left_name,
                "right_marking": right_name,
                "center_x_world": float(center[0]),
                "center_y_world": float(center[1]),
                "center_z_world": float(center[2]),
                "road_right_x": float(right[0]),
                "road_right_y": float(right[1]),
                "road_right_z": float(right[2]),
                "road_up_x": float(up[0]),
                "road_up_y": float(up[1]),
                "road_up_z": float(up[2]),
                "camera_x_world": float(cam_origin[0]),
                "camera_y_world": float(cam_origin[1]),
                "camera_z_world": float(cam_origin[2]),
            }

            # Save camera local->world rotation. This lets the UFLD script
            # back-project a predicted pixel ray exactly onto the local road plane.
            R = c2w[:3, :3]
            for i in range(3):
                for j in range(3):
                    rec[f"c2w_r{i}{j}"] = float(R[i, j])

            records.append(rec)

    out = pd.DataFrame(records)

    # Common intersection: same sample is lane-valid at every requested row.
    valid = out[
        (out["geometry_valid"] == 1) & (out["eligible"] == 1)
    ].copy()

    counts = valid.groupby("sample_id")["row_y"].nunique()
    common_ids = set(counts[counts == len(ROWS_Y)].index.astype(int))
    out["common_eligible"] = out["sample_id"].astype(int).isin(common_ids).astype(int)

    out.to_csv(OUT_CSV, index=False)

    summary_rows = []
    for row_y in ROWS_Y:
        rr = out[(out["row_y"] == row_y) & (out["eligible"] == 1)]
        cc = rr[rr["common_eligible"] == 1]
        summary_rows.append(
            {
                "row_y": row_y,
                "row_specific_eligible_n": len(rr),
                "common_eligible_n": len(cc),
                "distance_mean_m_common": (
                    float(cc["camera_to_ground_3d_m"].mean()) if len(cc) else np.nan
                ),
                "distance_median_m_common": (
                    float(cc["camera_to_ground_3d_m"].median()) if len(cc) else np.nan
                ),
                "distance_min_m_common": (
                    float(cc["camera_to_ground_3d_m"].min()) if len(cc) else np.nan
                ),
                "distance_max_m_common": (
                    float(cc["camera_to_ground_3d_m"].max()) if len(cc) else np.nan
                ),
                "lane_width_mean_m_common": (
                    float(cc["carla_lane_width_m"].mean()) if len(cc) else np.nan
                ),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV, index=False)

    # Critical self-check against the already-validated primary row.
    if "lane_gt_x" in meta.columns:
        y504 = out[
            (out["row_y"] == 504)
            & (out["geometry_valid"] == 1)
        ][["sample_id", "gt_center_x_px"]].copy()

        chk = meta[["sample_id", "lane_gt_x"]].merge(
            y504, on="sample_id", how="inner"
        )
        chk["lane_gt_x"] = pd.to_numeric(chk["lane_gt_x"], errors="coerce")
        chk = chk.dropna()

        if len(chk):
            diff = np.abs(
                chk["lane_gt_x"].to_numpy(dtype=float)
                - chk["gt_center_x_px"].to_numpy(dtype=float)
            )
            print()
            print(
                f"y=504 self-check vs recorded lane_gt_x: "
                f"mean abs={diff.mean():.6f}px, max={diff.max():.6f}px"
            )
            if diff.max() > 0.10:
                raise RuntimeError(
                    "y=504 geometry self-check FAILED (>0.10 px). "
                    "Do not run UFLD multirange evaluation yet."
                )
            print("y=504 self-check: PASS")

    print()
    print(summary.to_string(index=False))
    print()
    print("Common eligible sample count:", len(common_ids))
    print("Saved:", OUT_CSV)
    print("Saved:", SUMMARY_CSV)

if __name__ == "__main__":
    main()
