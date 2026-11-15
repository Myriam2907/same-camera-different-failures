#!/usr/bin/env python3
"""
UFLD-v2 multi-row matched-distance evaluation.

Run from the project root in the `ufldv2` conda environment AFTER
build_lane_multirange_geometry.py has produced:
  results/pilot/lane_multirange_geometry.csv

This script:
- reuses the exact model loader, preprocessing, official-style decoder, and
  fixed UFLD ego-lane slots from ufld_fog_sweep.py
- evaluates rows 504/480/460/440/420
- uses only the common CARLA-valid sample intersection across those rows
- converts predicted lane-center pixels to physical lateral error in metres by
  back-projecting the pixel ray onto the CARLA local road plane

Outputs:
  results/pilot/ufld_multirange_per_frame.csv
  results/pilot/ufld_multirange_summary.csv
  results/pilot/ufld_error_vs_distance.png
  results/pilot/ufld_availability_vs_distance.png

No new training and no detector tuning.
"""

from pathlib import Path
import inspect
import sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch

# UFLD-v2's own modules are named `utils`, `model`, `data`, etc.
# Put the cloned UFLD repository FIRST on sys.path so Python uses those
# modules rather than an unrelated PyPI package named `utils`.
PROJECT_ROOT = Path(__file__).resolve().parent
UFLD_ROOT = Path.home() / "Ultra-Fast-Lane-Detection-v2"

if not UFLD_ROOT.exists():
    raise RuntimeError(
        f"UFLD-v2 repository not found at {UFLD_ROOT}"
    )

sys.path.insert(0, str(UFLD_ROOT))

import ufld_fog_sweep as base

ROOT = Path(__file__).resolve().parent
GEOM_CSV = ROOT / "results" / "pilot" / "lane_multirange_geometry.csv"
OUT_DIR = ROOT / "results" / "pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROWS_Y = [504, 480, 460, 440, 420]
SEVERITIES = [0, 15, 30, 45, 60, 75, 90]

IMAGE_W = 1280
IMAGE_H = 720
FOV_DEG = 90.0
FX = IMAGE_W / (2.0 * np.tan(np.deg2rad(FOV_DEG) / 2.0))
FY = FX
CX = IMAGE_W / 2.0
CY = IMAGE_H / 2.0

def dataset_dir(severity):
    if severity == 0:
        return (
            ROOT / "data" / "recordings" / "stop3_clear_0_drift_0p75"
        )
    return (
        ROOT / "data" / "recordings"
        / f"stop3_fog_{severity}_drift_0p75"
    )

def find_model_loader():
    # Prefer obvious names first.
    for name in ["load_model", "build_model", "prepare_model", "get_model"]:
        fn = getattr(base, name, None)
        if inspect.isfunction(fn):
            try:
                src = inspect.getsource(fn)
            except Exception:
                src = ""
            if "return net, cfg" in src:
                return fn

    # Otherwise discover the exact function already used by ufld_fog_sweep.py.
    for name, fn in inspect.getmembers(base, inspect.isfunction):
        try:
            src = inspect.getsource(fn)
        except Exception:
            continue
        if "return net, cfg" in src and "load_state_dict" in src:
            return fn

    raise RuntimeError(
        "Could not find the existing UFLD model-loader function in "
        "ufld_fog_sweep.py."
    )

def resolve_existing_file(candidates, glob_roots, glob_patterns, label):
    for p in candidates:
        p = Path(p).expanduser()
        if p.exists():
            return p

    matches = []
    for root in glob_roots:
        root = Path(root).expanduser()
        if not root.exists():
            continue
        for pattern in glob_patterns:
            matches.extend(root.rglob(pattern))

    # Prefer the most exact / shortest-looking match.
    matches = [p for p in matches if p.is_file()]
    if matches:
        matches = sorted(
            set(matches),
            key=lambda p: (
                0 if "culane" in p.name.lower() else 1,
                0 if "res18" in p.name.lower() else 1,
                len(str(p)),
            ),
        )
        return matches[0]

    raise RuntimeError(
        f"Could not locate {label}. Checked known paths and searched "
        f"{[str(Path(r).expanduser()) for r in glob_roots]}."
    )


def call_model_loader(fn):
    sig = inspect.signature(fn)

    config_path = resolve_existing_file(
        candidates=[
            UFLD_ROOT / "configs" / "culane_res18.py",
            UFLD_ROOT / "configs" / "culane" / "res18.py",
            PROJECT_ROOT / "configs" / "culane_res18.py",
        ],
        glob_roots=[UFLD_ROOT, PROJECT_ROOT],
        glob_patterns=["culane_res18.py", "*culane*res18*.py"],
        label="UFLD CULane ResNet18 config",
    )

    weights_path = resolve_existing_file(
        candidates=[
            UFLD_ROOT / "weights" / "culane_res18.pth",
            PROJECT_ROOT / "weights" / "culane_res18.pth",
            PROJECT_ROOT / "culane_res18.pth",
        ],
        glob_roots=[UFLD_ROOT, PROJECT_ROOT],
        glob_patterns=[
            "culane_res18.pth",
            "*culane*res18*.pth",
            "*culane*.pth",
        ],
        label="UFLD CULane ResNet18 checkpoint",
    )

    print("Resolved config :", config_path)
    print("Resolved weights:", weights_path)

    args = []
    for name, p in sig.parameters.items():
        lname = name.lower()

        if p.default is not inspect._empty:
            continue

        if lname in {"config_path", "cfg_path", "config"}:
            args.append(str(config_path))
        elif lname in {"weights_path", "weight_path", "checkpoint_path", "ckpt_path"}:
            args.append(str(weights_path))
        elif "project" in lname and "root" in lname:
            args.append(PROJECT_ROOT)
        elif "root" in lname or "repo" in lname:
            args.append(UFLD_ROOT)
        else:
            raise RuntimeError(
                f"Found loader {fn.__name__}{sig}, but required parameter "
                f"'{name}' is not recognized."
            )

    result = fn(*args)
    if not (isinstance(result, tuple) and len(result) == 2):
        raise RuntimeError(
            f"{fn.__name__} did not return (net, cfg) as expected."
        )
    return result

def backproject_lateral_error_m(pred_x, row_y, g):
    origin = np.array(
        [
            g["camera_x_world"],
            g["camera_y_world"],
            g["camera_z_world"],
        ],
        dtype=float,
    )

    center = np.array(
        [
            g["center_x_world"],
            g["center_y_world"],
            g["center_z_world"],
        ],
        dtype=float,
    )

    right = np.array(
        [
            g["road_right_x"],
            g["road_right_y"],
            g["road_right_z"],
        ],
        dtype=float,
    )
    right /= max(np.linalg.norm(right), 1e-12)

    normal = np.array(
        [
            g["road_up_x"],
            g["road_up_y"],
            g["road_up_z"],
        ],
        dtype=float,
    )
    normal /= max(np.linalg.norm(normal), 1e-12)

    R = np.array(
        [
            [g["c2w_r00"], g["c2w_r01"], g["c2w_r02"]],
            [g["c2w_r10"], g["c2w_r11"], g["c2w_r12"]],
            [g["c2w_r20"], g["c2w_r21"], g["c2w_r22"]],
        ],
        dtype=float,
    )

    # CARLA camera local axes: X forward, Y right, Z up.
    ray_cam = np.array(
        [
            1.0,
            (float(pred_x) - CX) / FX,
            -(float(row_y) - CY) / FY,
        ],
        dtype=float,
    )
    ray_world = R @ ray_cam

    denom = float(np.dot(ray_world, normal))
    if abs(denom) < 1e-10:
        return np.nan, np.nan

    t = float(np.dot(center - origin, normal) / denom)
    if t <= 0.0:
        return np.nan, np.nan

    hit = origin + t * ray_world
    signed_lateral = float(np.dot(hit - center, right))
    return signed_lateral, abs(signed_lateral)

def main():
    if not GEOM_CSV.exists():
        raise RuntimeError(
            f"Missing {GEOM_CSV}. Run build_lane_multirange_geometry.py "
            f"in the carla environment first."
        )

    geom = pd.read_csv(GEOM_CSV)
    geom = geom[
        (geom["geometry_valid"] == 1)
        & (geom["eligible"] == 1)
        & (geom["common_eligible"] == 1)
    ].copy()

    if geom.empty:
        raise RuntimeError("No common eligible lane geometry rows found.")

    common_ids = set(geom["sample_id"].astype(int).unique())
    print("=" * 92)
    print("UFLD-v2 MULTI-RANGE MATCHED-DISTANCE EVALUATION")
    print("=" * 92)
    print("Common eligible samples:", len(common_ids))
    print("Rows:", ROWS_Y)

    loader = find_model_loader()
    print("Using existing model loader:", loader.__name__, inspect.signature(loader))
    net, cfg = call_model_loader(loader)
    transform = base.make_transform(cfg)

    all_records = []

    for severity in SEVERITIES:
        droot = dataset_dir(severity)
        frame_dir = droot / "frames"
        meta_csv = droot / "frames.csv"

        if not meta_csv.exists():
            raise RuntimeError(f"Missing dataset metadata: {meta_csv}")

        meta = pd.read_csv(meta_csv)
        meta = meta[meta["sample_id"].astype(int).isin(common_ids)].copy()

        print()
        print(f"Fog {severity}: {len(meta)} common samples")

        for _, mr in meta.iterrows():
            sid = int(mr["sample_id"])
            image_path = frame_dir / f"{sid:04d}.png"
            if not image_path.exists():
                raise RuntimeError(f"Missing image: {image_path}")

            pil = Image.open(image_path).convert("RGB")
            original_w, original_h = pil.size

            tensor = transform(pil)
            tensor = tensor[:, -cfg.train_height :, :]
            tensor = tensor.unsqueeze(0).cuda()

            with torch.no_grad():
                pred = net(tensor)

            # Exact same fixed slots as the validated y=504 pipeline.
            left_pts = base.decode_row_lane(
                pred, 1, cfg.row_anchor, original_w, original_h
            )
            right_pts = base.decode_row_lane(
                pred, 2, cfg.row_anchor, original_w, original_h
            )

            for row_y in ROWS_Y:
                gg = geom[
                    (geom["sample_id"].astype(int) == sid)
                    & (geom["row_y"].astype(int) == row_y)
                ]

                if len(gg) != 1:
                    raise RuntimeError(
                        f"Expected one geometry row for sample={sid}, y={row_y}; "
                        f"found {len(gg)}."
                    )

                g = gg.iloc[0]

                left_x = base.interpolate_x_at_y(left_pts, float(row_y))
                right_x = base.interpolate_x_at_y(right_pts, float(row_y))

                success = (
                    left_x is not None
                    and right_x is not None
                    and float(left_x) < float(right_x)
                )

                if success:
                    pred_center = 0.5 * (float(left_x) + float(right_x))
                    signed_px = pred_center - float(g["gt_center_x_px"])
                    error_px = abs(signed_px)

                    signed_m, error_m = backproject_lateral_error_m(
                        pred_center, row_y, g
                    )
                else:
                    pred_center = np.nan
                    signed_px = np.nan
                    error_px = np.nan
                    signed_m = np.nan
                    error_m = np.nan

                all_records.append(
                    {
                        "severity": severity,
                        "sample_id": sid,
                        "route_s_m": float(mr["route_s_m"]),
                        "row_y": row_y,
                        "camera_to_ground_3d_m": float(
                            g["camera_to_ground_3d_m"]
                        ),
                        "gt_center_x_px": float(g["gt_center_x_px"]),
                        "pred_left_x_px": (
                            float(left_x) if left_x is not None else np.nan
                        ),
                        "pred_right_x_px": (
                            float(right_x) if right_x is not None else np.nan
                        ),
                        "pred_center_x_px": pred_center,
                        "success": int(success),
                        "signed_error_px": signed_px,
                        "error_px": error_px,
                        "signed_lateral_error_m": signed_m,
                        "lateral_error_m": error_m,
                    }
                )

    per = pd.DataFrame(all_records)
    per_path = OUT_DIR / "ufld_multirange_per_frame.csv"
    per.to_csv(per_path, index=False)

    summaries = []
    for severity in SEVERITIES:
        for row_y in ROWS_Y:
            z = per[
                (per["severity"] == severity)
                & (per["row_y"] == row_y)
            ].copy()

            good = z[z["success"] == 1].copy()
            n = len(z)
            s = len(good)

            summaries.append(
                {
                    "severity": severity,
                    "row_y": row_y,
                    "n_common_frames": n,
                    "successes": s,
                    "availability_pct": 100.0 * s / n if n else np.nan,
                    "distance_mean_m": z["camera_to_ground_3d_m"].mean(),
                    "distance_median_m": z["camera_to_ground_3d_m"].median(),
                    "mae_px_success_only": good["error_px"].mean(),
                    "mae_m_success_only": good["lateral_error_m"].mean(),
                    "median_error_m_success_only": good["lateral_error_m"].median(),
                }
            )

    summary = pd.DataFrame(summaries)

    # Row-specific clear-relative physical error ratio for the sensitivity analysis.
    # This is NOT the preregistered primary R_lane; it is a secondary range-control ratio.
    clear = summary[summary["severity"] == 0][
        ["row_y", "mae_m_success_only"]
    ].rename(columns={"mae_m_success_only": "clear_mae_m"})

    summary = summary.merge(clear, on="row_y", how="left")
    summary["physical_error_ratio_vs_clear"] = (
        summary["mae_m_success_only"] / summary["clear_mae_m"]
    )

    summary_path = OUT_DIR / "ufld_multirange_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print(summary[
        [
            "severity", "row_y", "n_common_frames",
            "availability_pct", "distance_median_m",
            "mae_m_success_only", "physical_error_ratio_vs_clear",
        ]
    ].round(4).to_string(index=False))

    # Figure: physical localization error vs physical range, one curve per fog severity.
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for severity in SEVERITIES:
        z = summary[summary["severity"] == severity].sort_values(
            "distance_median_m"
        )
        ax.plot(
            z["distance_median_m"],
            z["mae_m_success_only"],
            marker="o",
            label=f"Fog {severity}",
        )
    ax.set_xlabel("Camera-to-lane target distance (m)")
    ax.set_ylabel("UFLD lane-center MAE (m)")
    ax.set_title("UFLD-v2 lane localization vs physical distance")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ufld_error_vs_distance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    for severity in SEVERITIES:
        z = summary[summary["severity"] == severity].sort_values(
            "distance_median_m"
        )
        ax.plot(
            z["distance_median_m"],
            z["availability_pct"],
            marker="o",
            label=f"Fog {severity}",
        )
    ax.set_xlabel("Camera-to-lane target distance (m)")
    ax.set_ylabel("Lane availability (%)")
    ax.set_title("UFLD-v2 lane availability vs physical distance")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ufld_availability_vs_distance.png", dpi=180)
    plt.close(fig)

    print()
    print("Saved:", per_path)
    print("Saved:", summary_path)
    print("Saved:", OUT_DIR / "ufld_error_vs_distance.png")
    print("Saved:", OUT_DIR / "ufld_availability_vs_distance.png")
    print()
    print(
        "NOTE: physical_error_ratio_vs_clear is a secondary range-control "
        "quantity only. It does not replace the preregistered y=504 R_lane."
    )

if __name__ == "__main__":
    main()
