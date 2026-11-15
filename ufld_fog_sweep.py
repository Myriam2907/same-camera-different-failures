#!/usr/bin/env python3
"""
UFLD-v2 numerical Clear gate for the CARLA lane project.

Run FROM the official Ultra-Fast-Lane-Detection-v2 repository.

What it does:
- Loads official CULane ResNet18 UFLD-v2 checkpoint.
- Runs ONLY the canonical Clear drift recording.
- Excludes junction frames exactly as the project does.
- Uses UFLD lane slots 1 and 2, validated visually in the 10-frame smoke test.
- Interpolates each lane at project lookahead row y=504.
- Predicted lane center = midpoint(left_x, right_x).
- If either ego-lane boundary is unavailable at y=504, error = 640 px.
- Writes per-frame CSV + summary.
- Prints the same Clear sanity diagnostics used for other lane methods.

This is still CLEAR ONLY. Do not run fog until the Clear gate is reviewed.
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

from utils.config import Config
from utils.common import get_model


LOOKAHEAD_Y = 504.0
FAILURE_ERROR_PX = 640.0
TRIVIAL_CENTER_ERROR = 40.922341


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/culane_res18.py")
    p.add_argument("--weights", default="weights/culane_res18.pth")
    p.add_argument(
        "--project_root",
        default=str(Path.home() / "same-camera-different-failures"),
    )
    return p.parse_args()


def load_model(config_path, weights_path):
    cfg = Config.fromfile(config_path)

    # Same anchors used by the official CULane configuration.
    cfg.row_anchor = np.linspace(0.42, 1.0, cfg.num_row)
    cfg.col_anchor = np.linspace(0.0, 1.0, cfg.num_col)

    net = get_model(cfg)

    checkpoint = torch.load(weights_path, map_location="cpu")
    state_dict = checkpoint["model"]

    compatible = {}
    for k, v in state_dict.items():
        compatible[k[7:] if k.startswith("module.") else k] = v

    missing, unexpected = net.load_state_dict(compatible, strict=False)

    if missing or unexpected:
        print("WARNING:")
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

    net.eval().cuda()
    return net, cfg


def make_transform(cfg):
    return transforms.Compose([
        transforms.Resize(
            (
                int(cfg.train_height / cfg.crop_ratio),
                cfg.train_width,
            )
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])


def decode_row_lane(pred, lane_id, row_anchor, original_w, original_h, local_width=1):
    """
    Decode one UFLD row-anchor lane exactly in the style of official demo.py.
    Returns list of (x,y) points in original-image coordinates.
    """
    _, num_grid_row, num_cls_row, _ = pred["loc_row"].shape

    max_indices_row = pred["loc_row"].argmax(1).cpu()
    valid_row = pred["exist_row"].argmax(1).cpu()
    loc_row = pred["loc_row"].detach().cpu()

    pts = []

    if valid_row[0, :, lane_id].sum() <= num_cls_row / 2:
        return pts

    for k in range(valid_row.shape[1]):
        if not valid_row[0, k, lane_id]:
            continue

        center = int(max_indices_row[0, k, lane_id])

        inds = torch.tensor(
            list(
                range(
                    max(0, center - local_width),
                    min(num_grid_row - 1, center + local_width) + 1,
                )
            )
        )

        out_tmp = (
            loc_row[0, inds, k, lane_id].softmax(0) * inds.float()
        ).sum() + 0.5

        x = float(out_tmp) / (num_grid_row - 1) * original_w
        y = float(row_anchor[k]) * original_h

        pts.append((x, y))

    return pts


def interpolate_x_at_y(points, target_y):
    if len(points) < 2:
        return None

    pts = sorted(points, key=lambda p: p[1])
    ys = np.array([p[1] for p in pts], dtype=float)
    xs = np.array([p[0] for p in pts], dtype=float)

    if target_y < ys.min() or target_y > ys.max():
        return None

    # Remove duplicate y values if any.
    unique_y, idx = np.unique(ys, return_index=True)
    unique_x = xs[idx]

    if len(unique_y) < 2:
        return None

    return float(np.interp(target_y, unique_y, unique_x))


def pearson_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])



def evaluate_dataset(net, cfg, transform, project_root, severity):
    dataset = (
        project_root
        / "data"
        / "recordings"
        / f"stop3_fog_{severity}_drift_0p75"
    )
    csv_path = dataset / "frames.csv"
    frames_dir = dataset / "frames"

    output_dir = project_root / "results" / "lane_ufld"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise RuntimeError(f"Missing dataset CSV: {csv_path}")

    with csv_path.open(newline="") as f:
        input_rows = list(csv.DictReader(f))

    out_rows = []

    total = 0
    junction_excluded = 0
    evaluated = 0
    successes = 0
    failures = 0

    for r in input_rows:
        total += 1

        sid = int(float(r["sample_id"]))
        is_junction = int(float(r["is_junction"]))

        # Keep this exactly consistent with the validated Clear evaluator.
        # The final 88-frame lookahead-valid mask is applied later by
        # lane_recompute.py, identically for every method and severity.
        if is_junction:
            junction_excluded += 1
            continue

        gt_text = r.get("lane_gt_x", "")
        if gt_text is None or gt_text == "":
            continue

        gt_x = float(gt_text)
        evaluated += 1

        image_path = frames_dir / f"{sid:04d}.png"
        if not image_path.exists():
            raise RuntimeError(f"Missing image: {image_path}")

        pil = Image.open(image_path).convert("RGB")
        original_w, original_h = pil.size

        # EXACT SAME preprocessing as ufld_clear_gate.py
        tensor = transform(pil)
        tensor = tensor[:, -cfg.train_height :, :]
        tensor = tensor.unsqueeze(0).cuda()

        with torch.no_grad():
            pred = net(tensor)

        # EXACT SAME fixed ego-lane slots as Clear:
        # slot 1 = left, slot 2 = right.
        # Ground truth is never used for lane selection.
        left_pts = decode_row_lane(
            pred, 1, cfg.row_anchor, original_w, original_h
        )
        right_pts = decode_row_lane(
            pred, 2, cfg.row_anchor, original_w, original_h
        )

        left_x = interpolate_x_at_y(left_pts, LOOKAHEAD_Y)
        right_x = interpolate_x_at_y(right_pts, LOOKAHEAD_Y)

        success = (
            left_x is not None
            and right_x is not None
            and left_x < right_x
        )

        if success:
            successes += 1
            pred_x = 0.5 * (left_x + right_x)
            error = abs(pred_x - gt_x)
            signed_error = pred_x - gt_x
            width = right_x - left_x
        else:
            failures += 1
            pred_x = None
            error = FAILURE_ERROR_PX
            signed_error = None
            width = None

        out_rows.append({
            "sample_id": sid,
            "route_s_m": r.get("route_s_m", ""),
            "lane_gt_x": gt_x,
            "left_x_y504": left_x if left_x is not None else "",
            "right_x_y504": right_x if right_x is not None else "",
            "pred_center_x_y504": pred_x if pred_x is not None else "",
            "signed_error_px": signed_error if signed_error is not None else "",
            "error_px": error,
            "width_px": width if width is not None else "",
            "success": int(success),
        })

    per_frame_path = output_dir / f"fog{severity}_per_frame.csv"

    with per_frame_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(out_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(out_rows)

    errors_all = np.array(
        [float(r["error_px"]) for r in out_rows],
        dtype=float,
    )
    good = [
        r for r in out_rows
        if int(r["success"]) == 1
    ]

    print()
    print("=" * 72)
    print(f"UFLD-v2 FOG {severity}")
    print("=" * 72)
    print(f"Dataset: {dataset}")
    print(f"Total frames: {total}")
    print(f"Ego-junction excluded: {junction_excluded}")
    print(f"Rows written before final 88-frame mask: {evaluated}")
    print(f"Successful detections: {successes}")
    print(f"Detector failures: {failures}")
    print(
        f"Detection rate on these {evaluated} rows: "
        f"{100.0 * successes / evaluated:.2f}%"
    )
    print(
        "Mean error before final 88-frame mask: "
        f"{errors_all.mean():.2f} px"
    )

    if good:
        success_errors = np.array(
            [float(r["error_px"]) for r in good],
            dtype=float,
        )
        widths = np.array(
            [float(r["width_px"]) for r in good],
            dtype=float,
        )

        print(
            "Success-only center MAE before final mask: "
            f"{success_errors.mean():.2f} px"
        )
        print(
            "Predicted width mean/std before final mask: "
            f"{widths.mean():.2f} / {widths.std():.2f} px"
        )

    print(f"Saved: {per_frame_path}")

    return {
        "severity": severity,
        "evaluated": evaluated,
        "successes": successes,
        "failures": failures,
        "availability_pct": 100.0 * successes / evaluated,
        "mean_error_pre_mask": float(errors_all.mean()),
    }


def main():
    args = parse_args()

    project_root = Path(args.project_root).expanduser()

    net, cfg = load_model(args.config, args.weights)
    transform = make_transform(cfg)

    print("=" * 72)
    print("UFLD-v2 FOG SWEEP")
    print("=" * 72)
    print("Model config:", args.config)
    print("Weights:", args.weights)
    print("Fixed lane slots: 1 (left), 2 (right)")
    print("Lookahead row:", LOOKAHEAD_Y)
    print("Failure penalty:", FAILURE_ERROR_PX, "px")
    print("num_grid_row:", cfg.num_grid_row)
    print(
        "Nominal x-bin spacing:",
        f"{1280.0 / (cfg.num_grid_row - 1):.3f} px"
    )
    print()
    print(
        "IMPORTANT: these per-frame files intentionally preserve the exact "
        "Clear evaluator behavior. The method-independent 88-frame "
        "lookahead-valid mask is applied afterward by lane_recompute.py."
    )

    severities = [15, 30, 45, 60, 75, 90]
    results = []

    for severity in severities:
        results.append(
            evaluate_dataset(
                net,
                cfg,
                transform,
                project_root,
                severity,
            )
        )

    print()
    print("=" * 72)
    print("FOG SWEEP COMPLETE")
    print("=" * 72)
    print(
        f"{'Fog':>5} {'Rows':>6} {'Succ':>6} "
        f"{'Fail':>6} {'Avail%':>9} {'PreMaskMAE':>12}"
    )

    for r in results:
        print(
            f"{r['severity']:>5} "
            f"{r['evaluated']:>6} "
            f"{r['successes']:>6} "
            f"{r['failures']:>6} "
            f"{r['availability_pct']:>9.2f} "
            f"{r['mean_error_pre_mask']:>12.2f}"
        )

    print()
    print("Next:")
    print("  cd ~/same-camera-different-failures")
    print("  conda activate carla")
    print("  python lane_recompute.py")
    print()
    print(
        "Use the lane_recompute.py output as the FINAL lane results, "
        "not the pre-mask numbers printed here."
    )


if __name__ == "__main__":
    main()
