#!/usr/bin/env python3
"""
Sliding-window + polynomial lane baseline v2 for the CARLA drift recordings.

Important protocol rules
------------------------
1. Detection never uses CARLA lane_gt_x.
2. Only non-junction frames with finite lane_gt_x are evaluated.
3. If the detector fails on an evaluable frame, error = 640 px.
4. Default run is CLEAR ONLY. Inspect the Clear gate first.
5. Only after Clear is accepted should you run --all and freeze the parameters.

The detector:
image -> fixed trapezoid ROI -> white/yellow color binary mask
      -> left/right histogram seeds -> sliding windows
      -> second-order x(y) polynomial fits
      -> lane center at y = 504

Outputs:
results/lane_poly/<dataset>_per_frame.csv
results/lane_poly/summary.csv
results/lane_poly/debug/<dataset>/*.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# FROZEN EXPERIMENT CONFIG
# ============================================================

IMAGE_W = 1280
IMAGE_H = 720
LOOKAHEAD_ROW = 504

TRIVIAL_CENTER_X = IMAGE_W / 2.0
FAILURE_ERROR_PX = IMAGE_W / 2.0
TRIVIAL_CLEAR_ERROR_PX = 40.922341

DATASETS = {
    "clear": {
        "root": "data/recordings/stop3_clear_0_drift_0p75",
        "severity": 0,
    },
    "fog15": {
        "root": "data/recordings/stop3_fog_15_drift_0p75",
        "severity": 15,
    },
    "fog30": {
        "root": "data/recordings/stop3_fog_30_drift_0p75",
        "severity": 30,
    },
    "fog45": {
        "root": "data/recordings/stop3_fog_45_drift_0p75",
        "severity": 45,
    },
    "fog60": {
        "root": "data/recordings/stop3_fog_60_drift_0p75",
        "severity": 60,
    },
    "fog75": {
        "root": "data/recordings/stop3_fog_75_drift_0p75",
        "severity": 75,
    },
    "fog90": {
        "root": "data/recordings/stop3_fog_90_drift_0p75",
        "severity": 90,
    },
}

OUTPUT_ROOT = Path("results/lane_poly")
DEBUG_ROOT = OUTPUT_ROOT / "debug"

# ============================================================
# DETECTOR PARAMETERS
# Freeze these before the fog sweep.
# ============================================================

# Trapezoid ROI.
ROI_TOP_Y_FRAC = 0.43
ROI_BOTTOM_LEFT_X_FRAC = 0.08
ROI_BOTTOM_RIGHT_X_FRAC = 0.92
ROI_TOP_LEFT_X_FRAC = 0.38
ROI_TOP_RIGHT_X_FRAC = 0.62

# Color thresholds.
# White lane markings in HLS.
WHITE_L_MIN = 150
WHITE_S_MAX = 170

# Yellow lane markings in HSV.
YELLOW_H_MIN = 12
YELLOW_H_MAX = 42
YELLOW_S_MIN = 55
YELLOW_V_MIN = 70

# Gradient cue.
SOBEL_MIN = 28

# Sliding windows.
NWINDOWS = 10
WINDOW_MARGIN = 90
MINPIX_RECENTER = 30
MIN_TOTAL_PIXELS_PER_SIDE = 160

# Broad geometric sanity only. These are intentionally loose.
MIN_LANE_WIDTH_PX = 220.0
MAX_LANE_WIDTH_PX = 650.0

# Histogram seed search ranges.
# IMPORTANT: left/right ranges do NOT overlap.
# The first version overlapped around the image center, which allowed
# both sliding-window searches to initialize on the same strong marking.
LEFT_SEED_MIN_X = int(0.15 * IMAGE_W)
LEFT_SEED_MAX_X = int(0.50 * IMAGE_W)
RIGHT_SEED_MIN_X = int(0.56 * IMAGE_W)
RIGHT_SEED_MAX_X = int(0.90 * IMAGE_W)

# Side-specific x limits used during sliding-window tracking.
LEFT_TRACK_MIN_X = int(0.12 * IMAGE_W)
LEFT_TRACK_MAX_X = int(0.55 * IMAGE_W)
RIGHT_TRACK_MIN_X = int(0.52 * IMAGE_W)
RIGHT_TRACK_MAX_X = int(0.93 * IMAGE_W)

# Representative Clear images saved automatically.
DEBUG_SAMPLE_IDS = {40, 60, 80, 100, 120, 140, 160}


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Sliding-window polynomial lane baseline."
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all seven weather conditions. Default: Clear only.",
    )
    return p.parse_args()


# ============================================================
# PREPROCESSING
# ============================================================

def make_roi_mask():
    mask = np.zeros((IMAGE_H, IMAGE_W), dtype=np.uint8)

    top_y = int(ROI_TOP_Y_FRAC * IMAGE_H)

    polygon = np.array(
        [[
            (int(ROI_BOTTOM_LEFT_X_FRAC * IMAGE_W), IMAGE_H - 1),
            (int(ROI_TOP_LEFT_X_FRAC * IMAGE_W), top_y),
            (int(ROI_TOP_RIGHT_X_FRAC * IMAGE_W), top_y),
            (int(ROI_BOTTOM_RIGHT_X_FRAC * IMAGE_W), IMAGE_H - 1),
        ]],
        dtype=np.int32,
    )

    cv2.fillPoly(mask, polygon, 255)
    return mask


ROI_MASK = make_roi_mask()


def preprocess_binary(image):
    """
    Build a fixed binary lane-candidate mask.

    This uses image cues only. It does not use lane_gt_x.
    """
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # White mask.
    _, l, s_hls = cv2.split(hls)
    white = (
        (l >= WHITE_L_MIN)
        & (s_hls <= WHITE_S_MAX)
    ).astype(np.uint8) * 255

    # Yellow mask.
    h, s_hsv, v = cv2.split(hsv)
    yellow = (
        (h >= YELLOW_H_MIN)
        & (h <= YELLOW_H_MAX)
        & (s_hsv >= YELLOW_S_MIN)
        & (v >= YELLOW_V_MIN)
    ).astype(np.uint8) * 255

    # Clear-development version: use lane-marking color only.
    # The previous global Sobel cue strongly activated road cracks,
    # shadows and STOP lettering, which distracted the sliding windows.
    binary = cv2.bitwise_or(white, yellow)

    # Restrict to fixed road ROI.
    binary = cv2.bitwise_and(binary, ROI_MASK)

    # Remove isolated one-pixel noise and reconnect short marking gaps.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    return binary


# ============================================================
# SLIDING-WINDOW POLYNOMIAL DETECTOR
# ============================================================

def histogram_seed(binary, x0, x1):
    """
    Find strongest lower-image histogram peak inside [x0, x1).
    """
    lower = binary[IMAGE_H // 2 :, :]
    hist = np.sum(lower > 0, axis=0).astype(np.float64)

    search = hist[x0:x1]
    if len(search) == 0 or search.max() <= 0:
        return None, hist

    return int(x0 + np.argmax(search)), hist


def collect_lane_pixels(binary, base_x, side):
    """
    Standard vertical sliding-window search.
    """
    nonzero_y, nonzero_x = binary.nonzero()

    window_height = IMAGE_H // NWINDOWS
    current_x = int(base_x)

    lane_indices = []

    for window in range(NWINDOWS):
        win_y_low = IMAGE_H - (window + 1) * window_height
        win_y_high = IMAGE_H - window * window_height

        # Do not search above the ROI top.
        roi_top = int(ROI_TOP_Y_FRAC * IMAGE_H)
        if win_y_high <= roi_top:
            break

        win_y_low = max(win_y_low, roi_top)

        win_x_low = current_x - WINDOW_MARGIN
        win_x_high = current_x + WINDOW_MARGIN

        # Keep the two searches separated. The first implementation let
        # both windows drift onto the same central/interior marking.
        if side == "left":
            win_x_low = max(win_x_low, LEFT_TRACK_MIN_X)
            win_x_high = min(win_x_high, LEFT_TRACK_MAX_X)
        elif side == "right":
            win_x_low = max(win_x_low, RIGHT_TRACK_MIN_X)
            win_x_high = min(win_x_high, RIGHT_TRACK_MAX_X)
        else:
            raise ValueError(f"Unknown side: {side}")

        if win_x_high <= win_x_low:
            continue

        good = (
            (nonzero_y >= win_y_low)
            & (nonzero_y < win_y_high)
            & (nonzero_x >= win_x_low)
            & (nonzero_x < win_x_high)
        )

        inds = np.nonzero(good)[0]
        lane_indices.append(inds)

        if len(inds) >= MINPIX_RECENTER:
            current_x = int(np.mean(nonzero_x[inds]))

    if not lane_indices:
        return np.array([], dtype=float), np.array([], dtype=float)

    lane_indices = np.concatenate(lane_indices)

    return (
        nonzero_x[lane_indices].astype(float),
        nonzero_y[lane_indices].astype(float),
    )


def fit_side(binary, base_x, side):
    x, y = collect_lane_pixels(binary, base_x, side)

    if len(x) < MIN_TOTAL_PIXELS_PER_SIDE:
        return None, len(x)

    # Fit x as a function of y.
    # x(y) = a*y^2 + b*y + c
    try:
        coeff = np.polyfit(y, x, 2)
    except Exception:
        return None, len(x)

    if not np.all(np.isfinite(coeff)):
        return None, len(x)

    return coeff, len(x)


def eval_poly(coeff, y):
    return float(
        coeff[0] * y * y
        + coeff[1] * y
        + coeff[2]
    )


def detect_lane(image):
    binary = preprocess_binary(image)

    left_base, hist = histogram_seed(
        binary,
        LEFT_SEED_MIN_X,
        LEFT_SEED_MAX_X,
    )
    right_base, _ = histogram_seed(
        binary,
        RIGHT_SEED_MIN_X,
        RIGHT_SEED_MAX_X,
    )

    debug = {
        "binary": binary,
        "hist": hist,
        "left_base": left_base,
        "right_base": right_base,
        "left_fit": None,
        "right_fit": None,
        "left_pixels": 0,
        "right_pixels": 0,
    }

    if left_base is None or right_base is None:
        return None, None, None, debug

    left_fit, left_n = fit_side(binary, left_base, "left")
    right_fit, right_n = fit_side(binary, right_base, "right")

    debug["left_fit"] = left_fit
    debug["right_fit"] = right_fit
    debug["left_pixels"] = left_n
    debug["right_pixels"] = right_n

    if left_fit is None or right_fit is None:
        return None, None, None, debug

    left_x = eval_poly(left_fit, LOOKAHEAD_ROW)
    right_x = eval_poly(right_fit, LOOKAHEAD_ROW)

    if not (
        np.isfinite(left_x)
        and np.isfinite(right_x)
    ):
        return None, None, None, debug

    if not (
        0.0 <= left_x < IMAGE_W
        and 0.0 <= right_x < IMAGE_W
    ):
        return None, None, None, debug

    width = right_x - left_x

    if not (
        MIN_LANE_WIDTH_PX
        <= width
        <= MAX_LANE_WIDTH_PX
    ):
        return None, None, None, debug

    pred_x = 0.5 * (left_x + right_x)

    return pred_x, left_x, right_x, debug


# ============================================================
# DEBUG VISUALIZATION
# ============================================================

def save_debug_image(
    image,
    out_path,
    lane_gt_x,
    pred_x,
    left_x,
    right_x,
    error_px,
    debug,
):
    vis = image.copy()

    # ROI polygon.
    top_y = int(ROI_TOP_Y_FRAC * IMAGE_H)
    roi_poly = np.array(
        [
            (int(ROI_BOTTOM_LEFT_X_FRAC * IMAGE_W), IMAGE_H - 1),
            (int(ROI_TOP_LEFT_X_FRAC * IMAGE_W), top_y),
            (int(ROI_TOP_RIGHT_X_FRAC * IMAGE_W), top_y),
            (int(ROI_BOTTOM_RIGHT_X_FRAC * IMAGE_W), IMAGE_H - 1),
        ],
        dtype=np.int32,
    )
    cv2.polylines(vis, [roi_poly], True, (255, 255, 0), 2)

    # Draw fitted curves.
    for fit, color in [
        (debug.get("left_fit"), (0, 255, 255)),
        (debug.get("right_fit"), (255, 255, 0)),
    ]:
        if fit is None:
            continue

        ys = np.arange(top_y, IMAGE_H, 5)
        xs = (
            fit[0] * ys * ys
            + fit[1] * ys
            + fit[2]
        )

        pts = []
        for x, y in zip(xs, ys):
            if 0 <= x < IMAGE_W:
                pts.append((int(round(x)), int(y)))

        if len(pts) >= 2:
            cv2.polylines(
                vis,
                [np.array(pts, dtype=np.int32)],
                False,
                color,
                4,
            )

    # Lookahead row.
    cv2.line(
        vis,
        (0, LOOKAHEAD_ROW),
        (IMAGE_W - 1, LOOKAHEAD_ROW),
        (255, 255, 255),
        1,
    )

    # GT point = red.
    if lane_gt_x is not None and np.isfinite(lane_gt_x):
        cv2.circle(
            vis,
            (int(round(lane_gt_x)), LOOKAHEAD_ROW),
            10,
            (0, 0, 255),
            -1,
        )

    # Predicted midpoint = green.
    if pred_x is not None:
        cv2.circle(
            vis,
            (int(round(pred_x)), LOOKAHEAD_ROW),
            10,
            (0, 255, 0),
            -1,
        )

    # Left/right predicted boundaries.
    if left_x is not None:
        cv2.circle(
            vis,
            (int(round(left_x)), LOOKAHEAD_ROW),
            8,
            (0, 255, 255),
            -1,
        )

    if right_x is not None:
        cv2.circle(
            vis,
            (int(round(right_x)), LOOKAHEAD_ROW),
            8,
            (255, 255, 0),
            -1,
        )

    label = (
        f"Poly={pred_x:.1f} GT={lane_gt_x:.1f} err={error_px:.1f}px"
        if pred_x is not None and lane_gt_x is not None
        else "Poly detector failure"
    )

    cv2.putText(
        vis,
        label,
        (25, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)

    mask_path = out_path.with_name(out_path.stem + "_mask.png")
    cv2.imwrite(str(mask_path), debug["binary"])


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(dataset_name, info):
    root = Path(info["root"])
    csv_path = root / "frames.csv"
    frame_dir = root / "frames"

    if not csv_path.exists():
        raise RuntimeError(f"Missing: {csv_path}")

    if not frame_dir.exists():
        raise RuntimeError(f"Missing: {frame_dir}")

    df = pd.read_csv(csv_path)

    required = {
        "sample_id",
        "severity",
        "route_s_m",
        "is_junction",
        "lane_gt_x",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            f"{csv_path} missing columns: {missing}"
        )

    results = []

    junction_excluded = 0
    evaluated_count = 0
    success_count = 0
    failure_count = 0

    errors_all = []
    errors_success = []

    print()
    print("=" * 60)
    print(f"Processing: {dataset_name}")
    print("=" * 60)

    for _, row in df.iterrows():
        sample_id = int(row["sample_id"])

        image_path = frame_dir / f"{sample_id:04d}.png"
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Could not read image: {image_path}"
            )

        if image.shape[:2] != (IMAGE_H, IMAGE_W):
            raise RuntimeError(
                f"Unexpected image shape {image.shape} in {image_path}"
            )

        is_junction = int(row["is_junction"]) != 0

        lane_gt_x = (
            None
            if pd.isna(row["lane_gt_x"])
            else float(row["lane_gt_x"])
        )

        pred_x, left_x, right_x, debug = detect_lane(image)

        if is_junction or lane_gt_x is None:
            status = "junction_excluded"
            evaluated = 0
            error_px = np.nan

            if is_junction:
                junction_excluded += 1

        else:
            evaluated = 1
            evaluated_count += 1

            if pred_x is None:
                status = "detector_failure"
                error_px = FAILURE_ERROR_PX
                failure_count += 1
                errors_all.append(error_px)
            else:
                status = "success"
                error_px = abs(pred_x - lane_gt_x)
                success_count += 1
                errors_all.append(error_px)
                errors_success.append(error_px)

        results.append({
            "sample_id": sample_id,
            "weather": row.get("weather", dataset_name),
            "severity": int(info["severity"]),
            "route_s_m": float(row["route_s_m"]),
            "is_junction": int(row["is_junction"]),
            "lane_gt_x": lane_gt_x,
            "poly_left_x": left_x,
            "poly_right_x": right_x,
            "poly_pred_x": pred_x,
            "evaluated": evaluated,
            "status": status,
            "error_px": error_px,
            "left_pixels": debug.get("left_pixels", 0),
            "right_pixels": debug.get("right_pixels", 0),
        })

        if (
            dataset_name == "clear"
            and not is_junction
            and sample_id in DEBUG_SAMPLE_IDS
        ):
            debug_error = (
                0.0
                if np.isnan(error_px)
                else float(error_px)
            )
            save_debug_image(
                image=image,
                out_path=DEBUG_ROOT / dataset_name / f"{sample_id:04d}.png",
                lane_gt_x=lane_gt_x,
                pred_x=pred_x,
                left_x=left_x,
                right_x=right_x,
                error_px=debug_error,
                debug=debug,
            )

    results_df = pd.DataFrame(results)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    per_frame_path = (
        OUTPUT_ROOT / f"{dataset_name}_per_frame.csv"
    )
    results_df.to_csv(per_frame_path, index=False)

    detection_rate = (
        100.0 * success_count / evaluated_count
        if evaluated_count
        else np.nan
    )

    mean_all = (
        float(np.mean(errors_all))
        if errors_all
        else np.nan
    )

    median_all = (
        float(np.median(errors_all))
        if errors_all
        else np.nan
    )

    mean_success = (
        float(np.mean(errors_success))
        if errors_success
        else np.nan
    )

    median_success = (
        float(np.median(errors_success))
        if errors_success
        else np.nan
    )

    print(f"Total frames: {len(df)}")
    print(f"Junction excluded: {junction_excluded}")
    print(f"Evaluated frames: {evaluated_count}")
    print(f"Successful detections: {success_count}")
    print(f"Detector failures: {failure_count}")
    print(f"Detection rate: {detection_rate:.2f}%")
    print(f"Mean error (failures included): {mean_all:.2f} px")
    print(f"Median error (failures included): {median_all:.2f} px")

    if np.isfinite(mean_success):
        print(f"Mean error (success only): {mean_success:.2f} px")
        print(f"Median error (success only): {median_success:.2f} px")

    return {
        "dataset": dataset_name,
        "severity": int(info["severity"]),
        "total_frames": len(df),
        "junction_excluded": junction_excluded,
        "evaluated_frames": evaluated_count,
        "successful_detections": success_count,
        "detector_failures": failure_count,
        "detection_rate_percent": detection_rate,
        "mean_error_penalized_px": mean_all,
        "median_error_penalized_px": median_all,
        "mean_error_success_only_px": mean_success,
        "median_error_success_only_px": median_success,
    }


# ============================================================
# SAME CLEAR GEOMETRY GATE USED FOR LSD / HOUGH
# ============================================================

def print_clear_gate():
    path = OUTPUT_ROOT / "clear_per_frame.csv"

    if not path.exists():
        raise RuntimeError(
            f"Missing Clear output: {path}"
        )

    df = pd.read_csv(path)

    d = df[
        (df["evaluated"] == 1)
        & (df["status"] == "success")
    ].copy()

    print()
    print("=" * 70)
    print("POLY CLEAR GEOMETRY DIAGNOSTIC")
    print("=" * 70)

    if len(d) == 0:
        print("No successful Clear frames.")
        return

    gt = d["lane_gt_x"].astype(float)
    left = d["poly_left_x"].astype(float)
    right = d["poly_right_x"].astype(float)
    pred = d["poly_pred_x"].astype(float)

    d["signed_error"] = pred - gt
    d["abs_error"] = abs(pred - gt)
    d["left_offset"] = gt - left
    d["right_offset"] = right - gt
    d["width"] = right - left
    d["straddle"] = (left < gt) & (gt < right)

    print(f"Successful frames: {len(d)}")

    print("\nCENTER ERROR")
    print(f"Mean signed error: {d['signed_error'].mean():.3f} px")
    print(f"Mean absolute error (success only): {d['abs_error'].mean():.3f} px")
    print(f"Median absolute error: {d['abs_error'].median():.3f} px")

    print("\nGT / BOUNDARY MOTION")
    print(f"GT std:    {gt.std():.3f} px")
    print(f"Left std:  {left.std():.3f} px")
    print(f"Right std: {right.std():.3f} px")

    print("\nTRACKING CORRELATION WITH GT CENTER")
    print(f"Left  vs GT: {left.corr(gt):.3f}")
    print(f"Right vs GT: {right.corr(gt):.3f}")

    print("\nOFFSETS FROM GT CENTER")
    print(
        f"Left  mean/std: "
        f"{d['left_offset'].mean():.3f} / "
        f"{d['left_offset'].std():.3f} px"
    )
    print(
        f"Right mean/std: "
        f"{d['right_offset'].mean():.3f} / "
        f"{d['right_offset'].std():.3f} px"
    )

    print("\nSTRADDLE")
    print(
        f"{d['straddle'].sum()}/{len(d)} "
        f"= {100*d['straddle'].mean():.2f}%"
    )

    print("\nWIDTH")
    width_mean = d["width"].mean()
    width_std = d["width"].std()

    print(f"Mean:   {width_mean:.3f} px")
    print(f"Median: {d['width'].median():.3f} px")
    print(f"Std:    {width_std:.3f} px")
    print(
        f"Coefficient of variation: "
        f"{width_std / width_mean:.3f}"
    )

    print("\nWIDTH VS CENTER ERROR")
    print(
        "Pearson r:",
        d["width"].corr(d["abs_error"])
    )

    # IMPORTANT:
    # Admission error uses failures too, unlike the success-only geometry metrics.
    all_eval = df[df["evaluated"] == 1].copy()
    penalized_mean = all_eval["error_px"].astype(float).mean()
    ratio = penalized_mean / TRIVIAL_CLEAR_ERROR_PX

    print("\nCLEAR SANITY GUARD")
    print(f"Poly penalized mean error: {penalized_mean:.3f} px")
    print(f"Constant-640 baseline:     {TRIVIAL_CLEAR_ERROR_PX:.3f} px")
    print(f"Error ratio Poly/trivial:  {ratio:.3f}")

    if ratio <= 0.75:
        label = "STRONG numerical improvement"
    elif ratio < 1.0:
        label = "MARGINAL improvement — do not auto-accept"
    else:
        label = "FAILS trivial-center guard"

    print(f"Numerical status: {label}")

    print()
    print("Do NOT accept from the ratio alone.")
    print("Also inspect: signed bias, BOTH GT correlations,")
    print("offset balance, straddle rate, width CoV, and debug overlays.")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.all:
        names = list(DATASETS.keys())
    else:
        names = ["clear"]

    summaries = []

    for name in names:
        summaries.append(
            process_dataset(
                name,
                DATASETS[name],
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = OUTPUT_ROOT / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print()
    print("=" * 60)
    print("POLYNOMIAL LANE EVALUATION COMPLETE")
    print(f"Summary: {summary_path}")
    print(f"Debug images: {DEBUG_ROOT}")
    print("=" * 60)

    # Always run the Clear gate when Clear was processed.
    if "clear" in names:
        print_clear_gate()

    if not args.all:
        print()
        print("CLEAR-ONLY RUN COMPLETE.")
        print("Do not run fog yet.")
        print("First decide whether this detector passes the Clear gate.")
        print("If accepted, freeze parameters, then run:")
        print("    python lane/poly_lane.py --all")


if __name__ == "__main__":
    main()
