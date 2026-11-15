#!/usr/bin/env python3
"""
YOLO STOP-sign baseline for the CARLA fog sweep.

Important design choices:
- Uses an Ultralytics COCO-pretrained YOLO detector.
- Verifies that class index 11 is exactly "stop sign" before any sweep.
- Runs inference down to a very low RAW_CONF_FLOOR, then applies the
  operating threshold (0.25) ourselves. This lets us save YOLO's own
  per-frame confidence even when no box passes the operating threshold.
- Saves a box-level CSV with every raw STOP-sign candidate so later
  confidence-threshold calibration can be done WITHOUT rerunning YOLO.
- Correctness rule remains the project rule: IoU >= 0.50 against CARLA GT.
- Same fixed inference configuration is used for all fog severities.
"""

from pathlib import Path
import argparse
import shutil

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

# Verified Ultralytics detection weight name.
MODEL_NAME = "yolo26s.pt"

# COCO-80 class index. We ASSERT the model label map matches this.
EXPECTED_STOP_CLASS_ID = 11
EXPECTED_STOP_CLASS_NAME = "stop sign"

# Run model at a low floor so we can observe low-confidence candidates.
# This is NOT the operating detection threshold.
RAW_CONF_FLOOR = 0.001

# Pilot operating threshold.
# We keep the Ultralytics standard/default operating value for this first run.
# It is saved explicitly in the output so it is never confused with a
# Clear-calibrated threshold.
OPERATING_CONF_THRESHOLD = 0.25

NMS_IOU_THRESHOLD = 0.70
IMAGE_SIZE = 1280
CORRECT_IOU_THRESHOLD = 0.50
MAX_DETECTIONS = 50

DATASETS = {
    "clear_drift": {
        "root": "data/recordings/stop3_clear_0_drift_0p75",
        "severity": 0,
    },
    "fog15_drift": {
        "root": "data/recordings/stop3_fog_15_drift_0p75",
        "severity": 15,
    },
    "fog30_drift": {
        "root": "data/recordings/stop3_fog_30_drift_0p75",
        "severity": 30,
    },
    "fog45_drift": {
        "root": "data/recordings/stop3_fog_45_drift_0p75",
        "severity": 45,
    },
    "fog60_drift": {
        "root": "data/recordings/stop3_fog_60_drift_0p75",
        "severity": 60,
    },
    "fog75_drift": {
        "root": "data/recordings/stop3_fog_75_drift_0p75",
        "severity": 75,
    },
    "fog90_drift": {
        "root": "data/recordings/stop3_fog_90_drift_0p75",
        "severity": 90,
    },
}

OUTPUT_ROOT = Path("results/sign_yolo_iou")
DEBUG_ROOT = OUTPUT_ROOT / "debug"

REQUIRED_COLUMNS = {
    "sample_id",
    "route_s_m",
    "distance_to_sign_m",
    "sign_in_view",
    "sign_bbox_x1",
    "sign_bbox_y1",
    "sign_bbox_x2",
    "sign_bbox_y2",
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Load weights, verify the COCO STOP-sign class, run one "
            "clear frame, print candidates, then exit."
        ),
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def validate_metadata_columns(df, csv_path):
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))

    if missing:
        raise RuntimeError(
            f"{csv_path} is missing required columns: {missing}"
        )


def gt_box_from_row(row):
    values = [
        row["sign_bbox_x1"],
        row["sign_bbox_y1"],
        row["sign_bbox_x2"],
        row["sign_bbox_y2"],
    ]

    if any(pd.isna(v) for v in values):
        return None

    x1, y1, x2, y2 = map(float, values)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def box_iou(box_a, box_b):
    if box_a is None or box_b is None:
        return np.nan

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return float(intersection / union)


def normalize_label(label):
    return str(label).strip().lower().replace("_", " ")


def verify_stop_class(model):
    names = model.names

    if isinstance(names, dict):
        if EXPECTED_STOP_CLASS_ID not in names:
            raise RuntimeError(
                f"Model does not contain class index "
                f"{EXPECTED_STOP_CLASS_ID}. names={names}"
            )

        actual_name = names[EXPECTED_STOP_CLASS_ID]

    else:
        if len(names) <= EXPECTED_STOP_CLASS_ID:
            raise RuntimeError(
                f"Model has only {len(names)} classes; cannot access "
                f"class {EXPECTED_STOP_CLASS_ID}."
            )

        actual_name = names[EXPECTED_STOP_CLASS_ID]

    print(
        f"model.names[{EXPECTED_STOP_CLASS_ID}] = "
        f"{actual_name!r}"
    )

    if normalize_label(actual_name) != EXPECTED_STOP_CLASS_NAME:
        raise RuntimeError(
            "STOP class verification FAILED. "
            f"Expected class {EXPECTED_STOP_CLASS_ID} to be "
            f"{EXPECTED_STOP_CLASS_NAME!r}, got {actual_name!r}. "
            "Aborting before any sweep."
        )

    print("STOP class verification: PASS")

    return EXPECTED_STOP_CLASS_ID


def run_raw_stop_predictions(
    model,
    image_path,
    stop_class_id,
    device,
):
    """
    Run at RAW_CONF_FLOOR, not at 0.25.

    This preserves low-confidence STOP candidates so we can save YOLO's
    self-assessment even when the final operating detector abstains.
    """

    prediction = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=RAW_CONF_FLOOR,
        iou=NMS_IOU_THRESHOLD,
        classes=[stop_class_id],
        max_det=MAX_DETECTIONS,
        device=device,
        verbose=False,
    )[0]

    candidates = []

    if prediction.boxes is None or len(prediction.boxes) == 0:
        return candidates

    xyxy = prediction.boxes.xyxy.detach().cpu().numpy()
    confs = prediction.boxes.conf.detach().cpu().numpy()

    for box, confidence in zip(xyxy, confs):
        candidates.append({
            "box": tuple(
                float(v)
                for v in box.tolist()
            ),
            "confidence": float(confidence),
        })

    candidates.sort(
        key=lambda item: item["confidence"],
        reverse=True,
    )

    return candidates


def draw_debug_image(
    image_path,
    output_path,
    gt_box,
    candidates,
    route_s,
    distance_to_sign,
):
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(
            f"Could not read {image_path}"
        )

    vis = image.copy()

    # CARLA ground truth = magenta.
    gx1, gy1, gx2, gy2 = [
        int(round(v))
        for v in gt_box
    ]

    cv2.rectangle(
        vis,
        (gx1, gy1),
        (gx2, gy2),
        (255, 0, 255),
        2,
    )

    accepted_exists = False
    accepted_correct = False

    for candidate in candidates:
        box = candidate["box"]
        confidence = candidate["confidence"]
        iou_value = candidate["iou"]
        accepted = candidate["accepted"]

        x1, y1, x2, y2 = [
            int(round(v))
            for v in box
        ]

        if not accepted:
            # Raw candidate below 0.25 = gray.
            color = (160, 160, 160)
        elif iou_value >= CORRECT_IOU_THRESHOLD:
            # Accepted and correct = green.
            color = (0, 255, 0)
            accepted_exists = True
            accepted_correct = True
        else:
            # Accepted but wrong = red.
            color = (0, 0, 255)
            accepted_exists = True

        cv2.rectangle(
            vis,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        label = (
            f"{confidence:.3f} "
            f"IoU={iou_value:.2f}"
        )

        cv2.putText(
            vis,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if accepted_correct:
        status = "CORRECT"
    elif accepted_exists:
        status = "FALSE POSITIVE"
    else:
        status = "ABSTAIN @ 0.25"

    max_raw_conf = (
        max(c["confidence"] for c in candidates)
        if candidates
        else 0.0
    )

    header = (
        f"{status} | raw_max_conf={max_raw_conf:.3f} | "
        f"route={route_s:+.1f}m | sign={distance_to_sign:.2f}m"
    )

    cv2.rectangle(
        vis,
        (0, 0),
        (1100, 55),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        vis,
        header,
        (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(output_path),
        vis,
    ):
        raise RuntimeError(
            f"Could not save {output_path}"
        )


# ============================================================
# SMOKE TEST
# ============================================================

def smoke_test(model, stop_class_id, device):
    """
    Use a close, clear frame where the sign is visible.
    This confirms:
      1) weights loaded,
      2) class label map is correct,
      3) inference works,
      4) raw confidences are accessible.
    """

    root = Path(
        DATASETS["clear_drift"]["root"]
    )

    csv_path = root / "frames.csv"
    metadata = pd.read_csv(csv_path)

    validate_metadata_columns(
        metadata,
        csv_path,
    )

    preferred_sample = 140

    matching = metadata[
        metadata["sample_id"] == preferred_sample
    ]

    if len(matching) == 1:
        row = matching.iloc[0]
    else:
        visible = metadata[
            metadata["sign_in_view"] == 1
        ]

        if len(visible) == 0:
            raise RuntimeError(
                "No visible STOP-sign frame available "
                "for smoke test."
            )

        row = visible.iloc[-1]
        preferred_sample = int(
            row["sample_id"]
        )

    image_path = (
        root
        / "frames"
        / f"{preferred_sample:04d}.png"
    )

    gt_box = gt_box_from_row(row)

    if gt_box is None:
        raise RuntimeError(
            "Smoke-test frame has no valid GT box."
        )

    candidates = run_raw_stop_predictions(
        model=model,
        image_path=image_path,
        stop_class_id=stop_class_id,
        device=device,
    )

    print()
    print("=" * 70)
    print("SMOKE TEST")
    print("=" * 70)
    print(f"Image: {image_path}")
    print(
        f"Route: {float(row['route_s_m']):+.1f} m"
    )
    print(
        "Sign distance: "
        f"{float(row['distance_to_sign_m']):.2f} m"
    )
    print(
        f"Raw STOP candidates >= {RAW_CONF_FLOOR}: "
        f"{len(candidates)}"
    )

    if not candidates:
        print(
            "No raw STOP candidate found on smoke-test frame."
        )
    else:
        for index, candidate in enumerate(
            candidates,
            start=1,
        ):
            iou_value = box_iou(
                candidate["box"],
                gt_box,
            )

            print(
                f"  #{index}: "
                f"conf={candidate['confidence']:.4f}, "
                f"IoU={iou_value:.4f}, "
                f"passes_0.25="
                f"{candidate['confidence'] >= OPERATING_CONF_THRESHOLD}"
            )

    print()
    print(
        "If you reached this point, model download/load, "
        "class-map verification, and inference all succeeded."
    )
    print("=" * 70)


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    model,
    stop_class_id,
    device,
    dataset_name,
    dataset_info,
):
    root = Path(dataset_info["root"])
    csv_path = root / "frames.csv"
    frame_dir = root / "frames"

    if not csv_path.exists():
        raise RuntimeError(
            f"Missing CSV: {csv_path}"
        )

    metadata = pd.read_csv(
        csv_path
    )

    validate_metadata_columns(
        metadata,
        csv_path,
    )

    # Match the existing STOP evaluation:
    # sign is visually in frame, and ego is on the approach.
    eval_df = metadata[
        (metadata["sign_in_view"] == 1)
        & (metadata["route_s_m"] <= 0.0)
    ].copy()

    eval_df = eval_df.sort_values(
        "sample_id"
    )

    frame_rows = []
    box_rows = []

    debug_dir = (
        DEBUG_ROOT
        / dataset_name
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        f"Processing: {dataset_name}"
    )
    print("=" * 70)

    for _, row in eval_df.iterrows():
        sample_id = int(
            row["sample_id"]
        )

        route_s = float(
            row["route_s_m"]
        )

        distance_to_sign = float(
            row["distance_to_sign_m"]
        )

        image_path = (
            frame_dir
            / f"{sample_id:04d}.png"
        )

        if not image_path.exists():
            raise RuntimeError(
                f"Missing image: {image_path}"
            )

        gt_box = gt_box_from_row(
            row
        )

        if gt_box is None:
            raise RuntimeError(
                f"Invalid GT bbox for "
                f"sample {sample_id}"
            )

        raw_candidates = (
            run_raw_stop_predictions(
                model=model,
                image_path=image_path,
                stop_class_id=stop_class_id,
                device=device,
            )
        )

        evaluated_candidates = []

        for candidate_index, candidate in enumerate(
            raw_candidates,
            start=1,
        ):
            confidence = float(
                candidate["confidence"]
            )

            pred_box = candidate["box"]

            iou_value = box_iou(
                pred_box,
                gt_box,
            )

            accepted = bool(
                confidence
                >= OPERATING_CONF_THRESHOLD
            )

            correct_by_iou = bool(
                iou_value
                >= CORRECT_IOU_THRESHOLD
            )

            item = {
                "box": pred_box,
                "confidence": confidence,
                "iou": iou_value,
                "accepted": accepted,
                "correct_by_iou": correct_by_iou,
            }

            evaluated_candidates.append(
                item
            )

            box_rows.append({
                "dataset": dataset_name,
                "severity": dataset_info["severity"],
                "sample_id": sample_id,
                "route_s_m": route_s,
                "distance_to_sign_m": distance_to_sign,
                "candidate_index": candidate_index,
                "confidence": confidence,
                "x1": pred_box[0],
                "y1": pred_box[1],
                "x2": pred_box[2],
                "y2": pred_box[3],
                "iou_with_gt": iou_value,
                "passes_operating_threshold": int(
                    accepted
                ),
                "correct_by_iou": int(
                    correct_by_iou
                ),
            })

        accepted_candidates = [
            c
            for c in evaluated_candidates
            if c["accepted"]
        ]

        raw_candidate_count = len(
            evaluated_candidates
        )

        accepted_count = len(
            accepted_candidates
        )

        max_raw_confidence = (
            max(
                c["confidence"]
                for c in evaluated_candidates
            )
            if evaluated_candidates
            else 0.0
        )

        max_accepted_confidence = (
            max(
                c["confidence"]
                for c in accepted_candidates
            )
            if accepted_candidates
            else np.nan
        )

        best_raw_iou = (
            max(
                c["iou"]
                for c in evaluated_candidates
            )
            if evaluated_candidates
            else 0.0
        )

        best_accepted_iou = (
            max(
                c["iou"]
                for c in accepted_candidates
            )
            if accepted_candidates
            else 0.0
        )

        detected = bool(
            accepted_count > 0
        )

        correct = bool(
            any(
                c["iou"]
                >= CORRECT_IOU_THRESHOLD
                for c in accepted_candidates
            )
        )

        false_positive_frame = bool(
            detected and not correct
        )

        false_positive_boxes = int(
            sum(
                1
                for c in accepted_candidates
                if c["iou"]
                < CORRECT_IOU_THRESHOLD
            )
        )

        # Confidence of the accepted box that best overlaps GT.
        if accepted_candidates:
            best_iou_candidate = max(
                accepted_candidates,
                key=lambda c: c["iou"],
            )

            confidence_of_best_iou_box = float(
                best_iou_candidate[
                    "confidence"
                ]
            )
        else:
            confidence_of_best_iou_box = np.nan

        frame_rows.append({
            "dataset": dataset_name,
            "severity": dataset_info["severity"],
            "sample_id": sample_id,
            "route_s_m": route_s,
            "distance_to_sign_m": distance_to_sign,

            # Raw YOLO self-assessment, BEFORE 0.25 threshold.
            "raw_candidate_count": raw_candidate_count,
            "max_raw_stop_confidence": max_raw_confidence,
            "best_raw_iou": best_raw_iou,

            # Operating detector at threshold 0.25.
            "accepted_detection_count": accepted_count,
            "detected_at_operating_threshold": int(
                detected
            ),
            "correct_iou_detection": int(
                correct
            ),
            "false_positive_frame": int(
                false_positive_frame
            ),
            "false_positive_boxes": (
                false_positive_boxes
            ),
            "best_accepted_iou": (
                best_accepted_iou
            ),
            "max_accepted_confidence": (
                max_accepted_confidence
            ),
            "confidence_of_best_iou_box": (
                confidence_of_best_iou_box
            ),

            "gt_x1": gt_box[0],
            "gt_y1": gt_box[1],
            "gt_x2": gt_box[2],
            "gt_y2": gt_box[3],
        })

        draw_debug_image(
            image_path=image_path,
            output_path=(
                debug_dir
                / f"{sample_id:04d}.jpg"
            ),
            gt_box=gt_box,
            candidates=evaluated_candidates,
            route_s=route_s,
            distance_to_sign=distance_to_sign,
        )

    frame_df = pd.DataFrame(
        frame_rows
    )

    box_df = pd.DataFrame(
        box_rows
    )

    frame_path = (
        OUTPUT_ROOT
        / f"{dataset_name}_frames.csv"
    )

    box_path = (
        OUTPUT_ROOT
        / f"{dataset_name}_raw_boxes.csv"
    )

    frame_df.to_csv(
        frame_path,
        index=False,
    )

    box_df.to_csv(
        box_path,
        index=False,
    )

    n = len(frame_df)

    detected_frames = int(
        frame_df[
            "detected_at_operating_threshold"
        ].sum()
    )

    correct_frames = int(
        frame_df[
            "correct_iou_detection"
        ].sum()
    )

    fp_frames = int(
        frame_df[
            "false_positive_frame"
        ].sum()
    )

    fp_boxes = int(
        frame_df[
            "false_positive_boxes"
        ].sum()
    )

    detection_rate = (
        100.0 * detected_frames / n
        if n
        else 0.0
    )

    correct_rate = (
        100.0 * correct_frames / n
        if n
        else 0.0
    )

    mean_max_raw_conf = (
        float(
            frame_df[
                "max_raw_stop_confidence"
            ].mean()
        )
        if n
        else np.nan
    )

    accepted_df = frame_df[
        frame_df[
            "detected_at_operating_threshold"
        ] == 1
    ]

    mean_best_accepted_iou = (
        float(
            accepted_df[
                "best_accepted_iou"
            ].mean()
        )
        if len(accepted_df) > 0
        else np.nan
    )

    mean_max_accepted_conf = (
        float(
            accepted_df[
                "max_accepted_confidence"
            ].mean()
        )
        if len(accepted_df) > 0
        else np.nan
    )

    correct_df = frame_df[
        frame_df[
            "correct_iou_detection"
        ] == 1
    ].sort_values(
        "sample_id"
    )

    if len(correct_df) > 0:
        first = correct_df.iloc[0]

        first_sample = int(
            first["sample_id"]
        )

        first_route = float(
            first["route_s_m"]
        )

        first_distance = float(
            first[
                "distance_to_sign_m"
            ]
        )

    else:
        first_sample = np.nan
        first_route = np.nan
        first_distance = 0.0

    summary = {
        "dataset": dataset_name,
        "severity": dataset_info["severity"],
        "model": MODEL_NAME,
        "stop_class_id": stop_class_id,
        "stop_class_name": EXPECTED_STOP_CLASS_NAME,
        "image_size": IMAGE_SIZE,
        "raw_conf_floor": RAW_CONF_FLOOR,
        "operating_conf_threshold": OPERATING_CONF_THRESHOLD,
        "threshold_source": (
            "Ultralytics default pilot value; "
            "NOT Clear-calibrated"
        ),
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "correct_iou_threshold": CORRECT_IOU_THRESHOLD,
        "evaluable_approach_frames": n,
        "detection_frames": detected_frames,
        "detection_rate_pct": detection_rate,
        "correct_iou_frames": correct_frames,
        "correct_detection_rate_pct": correct_rate,
        "false_positive_frames": fp_frames,
        "false_positive_boxes": fp_boxes,
        "mean_max_raw_stop_confidence_all_frames": (
            mean_max_raw_conf
        ),
        "mean_best_accepted_iou": (
            mean_best_accepted_iou
        ),
        "mean_max_accepted_confidence": (
            mean_max_accepted_conf
        ),
        "first_correct_detection_sample": (
            first_sample
        ),
        "first_correct_detection_route_m": (
            first_route
        ),
        "first_correct_detection_distance_m": (
            first_distance
        ),
    }

    print(
        f"Evaluable approach frames: {n}"
    )

    print(
        f"Accepted detection frames "
        f"(conf >= {OPERATING_CONF_THRESHOLD}): "
        f"{detected_frames}"
    )

    print(
        f"Detection rate: "
        f"{detection_rate:.2f}%"
    )

    print(
        f"Correct IoU detections: "
        f"{correct_frames}"
    )

    print(
        f"Correct detection rate: "
        f"{correct_rate:.2f}%"
    )

    print(
        f"False-positive frames: "
        f"{fp_frames}"
    )

    print(
        f"False-positive boxes: "
        f"{fp_boxes}"
    )

    print(
        "Mean max RAW STOP confidence "
        f"(all frames): "
        f"{mean_max_raw_conf:.4f}"
    )

    if np.isfinite(
        mean_best_accepted_iou
    ):
        print(
            "Mean best IoU on accepted frames: "
            f"{mean_best_accepted_iou:.4f}"
        )
    else:
        print(
            "Mean best IoU on accepted frames: N/A"
        )

    if len(correct_df) > 0:
        print(
            f"First CORRECT detection sample: "
            f"{first_sample}"
        )

        print(
            f"First CORRECT detection route: "
            f"{first_route:.1f}"
        )

        print(
            "First CORRECT detection distance: "
            f"{first_distance:.2f} m"
        )
    else:
        print(
            "First CORRECT detection: NONE"
        )

        print(
            "First CORRECT detection distance: "
            "0.00 m"
        )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    device = (
        0
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print(
        "YOLO STOP-SIGN IoU BASELINE"
    )
    print("=" * 70)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Image size: {IMAGE_SIZE}"
    )

    print(
        f"Raw confidence floor: "
        f"{RAW_CONF_FLOOR}"
    )

    print(
        f"Operating confidence threshold: "
        f"{OPERATING_CONF_THRESHOLD}"
    )

    print(
        f"NMS IoU threshold: "
        f"{NMS_IOU_THRESHOLD}"
    )

    print(
        f"Correctness rule: "
        f"IoU >= {CORRECT_IOU_THRESHOLD}"
    )

    print()
    print(
        "Loading/downloading model..."
    )

    model = YOLO(
        MODEL_NAME
    )

    print(
        "Model loaded successfully."
    )

    stop_class_id = (
        verify_stop_class(
            model
        )
    )

    if args.smoke_test:
        smoke_test(
            model=model,
            stop_class_id=stop_class_id,
            device=device,
        )
        return

    if OUTPUT_ROOT.exists():
        print(
            f"Deleting old YOLO results: "
            f"{OUTPUT_ROOT}"
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    DEBUG_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for (
        dataset_name,
        dataset_info,
    ) in DATASETS.items():

        summary = process_dataset(
            model=model,
            stop_class_id=stop_class_id,
            device=device,
            dataset_name=dataset_name,
            dataset_info=dataset_info,
        )

        summaries.append(
            summary
        )

    summary_df = pd.DataFrame(
        summaries
    )

    clear_rows = summary_df[
        summary_df["severity"] == 0
    ]

    if len(clear_rows) != 1:
        raise RuntimeError(
            "Expected exactly one Clear dataset "
            "for R_sign normalization."
        )

    d0 = float(
        clear_rows.iloc[0][
            "first_correct_detection_distance_m"
        ]
    )

    if d0 <= 0.0:
        summary_df[
            "R_sign"
        ] = np.nan

        print()
        print(
            "WARNING: YOLO never correctly detected "
            "the STOP sign in Clear, so R_sign "
            "cannot be computed."
        )

    else:
        summary_df[
            "R_sign"
        ] = (
            summary_df[
                "first_correct_detection_distance_m"
            ]
            / d0
        ).clip(
            upper=1.0
        )

    summary_df = (
        summary_df
        .sort_values(
            "severity"
        )
    )

    summary_path = (
        OUTPUT_ROOT
        / "summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "YOLO SUMMARY"
    )
    print("=" * 100)

    display_cols = [
        "severity",
        "detection_rate_pct",
        "correct_detection_rate_pct",
        "false_positive_frames",
        "mean_max_raw_stop_confidence_all_frames",
        "first_correct_detection_distance_m",
        "R_sign",
    ]

    print(
        summary_df[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print("=" * 70)
    print(
        "YOLO IoU VALIDATION COMPLETE"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Per-frame + raw-box CSVs: "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Debug images: {DEBUG_ROOT}"
    )

    print()
    print(
        "Magenta = CARLA STOP-sign GT bbox"
    )

    print(
        "Green = accepted YOLO box, IoU >= 0.50"
    )

    print(
        "Red = accepted YOLO box, IoU < 0.50"
    )

    print(
        "Gray = raw YOLO STOP candidate below 0.25"
    )

    print()
    print(
        "NOTE: 0.25 is a fixed default-value pilot threshold, "
        "NOT a Clear-calibrated threshold."
    )

    print(
        "Raw candidate CSVs were saved so later threshold "
        "calibration can be done without rerunning inference."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
