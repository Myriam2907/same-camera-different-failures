#!/usr/bin/env python3
"""
RF-DETR-L STOP-sign baseline for the CARLA fog sweep.

Main experiment:
- RF-DETR-Large, pretrained COCO checkpoint
- native RF-DETR-L inference resolution: 704 x 704
- raw confidence floor: 0.001 so low-confidence STOP candidates are saved
- operating confidence threshold: 0.50 (RF-DETR predict default)
- project correctness rule: IoU >= 0.50 against CARLA visual STOP-sign GT

Important:
- RF-DETR COCO class IDs can be represented differently across APIs/releases.
  We therefore identify STOP-sign detections by the runtime class-name metadata,
  NOT by a hard-coded class integer.
- We save every raw STOP candidate, including candidates below the operating
  threshold, so later confidence calibration can be done without rerunning
  inference.
"""

from pathlib import Path
import argparse
import shutil

import cv2
import numpy as np
import pandas as pd
import torch

from rfdetr import RFDETRLarge
from rfdetr.assets.coco_classes import COCO_CLASSES


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "RF-DETR-Large"

# Official native checkpoint resolution for RF-DETR-L.
NATIVE_RESOLUTION = 704

# Main experiment uses the native checkpoint resolution.
INFERENCE_SHAPE = (NATIVE_RESOLUTION, NATIVE_RESOLUTION)

# Ask RF-DETR for very low-confidence candidates, then apply the
# operating threshold ourselves. This preserves self-confidence
# values on frames where the final detector abstains.
RAW_CONF_FLOOR = 0.001

# RF-DETR predict() default threshold. This is a fixed pilot operating
# threshold, NOT a Clear-calibrated threshold.
OPERATING_CONF_THRESHOLD = 0.50

CORRECT_IOU_THRESHOLD = 0.50

EXPECTED_CLASS_NAME = "stop sign"

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

OUTPUT_ROOT = Path("results/sign_rfdetr_iou")
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
            "Load RF-DETR-L, verify STOP-sign class-name resolution, "
            "run one clear frame, print candidates, then exit."
        ),
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def normalize_label(value):
    return str(value).strip().lower().replace("_", " ")


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


def class_name_for_detection(detections, index):
    """
    Prefer RF-DETR's runtime class-name metadata.

    This avoids assuming that class_id uses dense 0..79 indexing or sparse
    original COCO category IDs.
    """

    data = getattr(detections, "data", None)

    if isinstance(data, dict) and "class_name" in data:
        names = data["class_name"]

        if len(names) > index:
            value = names[index]

            if value is not None:
                return str(value)

    # Fallback for pretrained COCO checkpoints.
    class_id = int(detections.class_id[index])

    if class_id in COCO_CLASSES:
        return str(COCO_CLASSES[class_id])

    # Some releases/tools may expose dense COCO positions.
    ordered_names = [
        name
        for _, name in sorted(COCO_CLASSES.items())
    ]

    if 0 <= class_id < len(ordered_names):
        return str(ordered_names[class_id])

    return f"UNKNOWN_CLASS_{class_id}"


def verify_coco_stop_mapping():
    """
    Verify that the installed RF-DETR package knows a STOP-sign COCO class.
    Do not assume the runtime integer ID.
    """

    matches = [
        (class_id, class_name)
        for class_id, class_name in COCO_CLASSES.items()
        if normalize_label(class_name) == EXPECTED_CLASS_NAME
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Could not uniquely resolve 'stop sign' in RF-DETR COCO_CLASSES. "
            f"Matches: {matches}"
        )

    class_id, class_name = matches[0]

    print(
        f"COCO_CLASSES runtime mapping contains "
        f"{class_id}: {class_name!r}"
    )

    print(
        "STOP class-name mapping verification: PASS"
    )

    return int(class_id)


def run_raw_stop_predictions(model, image_path):
    """
    Run RF-DETR at RAW_CONF_FLOOR and keep only candidates whose runtime
    class name resolves to 'stop sign'.

    The raw floor is not the operating threshold.
    """

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise RuntimeError(
            f"Could not read {image_path}"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    detections = model.predict(
        image_rgb,
        threshold=RAW_CONF_FLOOR,
        shape=INFERENCE_SHAPE,
        include_source_image=False,
    )

    candidates = []

    if detections is None:
        return candidates

    class_ids = getattr(
        detections,
        "class_id",
        None,
    )

    confidences = getattr(
        detections,
        "confidence",
        None,
    )

    xyxy = getattr(
        detections,
        "xyxy",
        None,
    )

    if (
        class_ids is None
        or confidences is None
        or xyxy is None
    ):
        raise RuntimeError(
            "Unexpected RF-DETR detection object: expected "
            ".class_id, .confidence, and .xyxy fields."
        )

    for index in range(len(class_ids)):
        class_name = class_name_for_detection(
            detections,
            index,
        )

        if normalize_label(class_name) != EXPECTED_CLASS_NAME:
            continue

        box = tuple(
            float(v)
            for v in np.asarray(
                xyxy[index]
            ).tolist()
        )

        confidence = float(
            confidences[index]
        )

        candidates.append({
            "box": box,
            "confidence": confidence,
            "class_id": int(
                class_ids[index]
            ),
            "class_name": class_name,
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
    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise RuntimeError(
            f"Could not read {image_path}"
        )

    vis = image.copy()

    gx1, gy1, gx2, gy2 = [
        int(round(v))
        for v in gt_box
    ]

    # CARLA GT = magenta
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
        x1, y1, x2, y2 = [
            int(round(v))
            for v in candidate["box"]
        ]

        confidence = candidate[
            "confidence"
        ]

        iou_value = candidate[
            "iou"
        ]

        accepted = candidate[
            "accepted"
        ]

        if not accepted:
            color = (160, 160, 160)
        elif iou_value >= CORRECT_IOU_THRESHOLD:
            color = (0, 255, 0)
            accepted_exists = True
            accepted_correct = True
        else:
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
            f"RF-DETR {confidence:.3f} "
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
        status = (
            f"ABSTAIN @ "
            f"{OPERATING_CONF_THRESHOLD:.2f}"
        )

    max_raw_conf = (
        max(
            c["confidence"]
            for c in candidates
        )
        if candidates
        else 0.0
    )

    header = (
        f"{status} | raw_max_conf={max_raw_conf:.3f} | "
        f"route={route_s:+.1f}m | "
        f"sign={distance_to_sign:.2f}m"
    )

    cv2.rectangle(
        vis,
        (0, 0),
        (1120, 55),
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

def smoke_test(model):
    root = Path(
        DATASETS["clear_drift"]["root"]
    )

    csv_path = root / "frames.csv"

    metadata = pd.read_csv(
        csv_path
    )

    validate_metadata_columns(
        metadata,
        csv_path,
    )

    preferred_sample = 140

    matching = metadata[
        metadata["sample_id"]
        == preferred_sample
    ]

    if len(matching) == 1:
        row = matching.iloc[0]
    else:
        visible = metadata[
            metadata["sign_in_view"] == 1
        ]

        if len(visible) == 0:
            raise RuntimeError(
                "No visible STOP-sign frame "
                "available for smoke test."
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

    gt_box = gt_box_from_row(
        row
    )

    if gt_box is None:
        raise RuntimeError(
            "Smoke-test frame has no valid GT box."
        )

    candidates = (
        run_raw_stop_predictions(
            model,
            image_path,
        )
    )

    print()
    print("=" * 70)
    print("RF-DETR SMOKE TEST")
    print("=" * 70)

    print(
        f"Image: {image_path}"
    )

    print(
        f"Inference shape: "
        f"{INFERENCE_SHAPE[0]}x"
        f"{INFERENCE_SHAPE[1]}"
    )

    print(
        f"Route: "
        f"{float(row['route_s_m']):+.1f} m"
    )

    print(
        "Sign distance: "
        f"{float(row['distance_to_sign_m']):.2f} m"
    )

    print(
        f"Raw STOP candidates >= "
        f"{RAW_CONF_FLOOR}: "
        f"{len(candidates)}"
    )

    if not candidates:
        print(
            "No raw STOP candidate found."
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
                f"class_id={candidate['class_id']}, "
                f"class_name={candidate['class_name']!r}, "
                f"conf={candidate['confidence']:.4f}, "
                f"IoU={iou_value:.4f}, "
                f"passes_{OPERATING_CONF_THRESHOLD:.2f}="
                f"{candidate['confidence'] >= OPERATING_CONF_THRESHOLD}"
            )

    print()
    print(
        "If you reached this point, RF-DETR-L loaded, "
        "class-name resolution worked, and inference succeeded."
    )

    print("=" * 70)


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    model,
    dataset_name,
    dataset_info,
):
    root = Path(
        dataset_info["root"]
    )

    csv_path = (
        root
        / "frames.csv"
    )

    frame_dir = (
        root
        / "frames"
    )

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
                model,
                image_path,
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

            pred_box = candidate[
                "box"
            ]

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

            evaluated = {
                **candidate,
                "iou": iou_value,
                "accepted": accepted,
                "correct_by_iou": (
                    correct_by_iou
                ),
            }

            evaluated_candidates.append(
                evaluated
            )

            box_rows.append({
                "dataset": dataset_name,
                "severity": dataset_info["severity"],
                "sample_id": sample_id,
                "route_s_m": route_s,
                "distance_to_sign_m": distance_to_sign,
                "candidate_index": candidate_index,
                "runtime_class_id": candidate["class_id"],
                "runtime_class_name": candidate["class_name"],
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

            "raw_candidate_count": raw_candidate_count,
            "max_raw_stop_confidence": max_raw_confidence,
            "best_raw_iou": best_raw_iou,

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
            "false_positive_boxes": false_positive_boxes,
            "best_accepted_iou": best_accepted_iou,
            "max_accepted_confidence": max_accepted_confidence,
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

    n = len(
        frame_df
    )

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
        100.0
        * detected_frames
        / n
        if n
        else 0.0
    )

    correct_rate = (
        100.0
        * correct_frames
        / n
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
        "native_resolution": NATIVE_RESOLUTION,
        "inference_height": INFERENCE_SHAPE[0],
        "inference_width": INFERENCE_SHAPE[1],
        "raw_conf_floor": RAW_CONF_FLOOR,
        "operating_conf_threshold": OPERATING_CONF_THRESHOLD,
        "threshold_source": (
            "RF-DETR default pilot value; "
            "NOT Clear-calibrated"
        ),
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
        "first_correct_detection_sample": first_sample,
        "first_correct_detection_route_m": first_route,
        "first_correct_detection_distance_m": first_distance,
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

    print("=" * 70)
    print(
        "RF-DETR-L STOP-SIGN IoU BASELINE"
    )
    print("=" * 70)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    print(
        f"Native inference resolution: "
        f"{NATIVE_RESOLUTION}x"
        f"{NATIVE_RESOLUTION}"
    )

    print(
        f"Experiment inference shape: "
        f"{INFERENCE_SHAPE[0]}x"
        f"{INFERENCE_SHAPE[1]}"
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
        f"Correctness rule: "
        f"IoU >= {CORRECT_IOU_THRESHOLD}"
    )

    print()
    print(
        "Verifying installed COCO class map..."
    )

    verify_coco_stop_mapping()

    print()
    print(
        "Loading/downloading RF-DETR-Large..."
    )

    model = RFDETRLarge()

    print(
        "RF-DETR-Large loaded successfully."
    )

    if args.smoke_test:
        smoke_test(
            model
        )
        return

    if OUTPUT_ROOT.exists():
        print(
            f"Deleting old RF-DETR results: "
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
            "WARNING: RF-DETR never correctly detected "
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
        "RF-DETR SUMMARY"
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
        "RF-DETR IoU VALIDATION COMPLETE"
    )

    print(
        f"Summary: {summary_path}"
    )

    print(
        f"Per-frame + raw-box CSVs: "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Debug images: "
        f"{DEBUG_ROOT}"
    )

    print()
    print(
        "Magenta = CARLA STOP-sign GT bbox"
    )

    print(
        "Green = accepted RF-DETR box, IoU >= 0.50"
    )

    print(
        "Red = accepted RF-DETR box, IoU < 0.50"
    )

    print(
        f"Gray = raw RF-DETR STOP candidate "
        f"below {OPERATING_CONF_THRESHOLD:.2f}"
    )

    print()
    print(
        "NOTE 1: RF-DETR-L is evaluated at its native "
        "704x704 checkpoint resolution."
    )

    print(
        "NOTE 2: YOLO26s was evaluated at imgsz=1280, "
        "so absolute first-detection ranges are not a "
        "pure architecture comparison."
    )

    print(
        "NOTE 3: 0.50 is RF-DETR's fixed default pilot "
        "confidence threshold, NOT a Clear-calibrated threshold."
    )

    print(
        "Raw candidates are saved so confidence calibration "
        "can be performed later without rerunning inference."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
