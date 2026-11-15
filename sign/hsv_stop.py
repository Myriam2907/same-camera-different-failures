import os
import shutil

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_W = 1280
IMAGE_H = 720

# IMPORTANT:
# During the IoU validation stage we evaluate ONLY the new
# laterally-varying Clear recording. Old fog recordings used the
# previous centerline geometry and must not be mixed with this run.
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

OUTPUT_ROOT = "results/sign_hsv_iou"
DEBUG_ROOT = os.path.join(
    OUTPUT_ROOT,
    "debug",
)

IOU_THRESHOLD = 0.50


# ============================================================
# HSV PARAMETERS
# ============================================================

LOWER_RED_1 = np.array(
    [0, 80, 60],
    dtype=np.uint8,
)

UPPER_RED_1 = np.array(
    [10, 255, 255],
    dtype=np.uint8,
)

LOWER_RED_2 = np.array(
    [170, 80, 60],
    dtype=np.uint8,
)

UPPER_RED_2 = np.array(
    [180, 255, 255],
    dtype=np.uint8,
)


# ============================================================
# SHAPE PARAMETERS
# ============================================================

MIN_AREA_PX = 20.0

MIN_ASPECT = 0.65
MAX_ASPECT = 1.35

MIN_VERTICES = 6
MAX_VERTICES = 10

MIN_RED_RATIO = 0.20


# ============================================================
# BROAD RUNTIME SEARCH REGION
# ============================================================

# These are the same runtime detector limits used in the pilot.
# Ground truth is NOT used to select a candidate.
SEARCH_X_MIN = int(
    0.45 * IMAGE_W
)

SEARCH_X_MAX = int(
    0.99 * IMAGE_W
)

SEARCH_Y_MIN = int(
    0.25 * IMAGE_H
)

SEARCH_Y_MAX = int(
    0.90 * IMAGE_H
)


# ============================================================
# OUTPUT
# ============================================================

def reset_output_directory():

    if os.path.exists(
        OUTPUT_ROOT
    ):

        print(
            "Deleting old IoU-validation results:",
            OUTPUT_ROOT,
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    os.makedirs(
        DEBUG_ROOT,
        exist_ok=True,
    )


# ============================================================
# GROUND-TRUTH HELPERS
# ============================================================

GT_COLUMNS = [
    "sign_bbox_x1",
    "sign_bbox_y1",
    "sign_bbox_x2",
    "sign_bbox_y2",
]


def validate_metadata_columns(
    metadata,
    csv_path,
):

    required = [
        "sample_id",
        "weather",
        "severity",
        "route_s_m",
        "distance_to_sign_m",
        "sign_in_view",
        *GT_COLUMNS,
    ]

    missing = [
        column
        for column in required
        if column not in metadata.columns
    ]

    if missing:

        raise RuntimeError(
            "This recording does not contain the new STOP-sign "
            "bounding-box ground truth required for IoU evaluation.\n"
            f"Missing columns: {missing}\n"
            f"CSV: {csv_path}\n"
            "Re-record the Clear validation dataset with the updated "
            "carla/collect_data.py first."
        )


def gt_box_from_row(
    row,
):

    if int(
        row["sign_in_view"]
    ) != 1:

        return None

    values = [
        float(row[column])
        for column in GT_COLUMNS
    ]

    if not np.all(
        np.isfinite(values)
    ):

        return None

    x1, y1, x2, y2 = values

    if (
        x2 <= x1
        or y2 <= y1
    ):

        return None

    return (
        x1,
        y1,
        x2,
        y2,
    )


def candidate_box(
    candidate,
):

    if candidate is None:
        return None

    x1 = float(
        candidate["x"]
    )

    y1 = float(
        candidate["y"]
    )

    x2 = (
        x1
        + float(candidate["w"])
    )

    y2 = (
        y1
        + float(candidate["h"])
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    return (
        x1,
        y1,
        x2,
        y2,
    )


def box_iou(
    box_a,
    box_b,
):

    if (
        box_a is None
        or box_b is None
    ):
        return np.nan

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(
        ax1,
        bx1,
    )

    inter_y1 = max(
        ay1,
        by1,
    )

    inter_x2 = min(
        ax2,
        bx2,
    )

    inter_y2 = min(
        ay2,
        by2,
    )

    inter_w = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_h = max(
        0.0,
        inter_y2 - inter_y1,
    )

    intersection = (
        inter_w
        * inter_h
    )

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - intersection
    )

    if union <= 0.0:
        return 0.0

    return float(
        intersection
        / union
    )


# ============================================================
# RED MASK
# ============================================================

def make_red_mask(
    image,
):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    mask1 = cv2.inRange(
        hsv,
        LOWER_RED_1,
        UPPER_RED_1,
    )

    mask2 = cv2.inRange(
        hsv,
        LOWER_RED_2,
        UPPER_RED_2,
    )

    mask = cv2.bitwise_or(
        mask1,
        mask2,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    return mask


# ============================================================
# SEARCH REGION
# ============================================================

def inside_search_region(
    center_x,
    center_y,
):

    return (
        SEARCH_X_MIN
        <= center_x
        <= SEARCH_X_MAX

        and

        SEARCH_Y_MIN
        <= center_y
        <= SEARCH_Y_MAX
    )


# ============================================================
# HSV CANDIDATES
# ============================================================

def find_stop_candidates(
    mask,
):

    """
    Runtime detector.

    Ground-truth information is deliberately absent here.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates = []

    for contour in contours:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area < MIN_AREA_PX:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if (
            w <= 0
            or h <= 0
        ):
            continue

        center_x = (
            x + w / 2.0
        )

        center_y = (
            y + h / 2.0
        )

        if not inside_search_region(
            center_x,
            center_y,
        ):
            continue

        aspect = (
            float(w)
            / float(h)
        )

        if not (
            MIN_ASPECT
            <= aspect
            <= MAX_ASPECT
        ):
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0.0:
            continue

        approx = cv2.approxPolyDP(
            contour,
            0.03 * perimeter,
            True,
        )

        vertices = len(
            approx
        )

        if not (
            MIN_VERTICES
            <= vertices
            <= MAX_VERTICES
        ):
            continue

        roi_mask = mask[
            y:y + h,
            x:x + w,
        ]

        if roi_mask.size == 0:
            continue

        red_ratio = (
            np.count_nonzero(
                roi_mask
            )
            / float(
                roi_mask.size
            )
        )

        if (
            red_ratio
            < MIN_RED_RATIO
        ):
            continue

        is_convex = bool(
            cv2.isContourConvex(
                approx
            )
        )

        vertex_score = (
            1.0
            / (
                1.0
                + abs(
                    vertices - 8
                )
            )
        )

        convex_score = (
            1.0
            if is_convex
            else 0.75
        )

        size_score = np.sqrt(
            area
        )

        score = (
            red_ratio
            * vertex_score
            * convex_score
            * size_score
        )

        candidates.append({
            "x":
                int(x),

            "y":
                int(y),

            "w":
                int(w),

            "h":
                int(h),

            "center_x":
                float(center_x),

            "center_y":
                float(center_y),

            "area":
                float(area),

            "aspect":
                float(aspect),

            "vertices":
                int(vertices),

            "red_ratio":
                float(red_ratio),

            "convex":
                int(is_convex),

            "score":
                float(score),
        })

    candidates.sort(
        key=lambda candidate:
            candidate["score"],
        reverse=True,
    )

    return candidates


# ============================================================
# RUNTIME DETECTOR
# ============================================================

def detect_stop_sign(
    image,
):

    mask = make_red_mask(
        image
    )

    candidates = (
        find_stop_candidates(
            mask
        )
    )

    if len(candidates) == 0:

        return (
            None,
            mask,
            [],
        )

    # Same pilot rule:
    # runtime output is the highest-scoring candidate.
    return (
        candidates[0],
        mask,
        candidates,
    )


# ============================================================
# DEBUG IMAGE
# ============================================================

def draw_box(
    image,
    box,
    color,
    thickness,
):

    if box is None:
        return

    x1, y1, x2, y2 = box

    cv2.rectangle(
        image,
        (
            int(round(x1)),
            int(round(y1)),
        ),
        (
            int(round(x2)),
            int(round(y2)),
        ),
        color,
        thickness,
    )


def save_debug(
    image,
    mask,
    candidates,
    best,
    gt_box,
    iou_value,
    correct_detection,
    output_path,
    distance_to_sign,
    route_s,
):

    """
    Blue    = broad runtime search region
    Magenta = projected CARLA visual STOP-sign GT box
    Green   = selected candidate has IoU >= 0.50
    Red     = selected candidate is false positive
    Yellow  = other runtime candidates
    """

    vis = image.copy()

    cv2.rectangle(
        vis,
        (
            SEARCH_X_MIN,
            SEARCH_Y_MIN,
        ),
        (
            SEARCH_X_MAX,
            SEARCH_Y_MAX,
        ),
        (
            255,
            0,
            0,
        ),
        2,
    )

    for candidate in candidates:

        if candidate is best:
            continue

        draw_box(
            vis,
            candidate_box(
                candidate
            ),
            (
                0,
                255,
                255,
            ),
            1,
        )

    if gt_box is not None:

        draw_box(
            vis,
            gt_box,
            (
                255,
                0,
                255,
            ),
            3,
        )

    if best is not None:

        if correct_detection:

            selected_color = (
                0,
                255,
                0,
            )

        else:

            selected_color = (
                0,
                0,
                255,
            )

        draw_box(
            vis,
            candidate_box(
                best
            ),
            selected_color,
            3,
        )

    if best is None:

        status = (
            "NO DETECTION"
        )

    elif correct_detection:

        status = (
            "CORRECT STOP DETECTION"
        )

    else:

        status = (
            "FALSE POSITIVE"
        )

    if np.isfinite(
        iou_value
    ):

        iou_text = (
            f"IoU={iou_value:.3f}"
        )

    else:

        iou_text = (
            "IoU=N/A"
        )

    text = (
        f"{status} | "
        f"{iou_text} | "
        f"route={route_s:+.1f}m | "
        f"sign={distance_to_sign:.1f}m"
    )

    cv2.putText(
        vis,
        text,
        (
            25,
            45,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (
            0,
            0,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    os.makedirs(
        os.path.dirname(
            output_path
        ),
        exist_ok=True,
    )

    if not cv2.imwrite(
        output_path,
        vis,
    ):

        raise RuntimeError(
            "Could not save debug image: "
            f"{output_path}"
        )

    base, _ = os.path.splitext(
        output_path
    )

    cv2.imwrite(
        base + "_mask.png",
        mask,
    )


# ============================================================
# PROCESS ONE CONDITION
# ============================================================

def process_dataset(
    dataset_name,
    dataset_info,
):

    root = dataset_info[
        "root"
    ]

    csv_path = os.path.join(
        root,
        "frames.csv",
    )

    frame_dir = os.path.join(
        root,
        "frames",
    )

    if not os.path.exists(
        csv_path
    ):

        raise RuntimeError(
            "Missing metadata: "
            f"{csv_path}"
        )

    metadata = pd.read_csv(
        csv_path
    )

    validate_metadata_columns(
        metadata,
        csv_path,
    )

    rows = []
    debug_cache = {}

    for _, row in metadata.iterrows():

        sample_id = int(
            row["sample_id"]
        )

        route_s = float(
            row["route_s_m"]
        )

        distance_to_sign = float(
            row[
                "distance_to_sign_m"
            ]
        )

        image_path = os.path.join(
            frame_dir,
            f"{sample_id:04d}.png",
        )

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise RuntimeError(
                "Could not read image: "
                f"{image_path}"
            )

        (
            best,
            mask,
            candidates,
        ) = detect_stop_sign(
            image
        )

        raw_detected = (
            best is not None
        )

        gt_box = gt_box_from_row(
            row
        )

        gt_available = (
            gt_box is not None
        )

        if (
            raw_detected
            and gt_available
        ):

            iou_value = box_iou(
                candidate_box(
                    best
                ),
                gt_box,
            )

        else:

            iou_value = np.nan

        correct_detection = bool(
            raw_detected
            and gt_available
            and np.isfinite(
                iou_value
            )
            and iou_value
            >= IOU_THRESHOLD
        )

        false_positive = bool(
            raw_detected
            and gt_available
            and not correct_detection
        )

        failure = bool(
            gt_available
            and not correct_detection
        )

        if best is not None:

            bbox_x = best["x"]
            bbox_y = best["y"]
            bbox_w = best["w"]
            bbox_h = best["h"]

            area = best["area"]
            vertices = best["vertices"]
            red_ratio = best["red_ratio"]
            score = best["score"]

        else:

            bbox_x = np.nan
            bbox_y = np.nan
            bbox_w = np.nan
            bbox_h = np.nan

            area = np.nan
            vertices = np.nan
            red_ratio = np.nan
            score = np.nan

        if gt_box is None:

            gt_x1 = np.nan
            gt_y1 = np.nan
            gt_x2 = np.nan
            gt_y2 = np.nan

        else:

            (
                gt_x1,
                gt_y1,
                gt_x2,
                gt_y2,
            ) = gt_box

        rows.append({
            "sample_id":
                sample_id,

            "weather":
                row["weather"],

            "severity":
                int(
                    row["severity"]
                ),

            "route_s_m":
                route_s,

            "distance_to_sign_m":
                distance_to_sign,

            "sign_in_view":
                int(
                    row["sign_in_view"]
                ),

            "gt_available":
                int(
                    gt_available
                ),

            "gt_bbox_x1":
                gt_x1,

            "gt_bbox_y1":
                gt_y1,

            "gt_bbox_x2":
                gt_x2,

            "gt_bbox_y2":
                gt_y2,

            "raw_detected":
                int(
                    raw_detected
                ),

            "correct_detection":
                int(
                    correct_detection
                ),

            "false_positive":
                int(
                    false_positive
                ),

            "failure":
                int(
                    failure
                ),

            "iou":
                iou_value,

            "iou_threshold":
                IOU_THRESHOLD,

            "num_candidates":
                len(candidates),

            "bbox_x":
                bbox_x,

            "bbox_y":
                bbox_y,

            "bbox_w":
                bbox_w,

            "bbox_h":
                bbox_h,

            "area_px":
                area,

            "vertices":
                vertices,

            "red_ratio":
                red_ratio,

            "score":
                score,
        })

        debug_cache[
            sample_id
        ] = {
            "image":
                image,

            "mask":
                mask,

            "candidates":
                candidates,

            "best":
                best,

            "gt_box":
                gt_box,

            "iou":
                iou_value,

            "correct_detection":
                correct_detection,

            "distance_to_sign":
                distance_to_sign,

            "route_s":
                route_s,
        }

    result_df = pd.DataFrame(
        rows
    )

    # The STOP encounter is the approach phase.
    # Only frames with a valid visual GT box are evaluable.
    approach = result_df[
        (
            result_df[
                "route_s_m"
            ] <= 0.0
        )
        & (
            result_df[
                "gt_available"
            ] == 1
        )
    ].copy()

    if len(approach) == 0:

        raise RuntimeError(
            "No evaluable approach frames with STOP-sign "
            "bounding-box ground truth."
        )

    raw_detections = (
        approach[
            approach[
                "raw_detected"
            ] == 1
        ]
        .sort_values(
            "sample_id"
        )
    )

    if len(
        raw_detections
    ) > 0:

        first_raw_sample = int(
            raw_detections.iloc[0][
                "sample_id"
            ]
        )

    else:

        first_raw_sample = None

    correct_detections = (
        approach[
            approach[
                "correct_detection"
            ] == 1
        ]
        .sort_values(
            "sample_id"
        )
    )

    if len(
        correct_detections
    ) > 0:

        first_correct = (
            correct_detections.iloc[0]
        )

        first_correct_sample = int(
            first_correct[
                "sample_id"
            ]
        )

        first_correct_route = float(
            first_correct[
                "route_s_m"
            ]
        )

        first_correct_distance = float(
            first_correct[
                "distance_to_sign_m"
            ]
        )

    else:

        first_correct_sample = None
        first_correct_route = None

        # Project definition:
        # no correct detection => D(s) = 0 m
        first_correct_distance = 0.0

    debug_samples = {
        1,
        8,
        20,
        30,
        40,
        60,
        80,
        90,
        95,
        99,
    }

    if first_raw_sample is not None:

        for sid in [
            first_raw_sample - 1,
            first_raw_sample,
            first_raw_sample + 1,
        ]:

            if (
                1
                <= sid
                <= len(metadata)
            ):

                debug_samples.add(
                    sid
                )

    if first_correct_sample is not None:

        for sid in [
            first_correct_sample - 2,
            first_correct_sample - 1,
            first_correct_sample,
            first_correct_sample + 1,
            first_correct_sample + 2,
        ]:

            if (
                1
                <= sid
                <= len(metadata)
            ):

                debug_samples.add(
                    sid
                )

    false_positives = (
        approach[
            approach[
                "false_positive"
            ] == 1
        ]
        .sort_values(
            "sample_id"
        )
    )

    if len(
        false_positives
    ) > 0:

        debug_samples.add(
            int(
                false_positives.iloc[0][
                    "sample_id"
                ]
            )
        )

    for sample_id in sorted(
        debug_samples
    ):

        if sample_id not in debug_cache:
            continue

        info = debug_cache[
            sample_id
        ]

        debug_path = os.path.join(
            DEBUG_ROOT,
            dataset_name,
            f"{sample_id:04d}.png",
        )

        save_debug(
            info["image"],
            info["mask"],
            info["candidates"],
            info["best"],
            info["gt_box"],
            info["iou"],
            info["correct_detection"],
            debug_path,
            info["distance_to_sign"],
            info["route_s"],
        )

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    output_path = os.path.join(
        OUTPUT_ROOT,
        f"{dataset_name}_per_frame.csv",
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    num_approach = len(
        approach
    )

    raw_count = int(
        approach[
            "raw_detected"
        ].sum()
    )

    correct_count = int(
        approach[
            "correct_detection"
        ].sum()
    )

    false_positive_count = int(
        approach[
            "false_positive"
        ].sum()
    )

    iou_on_raw = approach.loc[
        approach[
            "raw_detected"
        ] == 1,
        "iou",
    ].dropna()

    summary = {
        "dataset":
            dataset_name,

        "severity":
            dataset_info[
                "severity"
            ],

        "iou_threshold":
            IOU_THRESHOLD,

        "evaluable_approach_frames":
            num_approach,

        "raw_detection_count":
            raw_count,

        "raw_detection_rate":
            (
                raw_count
                / num_approach
            ),

        "correct_detection_count":
            correct_count,

        "correct_detection_rate":
            (
                correct_count
                / num_approach
            ),

        "false_positive_count":
            false_positive_count,

        "false_positive_rate":
            (
                false_positive_count
                / num_approach
            ),

        "mean_iou_when_raw_detected":
            (
                float(
                    iou_on_raw.mean()
                )
                if len(iou_on_raw) > 0
                else np.nan
            ),

        "median_iou_when_raw_detected":
            (
                float(
                    iou_on_raw.median()
                )
                if len(iou_on_raw) > 0
                else np.nan
            ),

        "first_correct_detection_sample":
            first_correct_sample,

        "first_correct_detection_route_s_m":
            first_correct_route,

        "first_correct_detection_distance_m":
            first_correct_distance,
    }

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    reset_output_directory()

    summaries = []

    for (
        dataset_name,
        dataset_info,
    ) in DATASETS.items():

        print()
        print(
            "=" * 70
        )
        print(
            "Processing:",
            dataset_name,
        )
        print(
            "=" * 70
        )

        summary = process_dataset(
            dataset_name,
            dataset_info,
        )

        summaries.append(
            summary
        )

        print(
            "Evaluable approach frames:",
            summary[
                "evaluable_approach_frames"
            ],
        )

        print(
            "Raw detections:",
            summary[
                "raw_detection_count"
            ],
        )

        print(
            "Raw detection rate:",
            f'{summary["raw_detection_rate"] * 100:.2f}%',
        )

        print(
            "Correct IoU detections:",
            summary[
                "correct_detection_count"
            ],
        )

        print(
            "Correct detection rate:",
            f'{summary["correct_detection_rate"] * 100:.2f}%',
        )

        print(
            "False positives:",
            summary[
                "false_positive_count"
            ],
        )

        print(
            "Mean IoU on raw detections:",
            f'{summary["mean_iou_when_raw_detected"]:.4f}',
        )

        print(
            "First CORRECT detection sample:",
            summary[
                "first_correct_detection_sample"
            ],
        )

        print(
            "First CORRECT detection route:",
            summary[
                "first_correct_detection_route_s_m"
            ],
        )

        print(
            "First CORRECT detection distance:",
            f'{summary["first_correct_detection_distance_m"]:.2f} m',
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = os.path.join(
        OUTPUT_ROOT,
        "summary.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "=" * 70
    )
    print(
        "HSV IoU VALIDATION COMPLETE"
    )
    print(
        "Correct detection rule: IoU >= 0.50"
    )
    print(
        "Summary:",
        summary_path,
    )
    print(
        "Debug images:",
        DEBUG_ROOT,
    )
    print()
    print(
        "Magenta box = CARLA visual STOP-sign GT"
    )
    print(
        "Green box   = correct detector output"
    )
    print(
        "Red box     = false positive"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
