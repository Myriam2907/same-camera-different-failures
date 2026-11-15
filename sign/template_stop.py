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
# During this IoU-validation stage we use ONLY the corrected
# laterally-varying Clear recording. Old pilot fog recordings
# must not be compared with this new geometry/metric.
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

OUTPUT_ROOT = "results/sign_template_iou"

DEBUG_ROOT = os.path.join(
    OUTPUT_ROOT,
    "debug",
)

TEMPLATE_PATH = os.path.join(
    OUTPUT_ROOT,
    "stop_template.png",
)

TEMPLATE_MASK_PATH = os.path.join(
    OUTPUT_ROOT,
    "stop_template_mask.png",
)

CALIBRATION_PATH = os.path.join(
    OUTPUT_ROOT,
    "clear_calibration.csv",
)

IOU_THRESHOLD = 0.50


# ============================================================
# TEMPLATE CREATION
# ============================================================

TEMPLATE_SAMPLE_ID = 140

TEMPLATE_SEARCH_HALF_SIZE = 70
TEMPLATE_PADDING = 4


# ============================================================
# MULTI-SCALE MATCHING
# ============================================================

MATCH_SCALES = np.geomspace(
    0.15,
    2.20,
    36,
)


# ============================================================
# BROAD TRAFFIC-SIGN SEARCH REGION
# ============================================================

SEARCH_X_MIN = int(
    0.45 * IMAGE_W
)

SEARCH_X_MAX = int(
    0.95 * IMAGE_W
)

SEARCH_Y_MIN = int(
    0.18 * IMAGE_H
)

SEARCH_Y_MAX = int(
    0.75 * IMAGE_H
)


# ============================================================
# OUTPUT
# ============================================================

def reset_output_directory():

    if os.path.exists(
        OUTPUT_ROOT
    ):

        print(
            "Deleting old IoU template results:",
            OUTPUT_ROOT,
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    os.makedirs(
        OUTPUT_ROOT,
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
# CREATE TEMPLATE AND MASK
# ============================================================

def create_template():

    """
    Create one fixed STOP-sign template from Clear weather.

    Ground truth is used only to localize the sign while creating
    the reference template. Runtime matching later searches the
    broad fixed region without ground-truth assistance.
    """

    clear_root = DATASETS[
        "clear_drift"
    ][
        "root"
    ]

    csv_path = os.path.join(
        clear_root,
        "frames.csv",
    )

    metadata = pd.read_csv(
        csv_path
    )

    validate_metadata_columns(
        metadata,
        csv_path,
    )

    matching = metadata[
        metadata[
            "sample_id"
        ] == TEMPLATE_SAMPLE_ID
    ]

    if len(matching) != 1:

        raise RuntimeError(
            "Could not find template frame."
        )

    row = matching.iloc[0]

    image_path = os.path.join(
        clear_root,
        "frames",
        f"{TEMPLATE_SAMPLE_ID:04d}.png",
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        raise RuntimeError(
            f"Could not read: {image_path}"
        )

    gt_box = gt_box_from_row(
        row
    )

    if gt_box is None:

        raise RuntimeError(
            "Template frame does not have a valid "
            "STOP-sign GT bounding box."
        )

    (
        gt_x1,
        gt_y1,
        gt_x2,
        gt_y2,
    ) = gt_box

    gt_x = int(
        round(
            (
                gt_x1
                + gt_x2
            )
            / 2.0
        )
    )

    gt_y = int(
        round(
            (
                gt_y1
                + gt_y2
            )
            / 2.0
        )
    )

    patch_x1 = max(
        0,
        gt_x
        - TEMPLATE_SEARCH_HALF_SIZE,
    )

    patch_y1 = max(
        0,
        gt_y
        - TEMPLATE_SEARCH_HALF_SIZE,
    )

    patch_x2 = min(
        IMAGE_W,
        gt_x
        + TEMPLATE_SEARCH_HALF_SIZE,
    )

    patch_y2 = min(
        IMAGE_H,
        gt_y
        + TEMPLATE_SEARCH_HALF_SIZE,
    )

    patch = image[
        patch_y1:patch_y2,
        patch_x1:patch_x2,
    ]

    hsv = cv2.cvtColor(
        patch,
        cv2.COLOR_BGR2HSV,
    )

    red1 = cv2.inRange(
        hsv,
        np.array(
            [0, 60, 40],
            dtype=np.uint8,
        ),
        np.array(
            [12, 255, 255],
            dtype=np.uint8,
        ),
    )

    red2 = cv2.inRange(
        hsv,
        np.array(
            [165, 60, 40],
            dtype=np.uint8,
        ),
        np.array(
            [180, 255, 255],
            dtype=np.uint8,
        ),
    )

    red_mask = cv2.bitwise_or(
        red1,
        red2,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    contours, _ = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    local_gt_x = (
        gt_x
        - patch_x1
    )

    local_gt_y = (
        gt_y
        - patch_y1
    )

    best_contour = None
    best_distance = float(
        "inf"
    )

    for contour in contours:

        area = float(
            cv2.contourArea(
                contour
            )
        )

        if area < 20.0:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        center_x = (
            x + w / 2.0
        )

        center_y = (
            y + h / 2.0
        )

        distance = np.hypot(
            center_x
            - local_gt_x,
            center_y
            - local_gt_y,
        )

        if distance < best_distance:

            best_distance = float(
                distance
            )

            best_contour = contour

    if best_contour is None:

        raise RuntimeError(
            "Could not isolate STOP sign "
            "for template creation."
        )

    x, y, w, h = cv2.boundingRect(
        best_contour
    )

    crop_x1 = max(
        0,
        x - TEMPLATE_PADDING,
    )

    crop_y1 = max(
        0,
        y - TEMPLATE_PADDING,
    )

    crop_x2 = min(
        patch.shape[1],
        x + w + TEMPLATE_PADDING,
    )

    crop_y2 = min(
        patch.shape[0],
        y + h + TEMPLATE_PADDING,
    )

    template = patch[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()

    if (
        template.shape[0] < 8
        or template.shape[1] < 8
    ):

        raise RuntimeError(
            "Template is too small."
        )

    template_hsv = cv2.cvtColor(
        template,
        cv2.COLOR_BGR2HSV,
    )

    template_red1 = cv2.inRange(
        template_hsv,
        np.array(
            [0, 50, 30],
            dtype=np.uint8,
        ),
        np.array(
            [12, 255, 255],
            dtype=np.uint8,
        ),
    )

    template_red2 = cv2.inRange(
        template_hsv,
        np.array(
            [165, 50, 30],
            dtype=np.uint8,
        ),
        np.array(
            [180, 255, 255],
            dtype=np.uint8,
        ),
    )

    template_red = cv2.bitwise_or(
        template_red1,
        template_red2,
    )

    template_red = cv2.morphologyEx(
        template_red,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    mask_contours, _ = cv2.findContours(
        template_red,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(mask_contours) == 0:

        raise RuntimeError(
            "Could not build template mask."
        )

    largest = max(
        mask_contours,
        key=cv2.contourArea,
    )

    hull = cv2.convexHull(
        largest
    )

    template_mask = np.zeros(
        template.shape[:2],
        dtype=np.uint8,
    )

    cv2.fillConvexPoly(
        template_mask,
        hull,
        255,
    )

    mask_fraction = (
        np.count_nonzero(
            template_mask
        )
        / float(
            template_mask.size
        )
    )

    if mask_fraction < 0.10:

        raise RuntimeError(
            "Template mask is unexpectedly small."
        )

    cv2.imwrite(
        TEMPLATE_PATH,
        template,
    )

    cv2.imwrite(
        TEMPLATE_MASK_PATH,
        template_mask,
    )

    print(
        "Template created from clear sample:",
        TEMPLATE_SAMPLE_ID,
    )

    print(
        "Template size:",
        f"{template.shape[1]}x"
        f"{template.shape[0]} px",
    )

    print(
        "Template mask coverage:",
        f"{mask_fraction * 100:.1f}%",
    )

    return (
        template,
        template_mask,
    )


# ============================================================
# MASKED COLOR TEMPLATE MATCHING
# ============================================================

def run_match_template(
    search,
    scaled_template,
    scaled_mask,
):

    try:

        response = cv2.matchTemplate(
            search,
            scaled_template,
            cv2.TM_CCORR_NORMED,
            mask=scaled_mask,
        )

    except cv2.error:

        mask_three_channel = cv2.merge(
            [
                scaled_mask,
                scaled_mask,
                scaled_mask,
            ]
        )

        response = cv2.matchTemplate(
            search,
            scaled_template,
            cv2.TM_CCORR_NORMED,
            mask=mask_three_channel,
        )

    response = np.nan_to_num(
        response,
        nan=-1.0,
        posinf=-1.0,
        neginf=-1.0,
    )

    return response


def raw_template_match(
    image,
    template,
    template_mask,
):

    """
    Return the single highest-scoring multi-scale template match.

    No GT information and no acceptance threshold are used here.
    """

    search = image[
        SEARCH_Y_MIN:SEARCH_Y_MAX,
        SEARCH_X_MIN:SEARCH_X_MAX,
    ]

    best = None
    best_score = -1.0

    original_h = template.shape[0]
    original_w = template.shape[1]

    for scale in MATCH_SCALES:

        scaled_w = int(
            round(
                original_w
                * scale
            )
        )

        scaled_h = int(
            round(
                original_h
                * scale
            )
        )

        if (
            scaled_w < 7
            or scaled_h < 7
        ):
            continue

        if (
            scaled_w
            >= search.shape[1]
            or scaled_h
            >= search.shape[0]
        ):
            continue

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        )

        scaled_template = cv2.resize(
            template,
            (
                scaled_w,
                scaled_h,
            ),
            interpolation=interpolation,
        )

        scaled_mask = cv2.resize(
            template_mask,
            (
                scaled_w,
                scaled_h,
            ),
            interpolation=cv2.INTER_NEAREST,
        )

        if np.count_nonzero(
            scaled_mask
        ) < 10:
            continue

        response = run_match_template(
            search,
            scaled_template,
            scaled_mask,
        )

        _, max_value, _, max_location = (
            cv2.minMaxLoc(
                response
            )
        )

        if max_value > best_score:

            best_score = float(
                max_value
            )

            best = {
                "x":
                    SEARCH_X_MIN
                    + int(
                        max_location[0]
                    ),

                "y":
                    SEARCH_Y_MIN
                    + int(
                        max_location[1]
                    ),

                "w":
                    int(
                        scaled_w
                    ),

                "h":
                    int(
                        scaled_h
                    ),

                "scale":
                    float(
                        scale
                    ),

                "score":
                    float(
                        max_value
                    ),
            }

    return best


# ============================================================
# CLEAR-ONLY THRESHOLD CALIBRATION
# ============================================================

def calibrate_threshold(
    template,
    template_mask,
):

    """
    Calibrate the template score threshold using Clear weather.

    Correctness during calibration is now defined by the final
    project rule:

        IoU(raw_match, CARLA_GT_box) >= 0.50

    NOTE:
    this is still calibration-set behavior, not an independent
    estimate of Clear performance. Final experiments will later
    use separate repeated Clear runs.
    """

    clear_root = DATASETS[
        "clear_drift"
    ][
        "root"
    ]

    csv_path = os.path.join(
        clear_root,
        "frames.csv",
    )

    metadata = pd.read_csv(
        csv_path
    )

    validate_metadata_columns(
        metadata,
        csv_path,
    )

    metadata = metadata[
        (
            metadata[
                "route_s_m"
            ] <= 0.0
        )
        & (
            metadata[
                "sign_in_view"
            ] == 1
        )
    ].copy()

    frame_dir = os.path.join(
        clear_root,
        "frames",
    )

    records = []

    print()
    print(
        "Running CLEAR-only IoU threshold calibration..."
    )

    for _, row in metadata.iterrows():

        sample_id = int(
            row["sample_id"]
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
                f"Could not read: "
                f"{image_path}"
            )

        raw_candidate = raw_template_match(
            image,
            template,
            template_mask,
        )

        gt_box = gt_box_from_row(
            row
        )

        if raw_candidate is None:

            score = -1.0
            iou_value = np.nan
            correct = False

        else:

            score = float(
                raw_candidate["score"]
            )

            iou_value = box_iou(
                candidate_box(
                    raw_candidate
                ),
                gt_box,
            )

            correct = bool(
                np.isfinite(
                    iou_value
                )
                and iou_value
                >= IOU_THRESHOLD
            )

        records.append({
            "sample_id":
                sample_id,

            "route_s_m":
                float(
                    row["route_s_m"]
                ),

            "distance_to_sign_m":
                float(
                    row[
                        "distance_to_sign_m"
                    ]
                ),

            "raw_best_score":
                score,

            "raw_best_iou":
                iou_value,

            "raw_best_match_correct":
                int(
                    correct
                ),
        })

    calibration_df = pd.DataFrame(
        records
    )

    calibration_df.to_csv(
        CALIBRATION_PATH,
        index=False,
    )

    scores = calibration_df[
        "raw_best_score"
    ].to_numpy(
        dtype=np.float64
    )

    correctness = calibration_df[
        "raw_best_match_correct"
    ].to_numpy(
        dtype=bool
    )

    thresholds = np.unique(
        scores[
            np.isfinite(
                scores
            )
        ]
    )

    thresholds = np.sort(
        thresholds
    )

    best_threshold = None
    best_metrics = None

    number_of_frames = len(
        calibration_df
    )

    for threshold in thresholds:

        accepted = (
            scores >= threshold
        )

        true_positive = int(
            np.sum(
                accepted
                & correctness
            )
        )

        false_positive = int(
            np.sum(
                accepted
                & ~correctness
            )
        )

        if (
            true_positive
            + false_positive
            > 0
        ):

            precision = (
                true_positive
                / (
                    true_positive
                    + false_positive
                )
            )

        else:

            precision = 0.0

        recall = (
            true_positive
            / number_of_frames
        )

        if (
            precision
            + recall
            > 0.0
        ):

            f1 = (
                2.0
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:

            f1 = 0.0

        candidate_key = (
            f1,
            precision,
            float(
                threshold
            ),
        )

        if best_metrics is None:

            choose = True

        else:

            best_key = (
                best_metrics[
                    "f1"
                ],
                best_metrics[
                    "precision"
                ],
                best_metrics[
                    "threshold"
                ],
            )

            choose = (
                candidate_key
                > best_key
            )

        if choose:

            best_threshold = float(
                threshold
            )

            best_metrics = {
                "threshold":
                    float(
                        threshold
                    ),

                "true_positive":
                    true_positive,

                "false_positive":
                    false_positive,

                "precision":
                    float(
                        precision
                    ),

                "recall":
                    float(
                        recall
                    ),

                "f1":
                    float(
                        f1
                    ),
            }

    if best_threshold is None:

        raise RuntimeError(
            "Could not calibrate threshold."
        )

    print()
    print(
        "=" * 70
    )

    print(
        "CLEAR IoU CALIBRATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "IoU correctness threshold:",
        IOU_THRESHOLD,
    )

    print(
        "Frozen template-score threshold:",
        f"{best_threshold:.4f}",
    )

    print(
        "Calibration precision:",
        f'{best_metrics["precision"] * 100:.2f}%',
    )

    print(
        "Calibration recall:",
        f'{best_metrics["recall"] * 100:.2f}%',
    )

    print(
        "Calibration F1:",
        f'{best_metrics["f1"]:.4f}',
    )

    return best_threshold


# ============================================================
# DEBUG VISUALIZATION
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
    candidate,
    gt_box,
    detected,
    correct,
    iou_value,
    threshold,
    output_path,
    route_s,
    distance_to_sign,
):

    """
    Blue    = runtime search region
    Magenta = CARLA visual STOP-sign GT bbox
    Green   = accepted correct match (IoU >= 0.50)
    Red     = accepted false positive
    Gray    = best raw match rejected by score threshold
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

    if candidate is not None:

        if not detected:

            color = (
                150,
                150,
                150,
            )

        elif correct:

            color = (
                0,
                255,
                0,
            )

        else:

            color = (
                0,
                0,
                255,
            )

        draw_box(
            vis,
            candidate_box(
                candidate
            ),
            color,
            3,
        )

    if candidate is None:

        status = (
            "NO RAW MATCH"
        )

        score_text = ""

    else:

        score_text = (
            f"score={candidate['score']:.3f} "
            f"thr={threshold:.3f}"
        )

        if not detected:

            status = (
                "REJECTED MATCH"
            )

        elif correct:

            status = (
                "CORRECT TEMPLATE DETECTION"
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
        f"{score_text} | "
        f"{iou_text} | "
        f"route={route_s:+.1f}m | "
        f"sign={distance_to_sign:.1f}m"
    )

    cv2.putText(
        vis,
        text,
        (
            20,
            42,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
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
            "Could not save: "
            f"{output_path}"
        )


# ============================================================
# PROCESS ONE WEATHER CONDITION
# ============================================================

def process_dataset(
    dataset_name,
    dataset_info,
    template,
    template_mask,
    threshold,
):

    root = dataset_info[
        "root"
    ]

    csv_path = os.path.join(
        root,
        "frames.csv",
    )

    metadata = pd.read_csv(
        csv_path
    )

    validate_metadata_columns(
        metadata,
        csv_path,
    )

    frame_dir = os.path.join(
        root,
        "frames",
    )

    rows = []
    debug_cache = {}

    for _, row in metadata.iterrows():

        sample_id = int(
            row["sample_id"]
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
                f"Could not read: "
                f"{image_path}"
            )

        route_s = float(
            row["route_s_m"]
        )

        distance_to_sign = float(
            row[
                "distance_to_sign_m"
            ]
        )

        raw_candidate = raw_template_match(
            image,
            template,
            template_mask,
        )

        if raw_candidate is None:

            best_score = -1.0
            detected = False

        else:

            best_score = float(
                raw_candidate[
                    "score"
                ]
            )

            detected = (
                best_score
                >= threshold
            )

        gt_box = gt_box_from_row(
            row
        )

        gt_available = (
            gt_box is not None
        )

        if (
            raw_candidate is not None
            and gt_available
        ):

            iou_value = box_iou(
                candidate_box(
                    raw_candidate
                ),
                gt_box,
            )

        else:

            iou_value = np.nan

        correct = bool(
            detected
            and gt_available
            and np.isfinite(
                iou_value
            )
            and iou_value
            >= IOU_THRESHOLD
        )

        false_positive = bool(
            detected
            and gt_available
            and not correct
        )

        failure = bool(
            gt_available
            and not correct
        )

        if raw_candidate is None:

            bbox_x = np.nan
            bbox_y = np.nan
            bbox_w = np.nan
            bbox_h = np.nan
            scale = np.nan

        else:

            bbox_x = raw_candidate[
                "x"
            ]

            bbox_y = raw_candidate[
                "y"
            ]

            bbox_w = raw_candidate[
                "w"
            ]

            bbox_h = raw_candidate[
                "h"
            ]

            scale = raw_candidate[
                "scale"
            ]

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

            "best_score":
                best_score,

            "threshold":
                threshold,

            "detected":
                int(
                    detected
                ),

            "correct_detection":
                int(
                    correct
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

            "bbox_x":
                bbox_x,

            "bbox_y":
                bbox_y,

            "bbox_w":
                bbox_w,

            "bbox_h":
                bbox_h,

            "match_scale":
                scale,
        })

        debug_cache[
            sample_id
        ] = {
            "image":
                image,

            "candidate":
                raw_candidate,

            "gt_box":
                gt_box,

            "detected":
                detected,

            "correct":
                correct,

            "iou":
                iou_value,

            "route_s":
                route_s,

            "distance":
                distance_to_sign,
        }

    result_df = pd.DataFrame(
        rows
    )

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
            "No evaluable approach frames with "
            "STOP-sign GT boxes."
        )

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

        first_sample = int(
            first_correct[
                "sample_id"
            ]
        )

        first_route = float(
            first_correct[
                "route_s_m"
            ]
        )

        first_distance = float(
            first_correct[
                "distance_to_sign_m"
            ]
        )

    else:

        first_sample = None
        first_route = None
        first_distance = 0.0

    debug_samples = {
        1,
        20,
        40,
        60,
        80,
        90,
        95,
        99,
    }

    if first_sample is not None:

        for sid in [
            first_sample - 2,
            first_sample - 1,
            first_sample,
            first_sample + 1,
            first_sample + 2,
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

        item = debug_cache[
            sample_id
        ]

        output_path = os.path.join(
            DEBUG_ROOT,
            dataset_name,
            f"{sample_id:04d}.png",
        )

        save_debug(
            item["image"],
            item["candidate"],
            item["gt_box"],
            item["detected"],
            item["correct"],
            item["iou"],
            threshold,
            output_path,
            item["route_s"],
            item["distance"],
        )

    output_csv = os.path.join(
        OUTPUT_ROOT,
        f"{dataset_name}_per_frame.csv",
    )

    result_df.to_csv(
        output_csv,
        index=False,
    )

    number_of_frames = len(
        approach
    )

    detection_count = int(
        approach[
            "detected"
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

    accepted_iou = approach.loc[
        approach[
            "detected"
        ] == 1,
        "iou",
    ].dropna()

    return {
        "dataset":
            dataset_name,

        "severity":
            dataset_info[
                "severity"
            ],

        "iou_threshold":
            IOU_THRESHOLD,

        "evaluable_approach_frames":
            number_of_frames,

        "detection_count":
            detection_count,

        "detection_rate":
            (
                detection_count
                / number_of_frames
            ),

        "correct_detection_count":
            correct_count,

        "correct_detection_rate":
            (
                correct_count
                / number_of_frames
            ),

        "false_positive_count":
            false_positive_count,

        "false_positive_rate":
            (
                false_positive_count
                / number_of_frames
            ),

        "mean_iou_when_accepted":
            (
                float(
                    accepted_iou.mean()
                )
                if len(accepted_iou) > 0
                else np.nan
            ),

        "median_iou_when_accepted":
            (
                float(
                    accepted_iou.median()
                )
                if len(accepted_iou) > 0
                else np.nan
            ),

        "first_correct_detection_sample":
            first_sample,

        "first_correct_detection_route_s_m":
            first_route,

        "first_correct_detection_distance_m":
            first_distance,

        "frozen_threshold":
            threshold,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    reset_output_directory()

    print()
    print(
        "=" * 70
    )
    print(
        "CREATING MASKED COLOR STOP TEMPLATE"
    )
    print(
        "=" * 70
    )

    (
        template,
        template_mask,
    ) = create_template()

    threshold = calibrate_threshold(
        template,
        template_mask,
    )

    print()
    print(
        "Template-score threshold is now FROZEN."
    )
    print(
        "Correctness rule is IoU >= 0.50."
    )

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
            template,
            template_mask,
            threshold,
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
            "Accepted detections:",
            summary[
                "detection_count"
            ],
        )

        print(
            "Detection rate:",
            f'{summary["detection_rate"] * 100:.2f}%',
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
            "Mean IoU on accepted detections:",
            f'{summary["mean_iou_when_accepted"]:.4f}',
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
        "TEMPLATE IoU VALIDATION COMPLETE"
    )
    print(
        "Summary:",
        summary_path,
    )
    print(
        "Calibration data:",
        CALIBRATION_PATH,
    )
    print(
        "Template:",
        TEMPLATE_PATH,
    )
    print(
        "Template mask:",
        TEMPLATE_MASK_PATH,
    )
    print(
        "Debug images:",
        DEBUG_ROOT,
    )
    print()
    print(
        "Magenta = CARLA visual STOP-sign GT bbox"
    )
    print(
        "Green   = accepted correct detection"
    )
    print(
        "Red     = accepted false positive"
    )
    print(
        "Gray    = best match rejected by score threshold"
    )
    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
