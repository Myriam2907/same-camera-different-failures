import os
import math
import shutil

import carla
import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_W = 1280
IMAGE_H = 720
CAMERA_FOV_DEG = 90.0

CAMERA_X = 1.5
CAMERA_Y = 0.0
CAMERA_Z = 1.7


DATASETS = {
    "clear": {
        "root": "data/recordings/stop3_clear_0",
        "severity": 0,
    },

    "fog45": {
        "root": "data/recordings/stop3_fog_45",
        "severity": 45,
    },

    "fog90": {
        "root": "data/recordings/stop3_fog_90",
        "severity": 90,
    },
}


OUTPUT_ROOT = "results/sign_template"

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


# ============================================================
# DELETE OLD / WRONG RESULTS
# ============================================================

def reset_output_directory():
    """
    Remove all previous template-matching results.

    This prevents obsolete / incorrect runs from being
    mixed with the corrected experiment.
    """

    if os.path.exists(
        OUTPUT_ROOT
    ):
        print(
            f"Deleting old results: "
            f"{OUTPUT_ROOT}"
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )


# ============================================================
# STOP SIGN WORLD LOCATION
# ============================================================

STOP_SIGN_WORLD_X = 26.04
STOP_SIGN_WORLD_Y = 75.15
STOP_SIGN_WORLD_Z = 2.31


# ============================================================
# TEMPLATE CREATION
# ============================================================

# Use one clear frame to create the fixed reference template.
TEMPLATE_SAMPLE_ID = 80

TEMPLATE_SEARCH_HALF_SIZE = 70
TEMPLATE_PADDING = 4


# ============================================================
# MULTI-SCALE MATCHING
# ============================================================

# Geometric spacing gives better coverage of both tiny
# distant signs and larger close signs.

MATCH_SCALES = np.geomspace(
    0.15,
    2.20,
    36,
)


# ============================================================
# BROAD TRAFFIC-SIGN SEARCH REGION
# ============================================================

# The target sign is on the right side of this fixed route.
#
# We also exclude the bottom road surface, because traffic
# signs are roadside objects rather than road texture.
#
# This region is frozen and used for every weather condition.

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
# CORRECTNESS EVALUATION
# ============================================================

GT_MARGIN_PX = 8


# ============================================================
# CAMERA PROJECTION
# ============================================================

def build_camera_matrix():

    focal = (
        IMAGE_W
        / (
            2.0
            * math.tan(
                math.radians(
                    CAMERA_FOV_DEG
                ) / 2.0
            )
        )
    )

    return np.array(
        [
            [
                focal,
                0.0,
                IMAGE_W / 2.0,
            ],

            [
                0.0,
                focal,
                IMAGE_H / 2.0,
            ],

            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )


CAMERA_K = build_camera_matrix()


def build_camera_transform_from_row(
    row,
):

    ego_transform = carla.Transform(
        carla.Location(
            x=float(
                row["ego_x"]
            ),
            y=float(
                row["ego_y"]
            ),
            z=float(
                row["ego_z"]
            ),
        ),

        carla.Rotation(
            pitch=float(
                row["ego_pitch"]
            ),
            yaw=float(
                row["ego_yaw"]
            ),
            roll=float(
                row["ego_roll"]
            ),
        ),
    )

    camera_location = (
        ego_transform.transform(
            carla.Location(
                x=CAMERA_X,
                y=CAMERA_Y,
                z=CAMERA_Z,
            )
        )
    )

    return carla.Transform(
        camera_location,

        carla.Rotation(
            pitch=float(
                row["ego_pitch"]
            ),
            yaw=float(
                row["ego_yaw"]
            ),
            roll=float(
                row["ego_roll"]
            ),
        ),
    )


def project_stop_sign_center(
    row,
):
    """
    Project the physical CARLA STOP sign center
    into image coordinates.

    Used ONLY for evaluation and template creation,
    never for runtime candidate selection.
    """

    camera_transform = (
        build_camera_transform_from_row(
            row
        )
    )

    world_to_camera = np.array(
        camera_transform.get_inverse_matrix(),
        dtype=np.float64,
    )

    point_world = np.array(
        [
            STOP_SIGN_WORLD_X,
            STOP_SIGN_WORLD_Y,
            STOP_SIGN_WORLD_Z,
            1.0,
        ],
        dtype=np.float64,
    )

    point_camera = (
        world_to_camera
        @ point_world
    )

    forward = float(
        point_camera[0]
    )

    right = float(
        point_camera[1]
    )

    up = float(
        point_camera[2]
    )

    if forward <= 0.0:
        return None

    fx = CAMERA_K[0, 0]
    fy = CAMERA_K[1, 1]

    cx = CAMERA_K[0, 2]
    cy = CAMERA_K[1, 2]

    image_x = (
        fx
        * right
        / forward
        + cx
    )

    image_y = (
        fy
        * (-up)
        / forward
        + cy
    )

    if not (
        np.isfinite(image_x)
        and np.isfinite(image_y)
    ):
        return None

    return (
        float(image_x),
        float(image_y),
    )


# ============================================================
# CREATE TEMPLATE AND MASK
# ============================================================

def create_template():
    """
    Create one STOP-sign template from CLEAR weather.

    The template itself keeps full BGR color.

    We also create a mask covering the octagonal sign.
    The mask removes surrounding background from the
    correlation calculation.
    """

    clear_root = DATASETS[
        "clear"
    ][
        "root"
    ]

    metadata = pd.read_csv(
        os.path.join(
            clear_root,
            "frames.csv",
        )
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

    gt_projection = (
        project_stop_sign_center(
            row
        )
    )

    if gt_projection is None:
        raise RuntimeError(
            "Could not project STOP sign "
            "during template creation."
        )

    gt_x = int(
        round(
            gt_projection[0]
        )
    )

    gt_y = int(
        round(
            gt_projection[1]
        )
    )

    # --------------------------------------------------------
    # Local region around real STOP sign
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Find red STOP sign contour inside that local region
    # --------------------------------------------------------

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
        gt_x - patch_x1
    )

    local_gt_y = (
        gt_y - patch_y1
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

        distance = math.hypot(
            center_x - local_gt_x,
            center_y - local_gt_y,
        )

        if distance < best_distance:

            best_distance = distance
            best_contour = contour

    if best_contour is None:

        raise RuntimeError(
            "Could not isolate STOP sign "
            "for template creation."
        )

    # --------------------------------------------------------
    # Crop template
    # --------------------------------------------------------

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

    # ========================================================
    # TEMPLATE MASK
    #
    # Find red sign region in the cropped template.
    # Fill its convex hull so that:
    #
    # - red body is included
    # - white STOP letters are included
    # - surrounding tree/building background is excluded
    # ========================================================

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

    # --------------------------------------------------------
    # Save fresh template files
    # --------------------------------------------------------

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

    print(
        "Template:",
        TEMPLATE_PATH,
    )

    print(
        "Mask:",
        TEMPLATE_MASK_PATH,
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
    """
    Perform masked COLOR template matching.

    TM_CCORR_NORMED supports a mask.

    We keep BGR color instead of converting everything
    to grayscale.
    """

    try:

        response = cv2.matchTemplate(
            search,
            scaled_template,
            cv2.TM_CCORR_NORMED,
            mask=scaled_mask,
        )

    except cv2.error:

        # Fallback for OpenCV versions requiring
        # same number of mask channels.
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

    # Some masked-normalized correlations can create
    # NaN/Inf values in zero-energy regions.
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
    Return the single highest-scoring multi-scale
    template match.

    IMPORTANT:
    No threshold is applied here.

    Threshold calibration happens separately using
    CLEAR weather only.
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
                original_w * scale
            )
        )

        scaled_h = int(
            round(
                original_h * scale
            )
        )

        if (
            scaled_w < 7
            or scaled_h < 7
        ):
            continue

        if (
            scaled_w >= search.shape[1]
            or scaled_h >= search.shape[0]
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
# EVALUATION AGAINST CARLA GT
# ============================================================

def candidate_matches_gt(
    candidate,
    gt_projection,
):
    """
    Evaluation only.

    The detector does not use this function when selecting
    its match.
    """

    if (
        candidate is None
        or gt_projection is None
    ):
        return False

    gt_x = gt_projection[0]
    gt_y = gt_projection[1]

    x1 = (
        candidate["x"]
        - GT_MARGIN_PX
    )

    y1 = (
        candidate["y"]
        - GT_MARGIN_PX
    )

    x2 = (
        candidate["x"]
        + candidate["w"]
        + GT_MARGIN_PX
    )

    y2 = (
        candidate["y"]
        + candidate["h"]
        + GT_MARGIN_PX
    )

    return (
        x1
        <= gt_x
        <= x2

        and

        y1
        <= gt_y
        <= y2
    )


# ============================================================
# CLEAR-ONLY THRESHOLD CALIBRATION
# ============================================================

def calibrate_threshold(
    template,
    template_mask,
):
    """
    Calibrate detection threshold using CLEAR weather ONLY.

    Fog45 and Fog90 are not examined here.

    For each clear approach frame:
        - find highest raw match
        - determine whether that raw match is the real STOP sign

    Then choose threshold maximizing F1 using:
        precision = correct / all accepted detections
        recall    = correct / all clear approach frames

    After this function returns, the threshold is frozen.
    """

    clear_root = DATASETS[
        "clear"
    ][
        "root"
    ]

    metadata = pd.read_csv(
        os.path.join(
            clear_root,
            "frames.csv",
        )
    )

    metadata = metadata[
        metadata[
            "route_s_m"
        ] <= 0.0
    ].copy()

    frame_dir = os.path.join(
        clear_root,
        "frames",
    )

    records = []

    print()
    print(
        "Running CLEAR-only threshold calibration..."
    )

    for index, row in metadata.iterrows():

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

        if raw_candidate is None:

            score = -1.0
            correct = False

        else:

            score = float(
                raw_candidate["score"]
            )

            gt_projection = (
                project_stop_sign_center(
                    row
                )
            )

            correct = (
                candidate_matches_gt(
                    raw_candidate,
                    gt_projection,
                )
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

    # Candidate thresholds taken directly from the
    # observed clear scores.

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

        # Every clear approach frame physically contains
        # the target route's STOP sign, so recall is measured
        # against all approach frames.

        recall = (
            true_positive
            / number_of_frames
        )

        if (
            precision
            + recall
            > 0
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

        # Prefer:
        #   1. larger F1
        #   2. larger precision
        #   3. higher threshold
        #
        # This avoids choosing an unnecessarily permissive
        # threshold when performance is tied.

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
        "=" * 60
    )

    print(
        "CLEAR CALIBRATION COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        "Frozen threshold:",
        f"{best_threshold:.4f}",
    )

    print(
        "Clear calibration precision:",
        f'{best_metrics["precision"] * 100:.2f}%',
    )

    print(
        "Clear calibration recall:",
        f'{best_metrics["recall"] * 100:.2f}%',
    )

    print(
        "Clear calibration F1:",
        f'{best_metrics["f1"]:.4f}',
    )

    return best_threshold


# ============================================================
# DEBUG VISUALIZATION
# ============================================================

def save_debug(
    image,
    candidate,
    gt_projection,
    detected,
    correct,
    threshold,
    output_path,
    route_s,
    distance_to_sign,
):
    """
    Blue:
        runtime search region

    Green:
        accepted correct STOP match

    Red:
        accepted false-positive match

    Gray:
        best raw match rejected by threshold

    Magenta:
        projected CARLA STOP center
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

    if candidate is not None:

        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

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

        cv2.rectangle(
            vis,
            (
                x,
                y,
            ),
            (
                x + w,
                y + h,
            ),
            color,
            3,
        )

    if gt_projection is not None:

        gt_x = int(
            round(
                gt_projection[0]
            )
        )

        gt_y = int(
            round(
                gt_projection[1]
            )
        )

        if (
            0 <= gt_x < IMAGE_W
            and
            0 <= gt_y < IMAGE_H
        ):

            cv2.circle(
                vis,
                (
                    gt_x,
                    gt_y,
                ),
                7,
                (
                    255,
                    0,
                    255,
                ),
                -1,
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

    text = (
        f"{status} | "
        f"{score_text} | "
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
        0.58,
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

    success = cv2.imwrite(
        output_path,
        vis,
    )

    if not success:

        raise RuntimeError(
            f"Could not save: "
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

    metadata = pd.read_csv(
        os.path.join(
            root,
            "frames.csv",
        )
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

        # ----------------------------------------------------
        # Runtime raw match
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Ground truth only for evaluation
        # ----------------------------------------------------

        gt_projection = (
            project_stop_sign_center(
                row
            )
        )

        if (
            detected
            and raw_candidate is not None
        ):

            correct = (
                candidate_matches_gt(
                    raw_candidate,
                    gt_projection,
                )
            )

        else:

            correct = False

        false_positive = (
            detected
            and not correct
        )

        # ----------------------------------------------------
        # Save candidate info
        # ----------------------------------------------------

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

        if gt_projection is None:

            gt_x = np.nan
            gt_y = np.nan

        else:

            gt_x = float(
                gt_projection[0]
            )

            gt_y = float(
                gt_projection[1]
            )

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

            "gt_sign_x":
                gt_x,

            "gt_sign_y":
                gt_y,

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

            "gt_projection":
                gt_projection,

            "detected":
                detected,

            "correct":
                correct,

            "route_s":
                route_s,

            "distance":
                distance_to_sign,
        }

    # ========================================================
    # APPROACH ONLY
    # ========================================================

    result_df = pd.DataFrame(
        rows
    )

    approach = result_df[
        result_df[
            "route_s_m"
        ] <= 0.0
    ].copy()

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

        # Project definition:
        # no correct detection => D(s) = 0 m
        first_distance = 0.0

    # ========================================================
    # DEBUG SAMPLES
    # ========================================================

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

    # Include first false positive if one exists.

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
            item["gt_projection"],
            item["detected"],
            item["correct"],
            threshold,
            output_path,
            item["route_s"],
            item["distance"],
        )

    # ========================================================
    # SAVE PER-FRAME RESULTS
    # ========================================================

    output_csv = os.path.join(
        OUTPUT_ROOT,
        f"{dataset_name}_per_frame.csv",
    )

    result_df.to_csv(
        output_csv,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

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

    return {
        "dataset":
            dataset_name,

        "severity":
            dataset_info[
                "severity"
            ],

        "approach_frames":
            number_of_frames,

        "detection_count":
            detection_count,

        "detection_rate":
            detection_count
            / number_of_frames,

        "correct_detection_count":
            correct_count,

        "correct_detection_rate":
            correct_count
            / number_of_frames,

        "false_positive_count":
            false_positive_count,

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

    # ========================================================
    # IMPORTANT:
    # delete the old incorrect result directory first.
    # ========================================================

    reset_output_directory()

    print()
    print(
        "=" * 60
    )

    print(
        "CREATING MASKED COLOR STOP TEMPLATE"
    )

    print(
        "=" * 60
    )

    (
        template,
        template_mask,
    ) = create_template()

    # ========================================================
    # CLEAR-ONLY CALIBRATION
    # ========================================================

    threshold = calibrate_threshold(
        template,
        template_mask,
    )

    print()
    print(
        "Threshold is now FROZEN."
    )

    print(
        "Fog45 and Fog90 do not influence calibration."
    )

    # ========================================================
    # EVALUATE ALL CONDITIONS
    # ========================================================

    summaries = []

    for (
        dataset_name,
        dataset_info,
    ) in DATASETS.items():

        print()
        print(
            "=" * 60
        )

        print(
            "Processing:",
            dataset_name,
        )

        print(
            "=" * 60
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
            "Approach frames:",
            summary[
                "approach_frames"
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
            "Correct detections:",
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

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

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
        "=" * 60
    )

    print(
        "CORRECTED TEMPLATE BASELINE COMPLETE"
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
        "Old template results were deleted "
        "before this run."
    )

    print()

    print(
        "Green = correct detection"
    )

    print(
        "Red = accepted false positive"
    )

    print(
        "Gray = best match rejected by threshold"
    )

    print(
        "Magenta = CARLA STOP-sign center"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()