import os
import math

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

# Camera mounting used by the recorder.
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


OUTPUT_ROOT = "results/sign_hsv"

DEBUG_ROOT = os.path.join(
    OUTPUT_ROOT,
    "debug",
)


# ============================================================
# STOP3 PHYSICAL SIGN LOCATION
# ============================================================

# Previously identified visible STOP3 sign center in CARLA.
#
# IMPORTANT:
# This is used ONLY after the detector produces its output.
# It does NOT help the detector choose its candidate.

STOP_SIGN_WORLD_X = 26.04
STOP_SIGN_WORLD_Y = 75.15
STOP_SIGN_WORLD_Z = 2.31


# Fixed tolerance around the projected GT point.
#
# This is not an IoU test.
# We simply verify that the selected detection contains
# the physical STOP-sign center, allowing a few pixels
# for small distant contours / projection approximation.

GT_MARGIN_PX = 8


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

# This is still part of the detector.
#
# It removes obviously irrelevant far-left storefronts,
# but it is deliberately broad enough that the detector
# can still make mistakes.
#
# We do NOT narrow this further after seeing results.

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
# CAMERA PROJECTION
# ============================================================

def get_camera_matrix():
    """
    Build the 3x3 pinhole camera intrinsic matrix.
    """

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

    K = np.array(
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

    return K


CAMERA_K = get_camera_matrix()


def build_camera_transform_from_row(
    row,
):
    """
    Reconstruct camera world pose from the recorded
    ego vehicle pose.

    The camera had relative location:

        x = 1.5 m forward
        y = 0
        z = 1.7 m upward

    and zero relative rotation.
    """

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

    relative_camera_location = (
        carla.Location(
            x=CAMERA_X,
            y=CAMERA_Y,
            z=CAMERA_Z,
        )
    )

    camera_world_location = (
        ego_transform.transform(
            relative_camera_location
        )
    )

    camera_transform = carla.Transform(
        camera_world_location,
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

    return camera_transform


def project_stop_sign_center(
    row,
):
    """
    Project physical STOP-sign center into the RGB image.

    CARLA / Unreal camera coordinates:

        x = forward
        y = right
        z = up

    Computer-vision camera coordinates use:

        horizontal = y
        vertical   = -z
        depth      = x
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

    # Sign behind camera.
    if forward <= 0.0:
        return None

    fx = CAMERA_K[0, 0]
    fy = CAMERA_K[1, 1]

    cx = CAMERA_K[0, 2]
    cy = CAMERA_K[1, 2]

    image_x = (
        fx
        * (
            right
            / forward
        )
        + cx
    )

    image_y = (
        fy
        * (
            -up
            / forward
        )
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
        float(forward),
    )


# ============================================================
# RED MASK
# ============================================================

def make_red_mask(
    image,
):
    """
    BGR
    -> HSV
    -> two red masks
    -> opening
    -> closing
    """

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

    Notice that this function knows NOTHING about
    the CARLA ground-truth STOP-sign position.
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

        # ----------------------------------------------------
        # Aspect ratio
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Polygon
        # ----------------------------------------------------

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        if perimeter <= 0:
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

        # ----------------------------------------------------
        # Red occupancy
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Convexity
        # ----------------------------------------------------

        is_convex = bool(
            cv2.isContourConvex(
                approx
            )
        )

        # ----------------------------------------------------
        # Candidate scoring
        #
        # IMPORTANT:
        # Ground truth does not appear here.
        # ----------------------------------------------------

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

    # Runtime output = highest scoring candidate.
    return (
        candidates[0],
        mask,
        candidates,
    )


# ============================================================
# EVALUATION
# ============================================================

def candidate_matches_gt(
    candidate,
    gt_projection,
):
    """
    Evaluation ONLY.

    Correct detection if the projected physical STOP-sign
    center lies inside the selected candidate box, allowing
    a fixed small pixel margin.

    No IoU is used.
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
# DEBUG IMAGE
# ============================================================

def save_debug(
    image,
    mask,
    candidates,
    best,
    gt_projection,
    correct_detection,
    output_path,
    distance_to_sign,
    route_s,
):
    """
    Colors:

        blue    = broad detector search region
        magenta = projected physical STOP-sign center
        green   = selected candidate is correct
        red     = selected candidate is false positive
        yellow  = other runtime candidates
    """

    vis = image.copy()

    # --------------------------------------------------------
    # Runtime search region
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Other candidates
    # --------------------------------------------------------

    for candidate in candidates:

        if candidate is best:
            continue

        x = candidate["x"]
        y = candidate["y"]
        w = candidate["w"]
        h = candidate["h"]

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
            (
                0,
                255,
                255,
            ),
            1,
        )

    # --------------------------------------------------------
    # Selected runtime candidate
    # --------------------------------------------------------

    if best is not None:

        x = best["x"]
        y = best["y"]
        w = best["w"]
        h = best["h"]

        if correct_detection:

            box_color = (
                0,
                255,
                0,
            )

        else:

            box_color = (
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
            box_color,
            3,
        )

    # --------------------------------------------------------
    # Projected CARLA sign center
    # --------------------------------------------------------

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

            cv2.circle(
                vis,
                (
                    gt_x,
                    gt_y,
                ),
                12,
                (
                    255,
                    0,
                    255,
                ),
                2,
            )

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

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

    text = (
        f"{status} | "
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
        0.67,
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
            f"Could not save debug image: "
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
            f"Missing metadata: "
            f"{csv_path}"
        )

    metadata = pd.read_csv(
        csv_path
    )

    rows = []

    debug_cache = {}

    # ========================================================
    # RUN DETECTOR
    # ========================================================

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
                f"Could not read image: "
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Runtime detector
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # GT projection for evaluation
        # ----------------------------------------------------

        gt_projection = (
            project_stop_sign_center(
                row
            )
        )

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

        # ----------------------------------------------------
        # Correctness
        # ----------------------------------------------------

        correct_detection = (
            candidate_matches_gt(
                best,
                gt_projection,
            )
        )

        false_positive = (
            raw_detected
            and not correct_detection
        )

        # ----------------------------------------------------
        # Candidate information
        # ----------------------------------------------------

        if best is not None:

            bbox_x = best["x"]
            bbox_y = best["y"]
            bbox_w = best["w"]
            bbox_h = best["h"]

            area = best["area"]

            vertices = (
                best["vertices"]
            )

            red_ratio = (
                best["red_ratio"]
            )

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

            "gt_projection":
                gt_projection,

            "correct_detection":
                correct_detection,

            "distance_to_sign":
                distance_to_sign,

            "route_s":
                route_s,
        }

    # ========================================================
    # DATAFRAME
    # ========================================================

    result_df = pd.DataFrame(
        rows
    )

    approach = result_df[
        result_df[
            "route_s_m"
        ] <= 0.0
    ].copy()

    # ========================================================
    # FIRST RAW DETECTION
    # ========================================================

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

    # ========================================================
    # FIRST CORRECT DETECTION
    # ========================================================

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

        # Defined failure convention.
        first_correct_distance = 0.0

    # ========================================================
    # DEBUG FRAMES
    # ========================================================

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
            info["gt_projection"],
            info["correct_detection"],
            debug_path,
            info["distance_to_sign"],
            info["route_s"],
        )

    # ========================================================
    # SAVE PER-FRAME RESULTS
    # ========================================================

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

    # ========================================================
    # SUMMARY
    # ========================================================

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

    num_approach = len(
        approach
    )

    summary = {
        "dataset":
            dataset_name,

        "severity":
            dataset_info[
                "severity"
            ],

        "approach_frames":
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

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

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
    # SUMMARY CSV
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
        "HSV STOP-SIGN EVALUATION COMPLETE"
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
        "Green box = correct STOP detection"
    )

    print(
        "Red box   = false positive"
    )

    print(
        "Magenta dot = projected CARLA STOP center"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()