import os

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_W = 1280
IMAGE_H = 720

LOOKAHEAD_ROW = 504

# Same failure penalty used by the Hough baseline.
# A detector failure is NOT dropped from evaluation.
FAILURE_ERROR_PX = IMAGE_W / 2.0  # 640 px


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


OUTPUT_ROOT = "results/lane_lsd"

DEBUG_DIR = os.path.join(
    OUTPUT_ROOT,
    "debug",
)


# ============================================================
# LSD DETECTOR
# ============================================================

LSD = cv2.createLineSegmentDetector()


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(image):
    """
    Preprocessing for LSD:

        BGR
        -> grayscale
        -> Gaussian blur

    IMPORTANT:
    We DO NOT apply a hard black ROI mask before LSD.

    A hard mask would create artificial high-contrast
    diagonal edges exactly along the ROI boundary.

    Instead:
        1. LSD detects on the normal blurred image.
        2. We later keep only segments geometrically
           located inside the road ROI.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # Same ROI geometry as the Hough baseline.
    polygon = np.array(
        [
            (0, IMAGE_H - 1),

            (
                int(0.38 * IMAGE_W),
                int(0.45 * IMAGE_H),
            ),

            (
                int(0.62 * IMAGE_W),
                int(0.45 * IMAGE_H),
            ),

            (
                IMAGE_W - 1,
                IMAGE_H - 1,
            ),
        ],
        dtype=np.int32,
    )

    return (
        gray,
        blurred,
        polygon,
    )


# ============================================================
# LSD SEGMENT DETECTION
# ============================================================

def find_lsd_segments(
    blurred,
):
    """
    Detect line segments using OpenCV LSD.

    Different OpenCV versions may return:

        (N, 1, 4)

    or similar shapes.

    We force the result into:

        (N, 4)

    where every row is:

        [x1, y1, x2, y2]
    """

    result = LSD.detect(
        blurred
    )

    if result is None:
        return np.empty(
            (0, 4),
            dtype=np.float64,
        )

    lines = result[0]

    if lines is None:
        return np.empty(
            (0, 4),
            dtype=np.float64,
        )

    lines = np.asarray(
        lines,
        dtype=np.float64,
    ).reshape(-1, 4)

    return lines


# ============================================================
# ROI + LEFT / RIGHT FILTERING
# ============================================================

def split_lane_segments(
    segments,
    polygon,
):
    """
    Filter LSD segments geometrically.

    A segment is considered only if its midpoint is
    inside the same trapezoidal road ROI used by Hough.

    Then candidate lane boundaries are separated using
    their image-space slope:

        left boundary  -> negative slope
        right boundary -> positive slope
    """

    left = []
    right = []

    contour = polygon.reshape(
        (-1, 1, 2)
    )

    for segment in segments:

        x1, y1, x2, y2 = map(
            float,
            segment,
        )

        # ----------------------------------------------------
        # Segment midpoint
        # ----------------------------------------------------

        midpoint_x = (
            x1 + x2
        ) / 2.0

        midpoint_y = (
            y1 + y2
        ) / 2.0

        # ----------------------------------------------------
        # Keep only segments whose midpoint lies in the ROI.
        #
        # This preserves the original image and prevents
        # artificial ROI-border lines.
        # ----------------------------------------------------

        inside = cv2.pointPolygonTest(
            contour,
            (
                float(midpoint_x),
                float(midpoint_y),
            ),
            False,
        )

        if inside < 0:
            continue

        # ----------------------------------------------------
        # Basic line geometry
        # ----------------------------------------------------

        dx = x2 - x1
        dy = y2 - y1

        length = float(
            np.hypot(
                dx,
                dy,
            )
        )

        # Same approximate minimum segment length used
        # in our Hough baseline.
        if length < 30.0:
            continue

        if abs(dx) < 1e-6:
            continue

        slope = dy / dx

        # Ignore nearly horizontal structures.
        if abs(slope) < 0.30:
            continue

        # Ignore extreme / unstable near-vertical segments.
        if abs(slope) > 10.0:
            continue

        record = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "length": length,
        }

        # ----------------------------------------------------
        # LEFT BOUNDARY
        # ----------------------------------------------------

        if (
            slope < 0
            and midpoint_x
            < 0.70 * IMAGE_W
        ):

            left.append(
                record
            )

        # ----------------------------------------------------
        # RIGHT BOUNDARY
        # ----------------------------------------------------

        elif (
            slope > 0
            and midpoint_x
            > 0.30 * IMAGE_W
        ):

            right.append(
                record
            )

    return (
        left,
        right,
    )


# ============================================================
# FIT ONE LANE BOUNDARY
# ============================================================

def fit_boundary(
    segments,
):
    """
    Fit one lane boundary as:

        x = a*y + b

    using all endpoints from the accepted segments.

    Longer line segments receive larger fitting weight.
    """

    if not segments:
        return None

    xs = []
    ys = []
    weights = []

    for segment in segments:

        length = (
            segment["length"]
        )

        xs.extend([
            segment["x1"],
            segment["x2"],
        ])

        ys.extend([
            segment["y1"],
            segment["y2"],
        ])

        weights.extend([
            length,
            length,
        ])

    if len(xs) < 2:
        return None

    xs = np.asarray(
        xs,
        dtype=np.float64,
    )

    ys = np.asarray(
        ys,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    try:

        coefficients = np.polyfit(
            ys,
            xs,
            deg=1,
            w=weights,
        )

    except Exception:
        return None

    a, b = coefficients

    if not (
        np.isfinite(a)
        and np.isfinite(b)
    ):
        return None

    return (
        float(a),
        float(b),
    )


def boundary_x_at_row(
    model,
    row,
):
    """
    Evaluate:

        x = a*y + b

    at the requested image row.
    """

    if model is None:
        return None

    a, b = model

    x = (
        a * row
        + b
    )

    if not np.isfinite(x):
        return None

    return float(x)


# ============================================================
# COMPLETE LSD LANE DETECTOR
# ============================================================

def detect_lane_center(
    image,
):
    """
    Detect the ego-lane boundaries with LSD.

    Returns:

        predicted_center_x
        left_boundary_x
        right_boundary_x
        debug_info
    """

    (
        gray,
        blurred,
        polygon,
    ) = preprocess(
        image
    )

    # Detect on ORIGINAL blurred grayscale image.
    segments = find_lsd_segments(
        blurred
    )

    # Geometrically restrict the LSD segments to ROI.
    (
        left_segments,
        right_segments,
    ) = split_lane_segments(
        segments,
        polygon,
    )

    left_model = fit_boundary(
        left_segments
    )

    right_model = fit_boundary(
        right_segments
    )

    left_x = boundary_x_at_row(
        left_model,
        LOOKAHEAD_ROW,
    )

    right_x = boundary_x_at_row(
        right_model,
        LOOKAHEAD_ROW,
    )

    debug = {
        "gray": gray,
        "blurred": blurred,
        "polygon": polygon,
        "segments": segments,
        "left_segments": left_segments,
        "right_segments": right_segments,
        "left_model": left_model,
        "right_model": right_model,
    }

    # --------------------------------------------------------
    # Both boundaries must exist
    # --------------------------------------------------------

    if (
        left_x is None
        or right_x is None
    ):

        return (
            None,
            left_x,
            right_x,
            debug,
        )

    # --------------------------------------------------------
    # Both crossings must be inside image
    # --------------------------------------------------------

    if not (
        0.0 <= left_x < IMAGE_W
        and
        0.0 <= right_x < IMAGE_W
    ):

        return (
            None,
            left_x,
            right_x,
            debug,
        )

    # --------------------------------------------------------
    # Left must be left of right
    # --------------------------------------------------------

    if left_x >= right_x:

        return (
            None,
            left_x,
            right_x,
            debug,
        )

    # --------------------------------------------------------
    # Lane width sanity check
    # --------------------------------------------------------

    lane_width_px = (
        right_x - left_x
    )

    if not (
        120.0
        <= lane_width_px
        <= 1000.0
    ):

        return (
            None,
            left_x,
            right_x,
            debug,
        )

    # --------------------------------------------------------
    # Lane center
    # --------------------------------------------------------

    predicted_center = (
        left_x + right_x
    ) / 2.0

    if not (
        0.0
        <= predicted_center
        < IMAGE_W
    ):

        return (
            None,
            left_x,
            right_x,
            debug,
        )

    return (
        float(predicted_center),
        float(left_x),
        float(right_x),
        debug,
    )


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def draw_model_line(
    image,
    model,
    color,
):
    """
    Draw one fitted lane boundary.
    """

    if model is None:
        return

    y_bottom = (
        IMAGE_H - 1
    )

    y_top = int(
        0.45 * IMAGE_H
    )

    x_bottom = boundary_x_at_row(
        model,
        y_bottom,
    )

    x_top = boundary_x_at_row(
        model,
        y_top,
    )

    if (
        x_bottom is None
        or x_top is None
    ):
        return

    cv2.line(
        image,
        (
            int(round(x_bottom)),
            y_bottom,
        ),
        (
            int(round(x_top)),
            y_top,
        ),
        color,
        4,
    )


def save_debug_image(
    image,
    output_path,
    lane_gt_x,
    pred_x,
    error_px,
    debug,
):
    """
    Save one LSD diagnostic image.

    Visualization:

        cyan polygon = ROI
        gray lines   = raw LSD segments
        yellow lines = fitted boundaries
        red dot      = CARLA GT
        green dot    = LSD prediction
    """

    vis = image.copy()

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    cv2.polylines(
        vis,
        [
            debug["polygon"]
        ],
        True,
        (255, 255, 0),
        2,
    )

    # --------------------------------------------------------
    # Lookahead row
    # --------------------------------------------------------

    cv2.line(
        vis,
        (
            0,
            LOOKAHEAD_ROW,
        ),
        (
            IMAGE_W - 1,
            LOOKAHEAD_ROW,
        ),
        (255, 255, 0),
        2,
    )

    # --------------------------------------------------------
    # Draw RAW LSD segments lightly
    # --------------------------------------------------------

    for segment in debug[
        "segments"
    ]:

        x1, y1, x2, y2 = map(
            int,
            np.round(segment),
        )

        cv2.line(
            vis,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (150, 150, 150),
            1,
        )

    # --------------------------------------------------------
    # Fitted boundaries
    # --------------------------------------------------------

    draw_model_line(
        vis,
        debug["left_model"],
        (0, 255, 255),
    )

    draw_model_line(
        vis,
        debug["right_model"],
        (0, 255, 255),
    )

    # --------------------------------------------------------
    # GT
    # --------------------------------------------------------

    if lane_gt_x is not None:

        cv2.circle(
            vis,
            (
                int(round(lane_gt_x)),
                LOOKAHEAD_ROW,
            ),
            10,
            (0, 0, 255),
            -1,
        )

    # --------------------------------------------------------
    # LSD prediction
    # --------------------------------------------------------

    if pred_x is not None:

        cv2.circle(
            vis,
            (
                int(round(pred_x)),
                LOOKAHEAD_ROW,
            ),
            10,
            (0, 255, 0),
            -1,
        )

        if lane_gt_x is not None:

            text = (
                f"LSD={pred_x:.1f} "
                f"GT={lane_gt_x:.1f} "
                f"err={error_px:.1f}px"
            )

        else:

            text = (
                f"LSD={pred_x:.1f}"
            )

    else:

        text = (
            f"LSD FAILURE "
            f"error={error_px:.1f}px"
        )

    cv2.putText(
        vis,
        text,
        (
            30,
            50,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
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


# ============================================================
# PROCESS ONE WEATHER DATASET
# ============================================================

def process_dataset(
    dataset_name,
    dataset_info,
):
    """
    Run LSD evaluation for one weather condition.
    """

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
            f"Missing CSV: {csv_path}"
        )

    metadata = pd.read_csv(
        csv_path
    )

    results = []

    eligible_count = 0
    success_count = 0
    failure_count = 0

    # ========================================================
    # PROCESS ALL FRAMES
    # ========================================================

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
                f"Could not read image: "
                f"{image_path}"
            )

        if (
            image.shape[1] != IMAGE_W
            or image.shape[0] != IMAGE_H
        ):

            raise RuntimeError(
                f"Unexpected image dimensions: "
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        is_junction = (
            int(
                row["is_junction"]
            )
            == 1
        )

        lane_gt_raw = (
            row["lane_gt_x"]
        )

        lane_gt_x = (
            None
            if pd.isna(
                lane_gt_raw
            )
            else float(
                lane_gt_raw
            )
        )

        # ----------------------------------------------------
        # LSD prediction
        # ----------------------------------------------------

        (
            pred_x,
            left_x,
            right_x,
            debug,
        ) = detect_lane_center(
            image
        )

        # ----------------------------------------------------
        # Junction:
        #
        # lane GT is intentionally undefined.
        # Do NOT evaluate lane error here.
        # ----------------------------------------------------

        if (
            is_junction
            or lane_gt_x is None
        ):

            evaluated = 0

            status = (
                "junction_excluded"
            )

            error_px = np.nan

        # ----------------------------------------------------
        # Non-junction:
        #
        # every frame is kept in evaluation.
        # ----------------------------------------------------

        else:

            eligible_count += 1

            evaluated = 1

            if pred_x is None:

                status = (
                    "detector_failure"
                )

                error_px = (
                    FAILURE_ERROR_PX
                )

                failure_count += 1

            else:

                status = (
                    "success"
                )

                error_px = abs(
                    pred_x
                    - lane_gt_x
                )

                success_count += 1

        # ----------------------------------------------------
        # Result row
        # ----------------------------------------------------

        results.append({
            "sample_id":
                sample_id,

            "weather":
                row["weather"],

            "severity":
                int(
                    row["severity"]
                ),

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

            "is_junction":
                int(
                    row["is_junction"]
                ),

            "lane_gt_x":
                lane_gt_x,

            "lsd_left_x":
                left_x,

            "lsd_right_x":
                right_x,

            "lsd_pred_x":
                pred_x,

            "lsd_num_raw_segments":
                len(
                    debug["segments"]
                ),

            "lsd_num_left_segments":
                len(
                    debug[
                        "left_segments"
                    ]
                ),

            "lsd_num_right_segments":
                len(
                    debug[
                        "right_segments"
                    ]
                ),

            "evaluated":
                evaluated,

            "status":
                status,

            "error_px":
                error_px,
        })

        # ----------------------------------------------------
        # Representative debug frames
        #
        # Same IDs used for Hough.
        # ----------------------------------------------------

        if (
            not is_junction
            and sample_id
            in {
                1,
                20,
                40,
                60,
                80,
            }
        ):

            debug_path = os.path.join(
                DEBUG_DIR,
                dataset_name,
                f"{sample_id:04d}.png",
            )

            debug_error = (
                0.0
                if np.isnan(
                    error_px
                )
                else float(
                    error_px
                )
            )

            save_debug_image(
                image,
                debug_path,
                lane_gt_x,
                pred_x,
                debug_error,
                debug,
            )

    # ========================================================
    # PER-FRAME CSV
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    output_csv = os.path.join(
        OUTPUT_ROOT,
        f"{dataset_name}_per_frame.csv",
    )

    results_df.to_csv(
        output_csv,
        index=False,
    )

    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    evaluated_df = results_df[
        results_df[
            "evaluated"
        ] == 1
    ]

    successful_df = evaluated_df[
        evaluated_df[
            "status"
        ] == "success"
    ]

    if len(
        evaluated_df
    ) > 0:

        mean_error = float(
            evaluated_df[
                "error_px"
            ].mean()
        )

        median_error = float(
            evaluated_df[
                "error_px"
            ].median()
        )

    else:

        mean_error = np.nan
        median_error = np.nan

    if len(
        successful_df
    ) > 0:

        mean_success_error = float(
            successful_df[
                "error_px"
            ].mean()
        )

        median_success_error = float(
            successful_df[
                "error_px"
            ].median()
        )

    else:

        mean_success_error = np.nan
        median_success_error = np.nan

    detection_rate = (
        success_count
        / eligible_count
        if eligible_count > 0
        else 0.0
    )

    summary = {
        "dataset":
            dataset_name,

        "severity":
            dataset_info[
                "severity"
            ],

        "total_frames":
            len(metadata),

        "junction_excluded":
            int(
                (
                    metadata[
                        "is_junction"
                    ]
                    == 1
                ).sum()
            ),

        "evaluated_frames":
            eligible_count,

        "successful_detections":
            success_count,

        "detector_failures":
            failure_count,

        "detection_rate":
            detection_rate,

        "mean_error_px_with_failures":
            mean_error,

        "median_error_px_with_failures":
            median_error,

        "mean_error_px_success_only":
            mean_success_error,

        "median_error_px_success_only":
            median_success_error,
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
            "Total frames:",
            summary[
                "total_frames"
            ],
        )

        print(
            "Junction excluded:",
            summary[
                "junction_excluded"
            ],
        )

        print(
            "Evaluated frames:",
            summary[
                "evaluated_frames"
            ],
        )

        print(
            "Successful detections:",
            summary[
                "successful_detections"
            ],
        )

        print(
            "Detector failures:",
            summary[
                "detector_failures"
            ],
        )

        print(
            "Detection rate:",
            f'{summary["detection_rate"] * 100:.2f}%',
        )

        print(
            "Mean error "
            "(failures included):",
            f'{summary["mean_error_px_with_failures"]:.2f} px',
        )

        print(
            "Median error "
            "(failures included):",
            f'{summary["median_error_px_with_failures"]:.2f} px',
        )

        print(
            "Mean error "
            "(success only):",
            f'{summary["mean_error_px_success_only"]:.2f} px',
        )

        print(
            "Median error "
            "(success only):",
            f'{summary["median_error_px_success_only"]:.2f} px',
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
        "LSD EVALUATION COMPLETE"
    )

    print(
        "Summary:",
        summary_path,
    )

    print(
        "Debug images:",
        DEBUG_DIR,
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()