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

# If Hough fails completely on a valid non-junction frame,
# KEEP that frame and assign a large failure error.
FAILURE_ERROR_PX = IMAGE_W / 2.0   # 640 px


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


OUTPUT_ROOT = "results/lane_hough"

DEBUG_DIR = os.path.join(
    OUTPUT_ROOT,
    "debug",
)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess(image):
    """
    Classical lane preprocessing:

        BGR
        -> grayscale
        -> Gaussian blur
        -> Canny
        -> trapezoidal road ROI
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

    edges = cv2.Canny(
        blurred,
        50,
        150,
    )

    # --------------------------------------------------------
    # Road ROI
    # --------------------------------------------------------

    mask = np.zeros_like(
        edges
    )

    polygon = np.array(
        [[
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
        ]],
        dtype=np.int32,
    )

    cv2.fillPoly(
        mask,
        polygon,
        255,
    )

    roi_edges = cv2.bitwise_and(
        edges,
        mask,
    )

    return (
        gray,
        edges,
        roi_edges,
        polygon,
    )


# ============================================================
# HOUGH LINE SEGMENTS
# ============================================================

def find_hough_segments(
    roi_edges,
):
    """
    Detect line segments with probabilistic Hough.

    Important:
    OpenCV versions can return shapes such as:

        (N, 1, 4)
        (N, 4)

    We reshape explicitly so every segment is always:

        [x1, y1, x2, y2]
    """

    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=30,
        minLineLength=30,
        maxLineGap=80,
    )

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
# SPLIT LEFT / RIGHT LINE SEGMENTS
# ============================================================

def split_lane_segments(
    segments,
):
    """
    Split Hough segments into left/right lane candidates.

    Image coordinates:
        x -> right
        y -> downward

    Typical perspective:
        left lane  -> negative dy/dx
        right lane -> positive dy/dx
    """

    left = []
    right = []

    for segment in segments:

        x1, y1, x2, y2 = map(
            float,
            segment,
        )

        dx = x2 - x1
        dy = y2 - y1

        # Avoid division by zero.
        if abs(dx) < 1e-6:
            continue

        slope = dy / dx

        # Ignore nearly horizontal structures.
        if abs(slope) < 0.30:
            continue

        # Reject extremely vertical/noisy segments.
        if abs(slope) > 10.0:
            continue

        midpoint_x = (
            x1 + x2
        ) / 2.0

        length = float(
            np.hypot(
                x2 - x1,
                y2 - y1,
            )
        )

        record = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "length": length,
        }

        # Left boundary
        if (
            slope < 0
            and midpoint_x < 0.70 * IMAGE_W
        ):

            left.append(
                record
            )

        # Right boundary
        elif (
            slope > 0
            and midpoint_x > 0.30 * IMAGE_W
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
    Fit:

        x = a*y + b

    using all endpoints from the selected Hough segments.

    Longer segments receive higher fitting weight.
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
    Evaluate fitted boundary x = a*y+b.
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
# COMPLETE HOUGH LANE DETECTOR
# ============================================================

def detect_lane_center(
    image,
):
    """
    Detect left/right lane boundaries and estimate
    lane-center x at LOOKAHEAD_ROW.

    Returns:

        pred_x
        left_x
        right_x
        debug
    """

    (
        gray,
        edges,
        roi_edges,
        polygon,
    ) = preprocess(
        image
    )

    segments = find_hough_segments(
        roi_edges
    )

    (
        left_segments,
        right_segments,
    ) = split_lane_segments(
        segments
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
        "edges": edges,
        "roi_edges": roi_edges,
        "polygon": polygon,
        "segments": segments,
        "left_segments": left_segments,
        "right_segments": right_segments,
        "left_model": left_model,
        "right_model": right_model,
    }

    # --------------------------------------------------------
    # Need both boundaries
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
    # Boundaries must cross the visible image
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

    # Left must really be left of right.
    if left_x >= right_x:

        return (
            None,
            left_x,
            right_x,
            debug,
        )

    # --------------------------------------------------------
    # Loose lane-width sanity check
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
# DEBUG VISUALIZATION
# ============================================================

def draw_model_line(
    image,
    model,
    color,
):
    """
    Draw fitted x = a*y+b lane boundary.
    """

    if model is None:
        return

    y_bottom = IMAGE_H - 1

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
    Save a visualization showing:

        ROI
        raw Hough segments
        fitted boundaries
        GT center
        predicted center
    """

    vis = image.copy()

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    cv2.polylines(
        vis,
        debug["polygon"],
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
    # Raw Hough segments
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
            (x1, y1),
            (x2, y2),
            (160, 160, 160),
            1,
        )

    # --------------------------------------------------------
    # Fitted lane boundaries
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
    # Ground truth
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
    # Hough prediction
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

        text = (
            f"Hough={pred_x:.1f} "
            f"GT={lane_gt_x:.1f} "
            f"err={error_px:.1f}px"
        )

    else:

        text = (
            f"HOUGH FAILURE "
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
# PROCESS ONE WEATHER CONDITION
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
            f"Missing CSV: {csv_path}"
        )

    metadata = pd.read_csv(
        csv_path
    )

    results = []

    eligible_count = 0
    success_count = 0
    failure_count = 0

    # --------------------------------------------------------
    # Process every saved image
    # --------------------------------------------------------

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
                f"Unexpected image size "
                f"for {image_path}: "
                f"{image.shape[1]}x"
                f"{image.shape[0]}"
            )

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
        # Run Hough detector
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
        # Junction frames:
        # lane GT intentionally undefined.
        # Do not include in lane-error evaluation.
        # ----------------------------------------------------

        if (
            is_junction
            or lane_gt_x is None
        ):

            status = (
                "junction_excluded"
            )

            evaluated = 0

            error_px = np.nan

        # ----------------------------------------------------
        # Valid non-junction frame
        # ----------------------------------------------------

        else:

            eligible_count += 1
            evaluated = 1

            # Detector completely failed.
            if pred_x is None:

                status = (
                    "detector_failure"
                )

                # KEEP the frame.
                error_px = (
                    FAILURE_ERROR_PX
                )

                failure_count += 1

            # Detector produced a prediction.
            else:

                status = "success"

                error_px = abs(
                    pred_x
                    - lane_gt_x
                )

                success_count += 1

        # ----------------------------------------------------
        # Save row
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

            "hough_left_x":
                left_x,

            "hough_right_x":
                right_x,

            "hough_pred_x":
                pred_x,

            "evaluated":
                evaluated,

            "status":
                status,

            "error_px":
                error_px,
        })

        # ----------------------------------------------------
        # Save representative images for inspection
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
    # SAVE PER-FRAME RESULTS
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
    # SUMMARY
    # ========================================================

    evaluated_df = results_df[
        results_df["evaluated"]
        == 1
    ]

    successful_df = evaluated_df[
        evaluated_df["status"]
        == "success"
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

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

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
        "HOUGH EVALUATION COMPLETE"
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