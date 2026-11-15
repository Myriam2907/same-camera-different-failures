import os
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

OUT_DIR = "results/part1_figures"

HOUGH_SUMMARY = "results/lane_hough/summary.csv"
LSD_SUMMARY = "results/lane_lsd/summary.csv"

HSV_SUMMARY = "results/sign_hsv/summary.csv"
TEMPLATE_SUMMARY = "results/sign_template/summary.csv"

SPECTRUM_SUMMARY = "results/spectrum/summary.csv"

CONDITIONS = [
    "clear",
    "fog45",
    "fog90",
]

SEVERITIES = {
    "clear": 0,
    "fog45": 45,
    "fog90": 90,
}

DISPLAY_NAMES = {
    "clear": "Clear",
    "fog45": "Fog45",
    "fog90": "Fog90",
}


# ============================================================
# DELETE OLD FIGURES
# ============================================================

def reset_output():

    if os.path.exists(OUT_DIR):

        print(
            "Deleting old figure results:",
            OUT_DIR,
        )

        shutil.rmtree(
            OUT_DIR
        )

    os.makedirs(
        OUT_DIR,
        exist_ok=True,
    )


# ============================================================
# HELPERS
# ============================================================

def ordered_rows(df):

    rows = []

    for condition in CONDITIONS:

        match = df[
            df["dataset"] == condition
        ]

        if len(match) != 1:

            raise RuntimeError(
                f"Expected one row for {condition}, "
                f"found {len(match)}"
            )

        rows.append(
            match.iloc[0]
        )

    return rows


def save_figure(
    filename,
):

    path = os.path.join(
        OUT_DIR,
        filename,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        path,
    )


# ============================================================
# FIGURE 1
# QUALITATIVE 3 x 2 GRID
# ============================================================

def choose_common_debug_sample():
    """
    Choose one sample that already has:
        Hough debug image
        HSV debug image

    for all three weather conditions.

    Prefer a far-distance frame because it makes
    sign degradation easy to see.
    """

    preferred = [
        1,
        20,
        40,
        60,
        80,
    ]

    for sample_id in preferred:

        exists = True

        for condition in CONDITIONS:

            hough_path = (
                f"results/lane_hough/debug/"
                f"{condition}/{sample_id:04d}.png"
            )

            hsv_path = (
                f"results/sign_hsv/debug/"
                f"{condition}/{sample_id:04d}.png"
            )

            if not (
                os.path.exists(hough_path)
                and os.path.exists(hsv_path)
            ):
                exists = False
                break

        if exists:
            return sample_id

    raise RuntimeError(
        "Could not find a common debug sample."
    )


def get_sample_distance(
    sample_id,
):

    frames_csv = (
        "data/recordings/"
        "stop3_clear_0/frames.csv"
    )

    df = pd.read_csv(
        frames_csv
    )

    row = df[
        df["sample_id"] == sample_id
    ]

    if len(row) != 1:
        return None

    return float(
        row.iloc[0][
            "distance_to_sign_m"
        ]
    )


def make_qualitative_grid():

    sample_id = (
        choose_common_debug_sample()
    )

    distance = get_sample_distance(
        sample_id
    )

    print()
    print(
        "Qualitative grid sample:",
        sample_id,
    )

    if distance is not None:

        print(
            "Distance to STOP sign:",
            f"{distance:.2f} m",
        )

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 8),
    )

    for column, condition in enumerate(
        CONDITIONS
    ):

        # ====================================================
        # TOP ROW: HOUGH
        # ====================================================

        hough_path = (
            f"results/lane_hough/debug/"
            f"{condition}/{sample_id:04d}.png"
        )

        image = cv2.imread(
            hough_path
        )

        if image is None:

            raise RuntimeError(
                f"Missing image: "
                f"{hough_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        axes[0, column].imshow(
            image
        )

        axes[0, column].set_title(
            DISPLAY_NAMES[
                condition
            ],
            fontsize=14,
        )

        axes[0, column].axis(
            "off"
        )

        # ====================================================
        # BOTTOM ROW: HSV SIGN DETECTION
        # ====================================================

        hsv_path = (
            f"results/sign_hsv/debug/"
            f"{condition}/{sample_id:04d}.png"
        )

        image = cv2.imread(
            hsv_path
        )

        if image is None:

            raise RuntimeError(
                f"Missing image: "
                f"{hsv_path}"
            )

        # Crop broad STOP-sign search area.
        #
        # Same crop for all three weather conditions.
        h, w = image.shape[:2]

        x1 = int(
            0.42 * w
        )

        x2 = int(
            0.99 * w
        )

        y1 = int(
            0.15 * h
        )

        y2 = int(
            0.90 * h
        )

        image = image[
            y1:y2,
            x1:x2,
        ]

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        axes[1, column].imshow(
            image
        )

        axes[1, column].axis(
            "off"
        )

    axes[0, 0].set_ylabel(
        "Hough lane",
        fontsize=14,
    )

    axes[1, 0].set_ylabel(
        "HSV STOP",
        fontsize=14,
    )

    if distance is None:

        title = (
            f"Matched Perception Outputs "
            f"(sample {sample_id})"
        )

    else:

        title = (
            "Matched Perception Outputs "
            f"at the Same Camera Pose "
            f"({distance:.2f} m from STOP sign)"
        )

    fig.suptitle(
        title,
        fontsize=16,
    )

    fig.subplots_adjust(
        top=0.90
    )

    path = os.path.join(
        OUT_DIR,
        "figure_01_qualitative_grid.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        path,
    )


# ============================================================
# FIGURE 2
# NORMALIZED DIVERGENCE
# ============================================================

def make_divergence_figure():

    hough = pd.read_csv(
        HOUGH_SUMMARY
    )

    lsd = pd.read_csv(
        LSD_SUMMARY
    )

    hsv = pd.read_csv(
        HSV_SUMMARY
    )

    template = pd.read_csv(
        TEMPLATE_SUMMARY
    )

    spectrum = pd.read_csv(
        SPECTRUM_SUMMARY
    )

    hough_rows = ordered_rows(
        hough
    )

    lsd_rows = ordered_rows(
        lsd
    )

    hsv_rows = ordered_rows(
        hsv
    )

    template_rows = ordered_rows(
        template
    )

    spectrum_rows = ordered_rows(
        spectrum
    )

    # --------------------------------------------------------
    # Extract measured quantities
    # --------------------------------------------------------

    hough_rate = np.array([
        float(
            row["detection_rate"]
        )
        for row in hough_rows
    ])

    lsd_rate = np.array([
        float(
            row["detection_rate"]
        )
        for row in lsd_rows
    ])

    hsv_correct_rate = np.array([
        float(
            row[
                "correct_detection_rate"
            ]
        )
        for row in hsv_rows
    ])

    template_correct_rate = np.array([
        float(
            row[
                "correct_detection_rate"
            ]
        )
        for row in template_rows
    ])

    spectral_distance = np.array([
        float(
            row[
                "mean_spectral_distance"
            ]
        )
        for row in spectrum_rows
    ])

    # --------------------------------------------------------
    # Normalize every metric to its own Clear value.
    #
    # Clear = 1.0 for every curve.
    # --------------------------------------------------------

    hough_norm = (
        hough_rate
        / hough_rate[0]
    )

    lsd_norm = (
        lsd_rate
        / lsd_rate[0]
    )

    hsv_norm = (
        hsv_correct_rate
        / hsv_correct_rate[0]
    )

    template_norm = (
        template_correct_rate
        / template_correct_rate[0]
    )

    fourier_norm = (
        spectral_distance
        / spectral_distance[0]
    )

    x = np.array(
        [0, 45, 90]
    )

    plt.figure(
        figsize=(9, 6),
    )

    plt.plot(
        x,
        fourier_norm,
        marker="o",
        linewidth=2,
        label="Fourier input distance",
    )

    plt.plot(
        x,
        hough_norm,
        marker="o",
        linewidth=2,
        label="Hough output rate",
    )

    plt.plot(
        x,
        lsd_norm,
        marker="o",
        linewidth=2,
        label="LSD output rate",
    )

    plt.plot(
        x,
        hsv_norm,
        marker="o",
        linewidth=2,
        label="HSV correct-detection rate",
    )

    plt.plot(
        x,
        template_norm,
        marker="o",
        linewidth=2,
        label="Template correct-detection rate",
    )

    # Clear normalized level.
    plt.axhline(
        1.0,
        linestyle="--",
        linewidth=1,
        label="Clear-condition level",
    )

    # Log scale keeps the very large Fourier increase
    # and the large template decrease visible together.
    plt.yscale(
        "log"
    )

    plt.xlabel(
        "Fog severity"
    )

    plt.ylabel(
        "Value normalized to Clear (= 1)"
    )

    plt.title(
        "Same Input Degradation, Different Algorithm Responses"
    )

    plt.xticks(
        [0, 45, 90],
        [
            "Clear",
            "Fog45",
            "Fog90",
        ],
    )

    plt.grid(
        True,
        which="both",
        alpha=0.25,
    )

    plt.legend(
        fontsize=9,
    )

    save_figure(
        "figure_02_divergence.png"
    )


# ============================================================
# FIGURE 3
# STOP SIGN DETECTION RANGE
# ============================================================

def make_range_figure():

    hsv = pd.read_csv(
        HSV_SUMMARY
    )

    template = pd.read_csv(
        TEMPLATE_SUMMARY
    )

    hsv_rows = ordered_rows(
        hsv
    )

    template_rows = ordered_rows(
        template
    )

    hsv_range = np.array([
        float(
            row[
                "first_correct_detection_distance_m"
            ]
        )
        for row in hsv_rows
    ])

    template_range = np.array([
        float(
            row[
                "first_correct_detection_distance_m"
            ]
        )
        for row in template_rows
    ])

    x = np.arange(
        3
    )

    width = 0.34

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    bars_hsv = ax.bar(
        x - width / 2,
        hsv_range,
        width,
        label="HSV",
    )

    bars_template = ax.bar(
        x + width / 2,
        template_range,
        width,
        label="Template matching",
    )

    # --------------------------------------------------------
    # Mark Clear bars as censored.
    # --------------------------------------------------------

    bars_hsv[0].set_hatch(
        "//"
    )

    bars_template[0].set_hatch(
        "//"
    )

    # Labels for every bar
    for index, bar in enumerate(
        bars_hsv
    ):

        value = hsv_range[
            index
        ]

        label = (
            f"≥{value:.2f}"
            if index == 0
            else f"{value:.2f}"
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.8,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for index, bar in enumerate(
        bars_template
    ):

        value = template_range[
            index
        ]

        label = (
            f"≥{value:.2f}"
            if index == 0
            else f"{value:.2f}"
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + 0.8,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Arrow to make censoring visually obvious.
    for bar in [
        bars_hsv[0],
        bars_template[0],
    ]:

        x_arrow = (
            bar.get_x()
            + bar.get_width() / 2
        )

        y_arrow = (
            bar.get_height()
        )

        ax.annotate(
            "",
            xy=(
                x_arrow,
                y_arrow + 5,
            ),
            xytext=(
                x_arrow,
                y_arrow + 0.5,
            ),
            arrowprops={
                "arrowstyle": "->",
            },
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Clear",
            "Fog45",
            "Fog90",
        ]
    )

    ax.set_ylabel(
        "First correct detection distance (m)"
    )

    ax.set_title(
        "STOP-Sign Detection Range Under Fog"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    ax.text(
        0.02,
        0.97,
        "Hatched Clear bars are right-censored: "
        "true range ≥ 46.81 m",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    path = os.path.join(
        OUT_DIR,
        "figure_03_sign_detection_range.png",
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        path,
    )


# ============================================================
# FIGURE 4
# DEGENERATE LANE METRIC
# ============================================================

def make_degenerate_metric_figure():

    hough = pd.read_csv(
        HOUGH_SUMMARY
    )

    lsd = pd.read_csv(
        LSD_SUMMARY
    )

    hough_rows = ordered_rows(
        hough
    )

    lsd_rows = ordered_rows(
        lsd
    )

    hough_error = np.array([
        float(
            row[
                "mean_error_px_success_only"
            ]
        )
        for row in hough_rows
    ])

    lsd_error = np.array([
        float(
            row[
                "mean_error_px_success_only"
            ]
        )
        for row in lsd_rows
    ])

    # Measured from the 99-frame trivial-baseline guard.
    constant_error = np.array(
        [
            0.00033758585858446497,
            0.00033758585858446497,
            0.00033758585858446497,
        ]
    )

    x = np.arange(
        3
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(8, 6),
    )

    ax.bar(
        x - width,
        hough_error,
        width,
        label="Hough",
    )

    ax.bar(
        x,
        lsd_error,
        width,
        label="LSD",
    )

    ax.bar(
        x + width,
        constant_error,
        width,
        label="Constant x=640",
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        [
            "Clear",
            "Fog45",
            "Fog90",
        ]
    )

    ax.set_ylabel(
        "Mean lane-center error (pixels, log scale)"
    )

    ax.set_title(
        "Sanity Check: Lane-Center Metric Is Degenerate"
    )

    ax.grid(
        axis="y",
        which="both",
        alpha=0.25,
    )

    ax.legend()

    ax.text(
        0.02,
        0.03,
        "The trivial constant predictor outperforms "
        "both vision pipelines by orders of magnitude.",
        transform=ax.transAxes,
        fontsize=9,
    )

    path = os.path.join(
        OUT_DIR,
        "figure_04_degenerate_lane_metric.png",
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "Saved:",
        path,
    )


# ============================================================
# PRELIMINARY ROC / AUC
# ============================================================

def roc_curve_manual(
    scores,
    labels,
):
    """
    labels:
        1 = detector failure
        0 = detector correct

    scores:
        larger spectral distance should predict failure
    """

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    labels = np.asarray(
        labels,
        dtype=np.int32,
    )

    positives = int(
        np.sum(
            labels == 1
        )
    )

    negatives = int(
        np.sum(
            labels == 0
        )
    )

    if (
        positives == 0
        or negatives == 0
    ):

        raise RuntimeError(
            "ROC requires both failure and "
            "correct frames."
        )

    thresholds = np.concatenate(
        [
            [
                np.inf
            ],

            np.sort(
                np.unique(
                    scores
                )
            )[::-1],

            [
                -np.inf
            ],
        ]
    )

    fpr = []
    tpr = []

    for threshold in thresholds:

        predicted_failure = (
            scores >= threshold
        )

        tp = np.sum(
            predicted_failure
            & (
                labels == 1
            )
        )

        fp = np.sum(
            predicted_failure
            & (
                labels == 0
            )
        )

        tpr.append(
            tp / positives
        )

        fpr.append(
            fp / negatives
        )

    fpr = np.asarray(
        fpr,
        dtype=np.float64,
    )

    tpr = np.asarray(
        tpr,
        dtype=np.float64,
    )

    order = np.argsort(
        fpr
    )

    fpr = fpr[
        order
    ]

    tpr = tpr[
        order
    ]

    auc = float(
        np.trapezoid(
            tpr,
            fpr,
        )
    )

    return (
        fpr,
        tpr,
        auc,
    )


def load_monitor_detector_pairs(
    detector_name,
    conditions,
):

    merged_parts = []

    for condition in conditions:

        spectrum_path = (
            f"results/spectrum/"
            f"{condition}_per_frame.csv"
        )

        if detector_name == "HSV":

            detector_path = (
                f"results/sign_hsv/"
                f"{condition}_per_frame.csv"
            )

        elif detector_name == "Template":

            detector_path = (
                f"results/sign_template/"
                f"{condition}_per_frame.csv"
            )

        else:

            raise RuntimeError(
                "Unknown detector."
            )

        spectrum = pd.read_csv(
            spectrum_path
        )

        detector = pd.read_csv(
            detector_path
        )

        # Approach frames only.
        detector = detector[
            detector[
                "route_s_m"
            ] <= 0.0
        ].copy()

        merged = detector.merge(
            spectrum[
                [
                    "sample_id",
                    "spectral_distance",
                ]
            ],
            on="sample_id",
            how="inner",
        )

        merged[
            "condition"
        ] = condition

        merged_parts.append(
            merged
        )

    return pd.concat(
        merged_parts,
        ignore_index=True,
    )


def make_preliminary_roc():
    """
    Recommended preliminary ROC:
        use Fog45 + Fog90 only.

    This avoids using the Clear template calibration
    sequence as an independent test set.

    Failure label:
        1 if correct_detection == 0
        0 if correct_detection == 1
    """

    fog_conditions = [
        "fog45",
        "fog90",
    ]

    results = []

    plt.figure(
        figsize=(7, 6),
    )

    for detector_name in [
        "HSV",
        "Template",
    ]:

        data = (
            load_monitor_detector_pairs(
                detector_name,
                fog_conditions,
            )
        )

        scores = data[
            "spectral_distance"
        ].to_numpy()

        failures = (
            1
            - data[
                "correct_detection"
            ].to_numpy(
                dtype=np.int32
            )
        )

        (
            fpr,
            tpr,
            auc,
        ) = roc_curve_manual(
            scores,
            failures,
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"{detector_name} "
                f"(AUC={auc:.3f})"
            ),
        )

        results.append({
            "detector":
                detector_name,

            "conditions":
                "fog45+fog90",

            "frames":
                len(data),

            "failure_frames":
                int(
                    failures.sum()
                ),

            "correct_frames":
                int(
                    (
                        failures == 0
                    ).sum()
                ),

            "auc":
                auc,
        })

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
        label="Chance",
    )

    plt.xlabel(
        "False-positive rate"
    )

    plt.ylabel(
        "True-positive rate"
    )

    plt.title(
        "Preliminary ROC: Can Fourier Distance "
        "Predict STOP-Sign Failure?"
    )

    plt.grid(
        alpha=0.25,
    )

    plt.legend()

    save_figure(
        "figure_05_preliminary_roc.png"
    )

    auc_df = pd.DataFrame(
        results
    )

    auc_path = os.path.join(
        OUT_DIR,
        "preliminary_auc.csv",
    )

    auc_df.to_csv(
        auc_path,
        index=False,
    )

    print()
    print(
        "Preliminary fog-only AUC:"
    )

    print(
        auc_df.to_string(
            index=False
        )
    )

    print(
        "Saved:",
        auc_path,
    )


# ============================================================
# WRITE NOTES / CAPTIONS
# ============================================================

def write_notes():

    path = os.path.join(
        OUT_DIR,
        "figure_notes.txt",
    )

    text = """
FIGURE 1 — QUALITATIVE GRID
Matched Clear, Fog45, and Fog90 frames at exactly the same
camera pose. Top row shows Hough lane output. Bottom row shows
HSV STOP-sign output. The matched geometry isolates weather
as the changing input condition.

FIGURE 2 — DIVERGENCE
Each metric is normalized to its own Clear value. Fourier
spectral distance increases strongly as fog increases, while
Hough output rate changes non-monotonically, LSD output rate
decreases, and both STOP-sign correct-detection rates collapse.
This is the main visual argument that input degradation and
perception reliability are not equivalent.

FIGURE 3 — STOP RANGE
The Clear bars are censored because both detectors were already
correct at the first and farthest evaluated frame. Therefore,
the clear range is >= 46.81 m rather than exactly 46.81 m.

FIGURE 4 — DEGENERATE LANE METRIC
The constant x=640 predictor has approximately 0.00034 px error,
orders of magnitude below Hough and LSD. This demonstrates that
the frozen centerline geometry makes lane-center localization
error unsuitable as a primary metric in Part I.

FIGURE 5 — PRELIMINARY ROC
Uses Fog45 and Fog90 frames only. The positive class is a
STOP-sign detector failure. Larger Fourier spectral distance is
treated as stronger evidence of failure. This is preliminary:
one scene, one deterministic route, and one trial per condition.
"""

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            text.strip()
            + "\n"
        )

    print(
        "Saved:",
        path,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    reset_output()

    print()
    print(
        "=" * 70
    )

    print(
        "GENERATING PART-I FIGURES"
    )

    print(
        "=" * 70
    )

    make_qualitative_grid()

    make_divergence_figure()

    make_range_figure()

    make_degenerate_metric_figure()

    # Preliminary trustworthiness analysis.
    make_preliminary_roc()

    write_notes()

    print()
    print(
        "=" * 70
    )

    print(
        "ALL PART-I FIGURES COMPLETE"
    )

    print(
        "Output directory:",
        OUT_DIR,
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()