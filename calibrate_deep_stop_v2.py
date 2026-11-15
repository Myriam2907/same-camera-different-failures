#!/usr/bin/env python3
"""
Pilot confidence calibration + distance-binned analysis for YOLO26s and RF-DETR-L.

What this script does:
1) Uses ONLY already-saved raw detections. No neural inference is rerun.
2) Finds two Clear-only max-F1 operating thresholds:
      A. all evaluable Clear frames
      B. Clear frames with distance_to_sign_m <= 40 m
3) Applies each frozen threshold to every fog condition.
4) Saves distance-binned performance so far/easy frames are not mixed together.
5) Preserves the original definition of raw confidence:
      max_raw_stop_confidence = 0.0 when there is no raw STOP candidate.

Primary pilot threshold:
- "all_frames" is the primary one because it matches the current template
  calibration objective/denominator.
- "within40" is a sensitivity analysis showing how much the operating point
  changes when calibration excludes very distant frames.

Final experiments still need separate calibration and held-out trials.
"""

from pathlib import Path
import numpy as np
import pandas as pd

IOU_THRESHOLD = 0.50
WITHIN_DISTANCE_M = 40.0

MODELS = {
    "yolo26s": {
        "root": Path("results/sign_yolo_iou"),
        "output": Path("results/sign_yolo_iou_calibrated_v2"),
    },
    "rfdetr_l": {
        "root": Path("results/sign_rfdetr_iou"),
        "output": Path("results/sign_rfdetr_iou_calibrated_v2"),
    },
}

DATASET_ORDER = [
    ("clear_drift", 0),
    ("fog15_drift", 15),
    ("fog30_drift", 30),
    ("fog45_drift", 45),
    ("fog60_drift", 60),
    ("fog75_drift", 75),
    ("fog90_drift", 90),
]

DISTANCE_BINS = [0, 10, 20, 30, 40, 50, 60, np.inf]
DISTANCE_LABELS = [
    "0-10m",
    "10-20m",
    "20-30m",
    "30-40m",
    "40-50m",
    "50-60m",
    "60m+",
]


def evaluate_threshold(frame_df, box_df, threshold):
    """
    Evaluate one threshold at frame level.

    A frame is correct if at least one accepted STOP candidate has IoU >= 0.50.
    """

    grouped = {
        int(sample_id): group
        for sample_id, group in box_df.groupby("sample_id")
    }

    rows = []

    for _, frame in frame_df.iterrows():
        sid = int(frame["sample_id"])
        boxes = grouped.get(sid)

        if boxes is None:
            accepted = None
            detected = False
            correct = False
            false_positive_boxes = 0
            best_accepted_iou = 0.0
            max_accepted_confidence = np.nan

        else:
            accepted = boxes[
                boxes["confidence"] >= threshold
            ]

            detected = len(accepted) > 0

            if detected:
                best_accepted_iou = float(
                    accepted["iou_with_gt"].max()
                )

                max_accepted_confidence = float(
                    accepted["confidence"].max()
                )

                correct = bool(
                    (
                        accepted["iou_with_gt"]
                        >= IOU_THRESHOLD
                    ).any()
                )

                false_positive_boxes = int(
                    (
                        accepted["iou_with_gt"]
                        < IOU_THRESHOLD
                    ).sum()
                )

            else:
                correct = False
                false_positive_boxes = 0
                best_accepted_iou = 0.0
                max_accepted_confidence = np.nan

        # IMPORTANT:
        # Keep raw confidence definition identical to original inference scripts.
        # Those scripts save 0.0 when no raw candidate exists.
        if "max_raw_stop_confidence" in frame.index:
            raw_conf = float(
                frame["max_raw_stop_confidence"]
            )
        else:
            raise RuntimeError(
                "Expected max_raw_stop_confidence in saved frame CSV."
            )

        rows.append({
            "sample_id": sid,
            "route_s_m": float(frame["route_s_m"]),
            "distance_to_sign_m": float(
                frame["distance_to_sign_m"]
            ),
            "max_raw_stop_confidence": raw_conf,
            "detected": int(detected),
            "correct": int(correct),
            "false_positive_frame": int(
                detected and not correct
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
        })

    out = pd.DataFrame(rows)

    n = len(out)
    detected_frames = int(out["detected"].sum())
    correct_frames = int(out["correct"].sum())
    fp_frames = int(
        out["false_positive_frame"].sum()
    )

    precision = (
        correct_frames / detected_frames
        if detected_frames > 0
        else 0.0
    )

    recall = (
        correct_frames / n
        if n > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    metrics = {
        "threshold": float(threshold),
        "evaluable_frames": n,
        "detected_frames": detected_frames,
        "correct_frames": correct_frames,
        "false_positive_frames": fp_frames,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    return out, metrics


def choose_threshold(clear_frame_df, clear_box_df):
    """
    Maximize F1. Tie-break by precision, then by higher threshold.
    This matches the current template calibration logic.
    """

    thresholds = np.arange(
        0.001,
        1.000,
        0.001,
    )

    records = []

    for threshold in thresholds:
        _, metrics = evaluate_threshold(
            clear_frame_df,
            clear_box_df,
            threshold,
        )
        records.append(metrics)

    scan_df = pd.DataFrame(records)

    best = (
        scan_df.sort_values(
            ["f1", "precision", "threshold"],
            ascending=[False, False, False],
        )
        .iloc[0]
    )

    return float(best["threshold"]), scan_df


def make_distance_bins(calibrated_df):
    df = calibrated_df.copy()

    df["distance_bin"] = pd.cut(
        df["distance_to_sign_m"],
        bins=DISTANCE_BINS,
        labels=DISTANCE_LABELS,
        right=False,
        include_lowest=True,
    )

    rows = []

    for label in DISTANCE_LABELS:
        group = df[
            df["distance_bin"] == label
        ]

        n = len(group)

        if n == 0:
            continue

        detected = int(
            group["detected"].sum()
        )

        correct = int(
            group["correct"].sum()
        )

        fp_frames = int(
            group["false_positive_frame"].sum()
        )

        fp_boxes = int(
            group["false_positive_boxes"].sum()
        )

        rows.append({
            "distance_bin": label,
            "frames": n,
            "mean_distance_m": float(
                group["distance_to_sign_m"].mean()
            ),
            "detection_frames": detected,
            "detection_rate_pct": (
                100.0 * detected / n
            ),
            "correct_frames": correct,
            "correct_detection_rate_pct": (
                100.0 * correct / n
            ),
            "false_positive_frames": (
                fp_frames
            ),
            "false_positive_boxes": (
                fp_boxes
            ),
            "mean_raw_stop_confidence": float(
                group[
                    "max_raw_stop_confidence"
                ].mean()
            ),
        })

    return pd.DataFrame(rows)


def summarize_condition(
    calibrated_df,
    model_name,
    dataset_name,
    severity,
    threshold,
    calibration_scope,
):
    n = len(calibrated_df)

    detected = int(
        calibrated_df["detected"].sum()
    )

    correct = int(
        calibrated_df["correct"].sum()
    )

    fp_frames = int(
        calibrated_df[
            "false_positive_frame"
        ].sum()
    )

    fp_boxes = int(
        calibrated_df[
            "false_positive_boxes"
        ].sum()
    )

    correct_df = calibrated_df[
        calibrated_df["correct"] == 1
    ].sort_values("sample_id")

    if len(correct_df) > 0:
        first = correct_df.iloc[0]

        first_sample = int(
            first["sample_id"]
        )

        first_route = float(
            first["route_s_m"]
        )

        first_distance = float(
            first["distance_to_sign_m"]
        )

    else:
        first_sample = np.nan
        first_route = np.nan
        first_distance = 0.0

    detected_df = calibrated_df[
        calibrated_df["detected"] == 1
    ]

    mean_best_iou = (
        float(
            detected_df[
                "best_accepted_iou"
            ].mean()
        )
        if len(detected_df) > 0
        else np.nan
    )

    precision = (
        correct / detected
        if detected > 0
        else 0.0
    )

    recall = (
        correct / n
        if n > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "model": model_name,
        "dataset": dataset_name,
        "severity": severity,
        "calibration_scope": calibration_scope,
        "calibrated_conf_threshold": threshold,
        "evaluable_frames": n,
        "detection_frames": detected,
        "detection_rate_pct": (
            100.0 * detected / n
            if n
            else 0.0
        ),
        "correct_frames": correct,
        "correct_detection_rate_pct": (
            100.0 * correct / n
            if n
            else 0.0
        ),
        "false_positive_frames": fp_frames,
        "false_positive_boxes": fp_boxes,
        "precision_pct": 100.0 * precision,
        "recall_pct": 100.0 * recall,
        "f1": f1,
        "mean_raw_stop_confidence_all_frames": float(
            calibrated_df[
                "max_raw_stop_confidence"
            ].mean()
        ),
        "mean_best_iou_on_detected_frames": (
            mean_best_iou
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


def run_scope(
    model_name,
    root,
    output,
    threshold,
    scope_name,
):
    scope_dir = output / scope_name

    scope_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []
    all_bins = []

    for dataset_name, severity in DATASET_ORDER:
        frame_df = pd.read_csv(
            root
            / f"{dataset_name}_frames.csv"
        )

        box_df = pd.read_csv(
            root
            / f"{dataset_name}_raw_boxes.csv"
        )

        calibrated_df, _ = evaluate_threshold(
            frame_df,
            box_df,
            threshold,
        )

        calibrated_df.insert(
            0,
            "severity",
            severity,
        )

        calibrated_df.insert(
            0,
            "dataset",
            dataset_name,
        )

        calibrated_df.to_csv(
            scope_dir
            / f"{dataset_name}_frames.csv",
            index=False,
        )

        summary = summarize_condition(
            calibrated_df,
            model_name,
            dataset_name,
            severity,
            threshold,
            scope_name,
        )

        summaries.append(summary)

        bins = make_distance_bins(
            calibrated_df
        )

        bins.insert(
            0,
            "severity",
            severity,
        )

        bins.insert(
            0,
            "dataset",
            dataset_name,
        )

        bins.insert(
            0,
            "model",
            model_name,
        )

        bins.insert(
            0,
            "calibration_scope",
            scope_name,
        )

        all_bins.append(bins)

    summary_df = pd.DataFrame(
        summaries
    )

    d0 = float(
        summary_df.loc[
            summary_df["severity"] == 0,
            "first_correct_detection_distance_m",
        ].iloc[0]
    )

    if d0 > 0:
        summary_df["R_sign"] = (
            summary_df[
                "first_correct_detection_distance_m"
            ]
            / d0
        ).clip(
            upper=1.0
        )
    else:
        summary_df["R_sign"] = np.nan

    summary_df.to_csv(
        scope_dir / "summary.csv",
        index=False,
    )

    bins_df = pd.concat(
        all_bins,
        ignore_index=True,
    )

    bins_df.to_csv(
        scope_dir / "distance_bins.csv",
        index=False,
    )

    return summary_df, bins_df


def process_model(model_name, config):
    root = config["root"]
    output = config["output"]

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    clear_frame_df = pd.read_csv(
        root / "clear_drift_frames.csv"
    )

    clear_box_df = pd.read_csv(
        root / "clear_drift_raw_boxes.csv"
    )

    # --------------------------------------------------------
    # Calibration A: all evaluable Clear frames
    # --------------------------------------------------------

    threshold_all, scan_all = choose_threshold(
        clear_frame_df,
        clear_box_df,
    )

    scan_all.to_csv(
        output
        / "clear_threshold_scan_all_frames.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Calibration B: only <=40 m Clear frames
    # --------------------------------------------------------

    clear_40 = clear_frame_df[
        clear_frame_df[
            "distance_to_sign_m"
        ] <= WITHIN_DISTANCE_M
    ].copy()

    threshold_40, scan_40 = choose_threshold(
        clear_40,
        clear_box_df,
    )

    scan_40.to_csv(
        output
        / "clear_threshold_scan_within40m.csv",
        index=False,
    )

    best_all = (
        scan_all.iloc[
            (
                scan_all["threshold"]
                - threshold_all
            ).abs().argmin()
        ]
    )

    best_40 = (
        scan_40.iloc[
            (
                scan_40["threshold"]
                - threshold_40
            ).abs().argmin()
        ]
    )

    comparison = pd.DataFrame([
        {
            "model": model_name,
            "calibration_scope": "all_frames",
            "frames_used": len(
                clear_frame_df
            ),
            "threshold": threshold_all,
            "clear_precision_pct": (
                100.0
                * best_all["precision"]
            ),
            "clear_recall_pct": (
                100.0
                * best_all["recall"]
            ),
            "clear_f1": best_all["f1"],
        },
        {
            "model": model_name,
            "calibration_scope": "within40m",
            "frames_used": len(
                clear_40
            ),
            "threshold": threshold_40,
            "clear_precision_pct": (
                100.0
                * best_40["precision"]
            ),
            "clear_recall_pct": (
                100.0
                * best_40["recall"]
            ),
            "clear_f1": best_40["f1"],
        },
    ])

    comparison["threshold_difference"] = (
        abs(
            threshold_all
            - threshold_40
        )
    )

    comparison.to_csv(
        output
        / "calibration_scope_comparison.csv",
        index=False,
    )

    print()
    print("=" * 84)
    print(
        f"{model_name}: CALIBRATION COMPARISON"
    )
    print("=" * 84)

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    # Primary, matching current template calibration:
    all_summary, all_bins = run_scope(
        model_name,
        root,
        output,
        threshold_all,
        "all_frames",
    )

    # Sensitivity analysis:
    in40_summary, in40_bins = run_scope(
        model_name,
        root,
        output,
        threshold_40,
        "within40m",
    )

    print()
    print(
        f"{model_name} PRIMARY TABLE "
        f"(all-frame max-F1 threshold)"
    )

    print(
        all_summary[
            [
                "severity",
                "correct_detection_rate_pct",
                "false_positive_frames",
                "mean_raw_stop_confidence_all_frames",
                "first_correct_detection_distance_m",
                "R_sign",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    print()
    print(
        f"{model_name} CLEAR DISTANCE BINS "
        f"(primary threshold)"
    )

    print(
        all_bins[
            all_bins["severity"] == 0
        ][
            [
                "distance_bin",
                "frames",
                "correct_detection_rate_pct",
                "false_positive_frames",
                "mean_raw_stop_confidence",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.4f}"
            ),
        )
    )

    return (
        comparison,
        all_summary,
        in40_summary,
        all_bins,
        in40_bins,
    )


def main():
    print("=" * 84)
    print(
        "DEEP STOP CALIBRATION + DISTANCE-BIN ANALYSIS"
    )
    print("=" * 84)

    print(
        "No YOLO or RF-DETR inference will be rerun."
    )

    print(
        "Primary criterion: max-F1 on all evaluable Clear frames."
    )

    print(
        "Sensitivity criterion: max-F1 on Clear frames <= 40 m."
    )

    print()

    comparisons = []
    primary_summaries = []
    sensitivity_summaries = []
    primary_bins = []
    sensitivity_bins = []

    for model_name, config in MODELS.items():
        (
            comparison,
            primary_summary,
            sensitivity_summary,
            p_bins,
            s_bins,
        ) = process_model(
            model_name,
            config,
        )

        comparisons.append(
            comparison
        )

        primary_summaries.append(
            primary_summary
        )

        sensitivity_summaries.append(
            sensitivity_summary
        )

        primary_bins.append(
            p_bins
        )

        sensitivity_bins.append(
            s_bins
        )

    results_root = Path("results")

    pd.concat(
        comparisons,
        ignore_index=True,
    ).to_csv(
        results_root
        / "deep_stop_calibration_scope_comparison.csv",
        index=False,
    )

    pd.concat(
        primary_summaries,
        ignore_index=True,
    ).to_csv(
        results_root
        / "deep_stop_calibrated_primary_summary.csv",
        index=False,
    )

    pd.concat(
        sensitivity_summaries,
        ignore_index=True,
    ).to_csv(
        results_root
        / "deep_stop_calibrated_within40m_summary.csv",
        index=False,
    )

    pd.concat(
        primary_bins,
        ignore_index=True,
    ).to_csv(
        results_root
        / "deep_stop_primary_distance_bins.csv",
        index=False,
    )

    pd.concat(
        sensitivity_bins,
        ignore_index=True,
    ).to_csv(
        results_root
        / "deep_stop_within40m_distance_bins.csv",
        index=False,
    )

    print()
    print("=" * 84)
    print("DONE")
    print("=" * 84)

    print(
        "Primary threshold = max-F1 over all Clear evaluable frames."
    )

    print(
        "Within-40m threshold = sensitivity analysis only."
    )

    print(
        "Raw-confidence means use the ORIGINAL saved definition, "
        "including 0.0 on frames with no raw STOP candidate."
    )

    print(
        "Final study still requires separate calibration and held-out trials."
    )


if __name__ == "__main__":
    main()
