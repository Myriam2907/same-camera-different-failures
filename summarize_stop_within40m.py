#!/usr/bin/env python3
"""
Summarize the PRIMARY <=40 m STOP-sign operating region for the calibrated
YOLO26s and RF-DETR-L pilot runs.

Uses the already-created calibrated frame CSVs from:
  results/sign_yolo_iou_calibrated_v2/within40m/
  results/sign_rfdetr_iou_calibrated_v2/within40m/

No detector inference is rerun.
"""

from pathlib import Path
import numpy as np
import pandas as pd

MAX_DISTANCE_M = 40.0

MODELS = {
    "yolo26s": {
        "root": Path("results/sign_yolo_iou_calibrated_v2/within40m"),
        "threshold": 0.041,
    },
    "rfdetr_l": {
        "root": Path("results/sign_rfdetr_iou_calibrated_v2/within40m"),
        "threshold": 0.043,
    },
}

DATASETS = [
    ("clear_drift", 0),
    ("fog15_drift", 15),
    ("fog30_drift", 30),
    ("fog45_drift", 45),
    ("fog60_drift", 60),
    ("fog75_drift", 75),
    ("fog90_drift", 90),
]


def summarize(df, model, threshold, dataset, severity):
    d = df[
        df["distance_to_sign_m"] <= MAX_DISTANCE_M
    ].copy()

    n = len(d)
    detected = int(d["detected"].sum())
    correct = int(d["correct"].sum())
    fp_frames = int(d["false_positive_frame"].sum())
    fp_boxes = int(d["false_positive_boxes"].sum())

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
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    correct_df = d[d["correct"] == 1].sort_values("sample_id")

    if len(correct_df) > 0:
        first = correct_df.iloc[0]
        first_correct_distance = float(first["distance_to_sign_m"])
    else:
        first_correct_distance = 0.0

    return {
        "model": model,
        "severity": severity,
        "threshold": threshold,
        "frames_le_40m": n,
        "detected_frames": detected,
        "correct_frames": correct,
        "correct_rate_pct": 100.0 * recall,
        "false_positive_frames": fp_frames,
        "false_positive_boxes": fp_boxes,
        "precision_pct": 100.0 * precision,
        "f1": f1,
        "mean_raw_confidence": float(
            d["max_raw_stop_confidence"].mean()
        ),
        "first_correct_distance_within40m": first_correct_distance,
    }


def main():
    rows = []

    for model, cfg in MODELS.items():
        for dataset, severity in DATASETS:
            path = cfg["root"] / f"{dataset}_frames.csv"

            if not path.exists():
                raise RuntimeError(f"Missing {path}")

            df = pd.read_csv(path)

            rows.append(
                summarize(
                    df,
                    model,
                    cfg["threshold"],
                    dataset,
                    severity,
                )
            )

    out = pd.DataFrame(rows)

    out_path = Path(
        "results/deep_stop_primary_within40m_summary.csv"
    )

    out.to_csv(out_path, index=False)

    print("=" * 108)
    print("PRIMARY STOP-SIGN OPERATING REGION: distance_to_sign_m <= 40 m")
    print("=" * 108)

    for model in MODELS:
        print()
        print(model)
        print("-" * 108)

        m = out[out["model"] == model]

        print(
            m[
                [
                    "severity",
                    "frames_le_40m",
                    "correct_rate_pct",
                    "precision_pct",
                    "false_positive_frames",
                    "false_positive_boxes",
                    "mean_raw_confidence",
                ]
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
