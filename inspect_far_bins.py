#!/usr/bin/env python3
"""
Inspect the suspicious 50-60 m vs 60 m+ Clear bins for YOLO26s and RF-DETR-L.

Uses:
- Clear recording metadata
- already-saved raw detector candidates
- the within-40m calibrated threshold from calibrate_deep_stop_v2.py

Checks:
1) whether distance_to_sign_m decreases as sample_id increases,
2) whether any 60m+ frame is actually past the sign / outside sign_in_view,
3) GT box width/height/area in each far-distance frame,
4) raw confidence, accepted detection, IoU and correctness for each model.

No neural inference is rerun.
"""

from pathlib import Path
import numpy as np
import pandas as pd

CLEAR_ROOT = Path("data/recordings/stop3_clear_0_drift_0p75")
META_PATH = CLEAR_ROOT / "frames.csv"

MODELS = {
    "yolo26s": {
        "raw_root": Path("results/sign_yolo_iou"),
        "calib_path": Path(
            "results/sign_yolo_iou_calibrated_v2/"
            "calibration_scope_comparison.csv"
        ),
    },
    "rfdetr_l": {
        "raw_root": Path("results/sign_rfdetr_iou"),
        "calib_path": Path(
            "results/sign_rfdetr_iou_calibrated_v2/"
            "calibration_scope_comparison.csv"
        ),
    },
}

IOU_THRESHOLD = 0.50
OUT_PATH = Path("results/far_bin_diagnostic.csv")


def load_within40_threshold(calib_path):
    df = pd.read_csv(calib_path)

    row = df[
        df["calibration_scope"] == "within40m"
    ]

    if len(row) != 1:
        raise RuntimeError(
            f"Could not find unique within40m threshold in {calib_path}"
        )

    return float(row.iloc[0]["threshold"])


def classify_frame(boxes, threshold):
    if boxes is None or len(boxes) == 0:
        return {
            "max_raw_conf": 0.0,
            "accepted_count": 0,
            "best_accepted_iou": 0.0,
            "correct": 0,
        }

    max_raw_conf = float(
        boxes["confidence"].max()
    )

    accepted = boxes[
        boxes["confidence"] >= threshold
    ]

    if len(accepted) == 0:
        return {
            "max_raw_conf": max_raw_conf,
            "accepted_count": 0,
            "best_accepted_iou": 0.0,
            "correct": 0,
        }

    best_iou = float(
        accepted["iou_with_gt"].max()
    )

    correct = int(
        (
            accepted["iou_with_gt"]
            >= IOU_THRESHOLD
        ).any()
    )

    return {
        "max_raw_conf": max_raw_conf,
        "accepted_count": len(accepted),
        "best_accepted_iou": best_iou,
        "correct": correct,
    }


def main():
    meta = pd.read_csv(META_PATH)

    required = {
        "sample_id",
        "route_s_m",
        "distance_to_sign_m",
        "sign_in_view",
        "sign_bbox_x1",
        "sign_bbox_y1",
        "sign_bbox_x2",
        "sign_bbox_y2",
    }

    missing = sorted(required - set(meta.columns))

    if missing:
        raise RuntimeError(
            f"Missing metadata columns: {missing}"
        )

    # Same approach-frame rule used by the detector scripts.
    approach = meta[
        (meta["route_s_m"] <= 0.0)
        & (meta["sign_in_view"] == 1)
    ].copy()

    approach = approach.sort_values("sample_id")

    approach["gt_w_px"] = (
        approach["sign_bbox_x2"]
        - approach["sign_bbox_x1"]
    )

    approach["gt_h_px"] = (
        approach["sign_bbox_y2"]
        - approach["sign_bbox_y1"]
    )

    approach["gt_area_px2"] = (
        approach["gt_w_px"]
        * approach["gt_h_px"]
    )

    # Distance should generally fall while approaching.
    distances = approach["distance_to_sign_m"].to_numpy()

    increases = np.where(
        np.diff(distances) > 1e-6
    )[0]

    print("=" * 88)
    print("CLEAR DISTANCE / FAR-BIN DIAGNOSTIC")
    print("=" * 88)
    print(f"Evaluable approach frames: {len(approach)}")
    print(
        f"distance_to_sign_m min/max: "
        f"{approach['distance_to_sign_m'].min():.4f} / "
        f"{approach['distance_to_sign_m'].max():.4f}"
    )
    print(
        f"Frames with route_s_m > 0 inside evaluation set: "
        f"{int((approach['route_s_m'] > 0).sum())}"
    )
    print(
        f"Frames with sign_in_view != 1 inside evaluation set: "
        f"{int((approach['sign_in_view'] != 1).sum())}"
    )
    print(
        f"Number of distance increases as sample_id grows: "
        f"{len(increases)}"
    )

    if len(increases) > 0:
        print("Distance increases at:")
        for idx in increases[:20]:
            a = approach.iloc[idx]
            b = approach.iloc[idx + 1]
            print(
                f"  sample {int(a['sample_id'])} "
                f"{a['distance_to_sign_m']:.4f} -> "
                f"sample {int(b['sample_id'])} "
                f"{b['distance_to_sign_m']:.4f}"
            )

    all_rows = []

    for model_name, cfg in MODELS.items():
        threshold = load_within40_threshold(
            cfg["calib_path"]
        )

        boxes = pd.read_csv(
            cfg["raw_root"]
            / "clear_drift_raw_boxes.csv"
        )

        grouped = {
            int(sid): group
            for sid, group in boxes.groupby("sample_id")
        }

        print()
        print("=" * 88)
        print(
            f"{model_name} | within40m threshold = {threshold:.4f}"
        )
        print("=" * 88)

        for bin_name, lo, hi in [
            ("50-60m", 50.0, 60.0),
            ("60m+", 60.0, np.inf),
        ]:
            subset = approach[
                (approach["distance_to_sign_m"] >= lo)
                & (approach["distance_to_sign_m"] < hi)
            ].copy()

            diagnostic_rows = []

            for _, row in subset.iterrows():
                sid = int(row["sample_id"])

                det = classify_frame(
                    grouped.get(sid),
                    threshold,
                )

                out = {
                    "model": model_name,
                    "distance_bin": bin_name,
                    "threshold": threshold,
                    "sample_id": sid,
                    "route_s_m": float(row["route_s_m"]),
                    "distance_to_sign_m": float(
                        row["distance_to_sign_m"]
                    ),
                    "sign_in_view": int(
                        row["sign_in_view"]
                    ),
                    "gt_x1": float(row["sign_bbox_x1"]),
                    "gt_y1": float(row["sign_bbox_y1"]),
                    "gt_x2": float(row["sign_bbox_x2"]),
                    "gt_y2": float(row["sign_bbox_y2"]),
                    "gt_w_px": float(row["gt_w_px"]),
                    "gt_h_px": float(row["gt_h_px"]),
                    "gt_area_px2": float(row["gt_area_px2"]),
                    **det,
                }

                diagnostic_rows.append(out)
                all_rows.append(out)

            ddf = pd.DataFrame(diagnostic_rows)

            print()
            print(f"{bin_name}: {len(ddf)} frames")

            if len(ddf) == 0:
                continue

            print(
                f"correct: {int(ddf['correct'].sum())}/{len(ddf)} "
                f"= {100*ddf['correct'].mean():.2f}%"
            )

            print(
                f"GT width mean/min/max: "
                f"{ddf['gt_w_px'].mean():.2f} / "
                f"{ddf['gt_w_px'].min():.2f} / "
                f"{ddf['gt_w_px'].max():.2f} px"
            )

            print(
                f"GT area mean/min/max: "
                f"{ddf['gt_area_px2'].mean():.2f} / "
                f"{ddf['gt_area_px2'].min():.2f} / "
                f"{ddf['gt_area_px2'].max():.2f} px^2"
            )

            print()
            print(
                ddf[
                    [
                        "sample_id",
                        "route_s_m",
                        "distance_to_sign_m",
                        "gt_w_px",
                        "gt_h_px",
                        "gt_area_px2",
                        "max_raw_conf",
                        "accepted_count",
                        "best_accepted_iou",
                        "correct",
                    ]
                ].to_string(
                    index=False,
                    float_format=lambda x: f"{x:.4f}",
                )
            )

    out_df = pd.DataFrame(all_rows)
    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_df.to_csv(
        OUT_PATH,
        index=False,
    )

    print()
    print("=" * 88)
    print(f"Saved diagnostic CSV: {OUT_PATH}")
    print("=" * 88)


if __name__ == "__main__":
    main()
