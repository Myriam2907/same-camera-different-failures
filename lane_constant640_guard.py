#!/usr/bin/env python3
"""
Lane geometry guard: compare the drift ground truth against a trivial
constant x=640 predictor for every current fog dataset.

IMPORTANT:
- x=640 here is a PREDICTION used only for the trivial baseline.
- A lane-detector non-detection uses ERROR = W/2 = 640 px.
  Those two uses of the number 640 are not the same operation.

Lane-evaluable frames:
- is_junction == 0
- finite lane_gt_x

No images or detector inference are needed.
"""

from pathlib import Path
import numpy as np
import pandas as pd

IMAGE_W = 1280
TRIVIAL_PRED_X = IMAGE_W / 2.0
NONDETECTION_ERROR = IMAGE_W / 2.0

DATASETS = [
    ("clear_drift", 0, "data/recordings/stop3_clear_0_drift_0p75"),
    ("fog15_drift", 15, "data/recordings/stop3_fog_15_drift_0p75"),
    ("fog30_drift", 30, "data/recordings/stop3_fog_30_drift_0p75"),
    ("fog45_drift", 45, "data/recordings/stop3_fog_45_drift_0p75"),
    ("fog60_drift", 60, "data/recordings/stop3_fog_60_drift_0p75"),
    ("fog75_drift", 75, "data/recordings/stop3_fog_75_drift_0p75"),
    ("fog90_drift", 90, "data/recordings/stop3_fog_90_drift_0p75"),
]

OUT = Path("results/lane_constant640_guard.csv")


def main():
    rows = []

    print("=" * 100)
    print("LANE GEOMETRY GUARD — TRIVIAL CONSTANT-x BASELINE")
    print("=" * 100)
    print(f"Trivial predictor: x_pred = {TRIVIAL_PRED_X:.1f} px")
    print(f"Non-detection penalty for real detectors: error = {NONDETECTION_ERROR:.1f} px")
    print()

    for name, severity, root_str in DATASETS:
        root = Path(root_str)
        csv_path = root / "frames.csv"

        if not csv_path.exists():
            raise RuntimeError(f"Missing {csv_path}")

        df = pd.read_csv(csv_path)

        needed = {"lane_gt_x", "is_junction"}
        missing = sorted(needed - set(df.columns))
        if missing:
            raise RuntimeError(f"{csv_path} missing columns: {missing}")

        usable = df[
            (df["is_junction"] == 0)
            & np.isfinite(df["lane_gt_x"])
        ].copy()

        if len(usable) == 0:
            raise RuntimeError(f"No usable lane frames in {csv_path}")

        gt = usable["lane_gt_x"].astype(float).to_numpy()
        err = np.abs(TRIVIAL_PRED_X - gt)

        rows.append({
            "dataset": name,
            "severity": severity,
            "usable_nonjunction_frames": len(usable),
            "gt_min_px": float(gt.min()),
            "gt_max_px": float(gt.max()),
            "gt_range_px": float(gt.max() - gt.min()),
            "gt_mean_px": float(gt.mean()),
            "gt_std_px": float(gt.std()),
            "constant640_mean_abs_error_px": float(err.mean()),
            "constant640_median_abs_error_px": float(np.median(err)),
            "constant640_max_abs_error_px": float(err.max()),
        })

    out = pd.DataFrame(rows).sort_values("severity")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(
        out.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    clear = out[out["severity"] == 0].iloc[0]
    clear_err = float(clear["constant640_mean_abs_error_px"])

    print()
    print("=" * 100)
    print(f"CLEAR TRIVIAL BASELINE: {clear_err:.6f} px")
    print("=" * 100)
    print(
        "Guard rule for lane detectors: a detector with mean penalized error "
        f">= {clear_err:.6f} px on Clear is worse than simply predicting x=640."
    )
    print(
        "Again: detector NON-DETECTION must be recorded as error=640 px, "
        "not as predicted_x=640."
    )
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
