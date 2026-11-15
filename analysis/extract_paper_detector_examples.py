#!/usr/bin/env python3

from pathlib import Path
import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "results" / "final" / "paper_detector_examples"
OUT.mkdir(parents=True, exist_ok=True)

STOP_DIRS = {
    "HSV": ROOT / "results" / "final" / "stop_hsv",
    "Template": ROOT / "results" / "final" / "stop_template",
    "YOLO26s": ROOT / "results" / "final" / "stop_yolo",
    "RF-DETR-L": ROOT / "results" / "final" / "stop_rfdetr",
}

TRIALS = [3, 1, 2, 4, 5]

TARGET_M = 12.0
MIN_M = 8.0
MAX_M = 16.0

GT_COLOR = (255, 0, 255)      # magenta
CORRECT_COLOR = (0, 200, 0)   # green
FALSE_COLOR = (0, 0, 255)     # red


def rec_name(trial):
    return f"stop3_clear_0_drift_0p75_trial_{trial}"


def prefix(trial):
    return f"clear_trial_{trial}"


def boolish(v):
    if pd.isna(v):
        return False
    if isinstance(v, str):
        return v.lower().strip() in {"1", "true", "yes"}
    return bool(v)


def require(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def get_meta(trial):
    p = require(
        ROOT / "data" / "recordings" /
        rec_name(trial) / "frames.csv"
    )
    return pd.read_csv(p)


def classical_correct(detector, trial):
    p = require(
        STOP_DIRS[detector] /
        f"{prefix(trial)}_per_frame.csv"
    )

    df = pd.read_csv(p)

    return set(
        df.loc[
            df["correct_detection"].map(boolish),
            "sample_id"
        ].astype(int)
    )


def deep_correct(detector, trial):
    p = require(
        STOP_DIRS[detector] /
        f"{prefix(trial)}_raw_boxes.csv"
    )

    df = pd.read_csv(p)

    if "passes_operating_threshold" in df.columns:
        df = df[
            df["passes_operating_threshold"].map(boolish)
        ]

    return set(
        df.loc[
            df["correct_by_iou"].map(boolish),
            "sample_id"
        ].astype(int)
    )


def choose_frame():
    candidates = []

    for trial in TRIALS:

        meta = get_meta(trial).copy()
        meta["sample_id"] = meta["sample_id"].astype(int)

        if "route_s_m" in meta.columns:
            meta = meta[meta["route_s_m"] <= 0]

        common = (
            classical_correct("HSV", trial)
            & classical_correct("Template", trial)
            & deep_correct("YOLO26s", trial)
            & deep_correct("RF-DETR-L", trial)
        )

        q = meta[
            meta["sample_id"].isin(common)
        ].copy()

        q = q[
            np.isfinite(q["distance_to_sign_m"])
            & (q["distance_to_sign_m"] > 0)
        ]

        if q.empty:
            continue

        q["trial"] = trial
        q["target_error"] = (
            q["distance_to_sign_m"] - TARGET_M
        ).abs()

        q["preferred"] = (
            (q["distance_to_sign_m"] >= MIN_M)
            & (q["distance_to_sign_m"] <= MAX_M)
        )

        q = q.sort_values(
            ["preferred", "target_error"],
            ascending=[False, True]
        )

        candidates.append(q.iloc[0])

    if not candidates:
        raise RuntimeError(
            "No common close frame found where all four "
            "STOP detectors are correct."
        )

    c = pd.DataFrame(candidates)

    c = c.sort_values(
        ["preferred", "target_error"],
        ascending=[False, True]
    )

    return c.iloc[0]


def load_frame(trial, sid):
    p = require(
        ROOT / "data" / "recordings" /
        rec_name(trial) / "frames" /
        f"{sid:04d}.png"
    )

    img = cv2.imread(str(p))

    if img is None:
        raise RuntimeError(f"Could not load {p}")

    return img, p


def rect(img, box, color, thick=4):
    x1, y1, x2, y2 = [
        int(round(float(x))) for x in box
    ]

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        thick,
        cv2.LINE_AA
    )


def classical_gt(row):
    cols = [
        "gt_bbox_x1", "gt_bbox_y1",
        "gt_bbox_x2", "gt_bbox_y2"
    ]

    if all(c in row.index for c in cols):
        vals = [row[c] for c in cols]

        if all(pd.notna(v) for v in vals):
            return tuple(vals)

    return None


def deep_gt(row):
    cols = ["gt_x1", "gt_y1", "gt_x2", "gt_y2"]

    if all(c in row.index for c in cols):
        vals = [row[c] for c in cols]

        if all(pd.notna(v) for v in vals):
            return tuple(vals)

    return None


def draw_classical(detector, trial, sid, base):

    p = require(
        STOP_DIRS[detector] /
        f"{prefix(trial)}_per_frame.csv"
    )

    df = pd.read_csv(p)

    q = df[
        df["sample_id"].astype(int) == sid
    ]

    if q.empty:
        raise RuntimeError(
            f"{detector}: sample {sid} missing"
        )

    row = q.iloc[0]

    out = base.copy()

    gt = classical_gt(row)

    if gt is not None:
        rect(out, gt, GT_COLOR)

    pred = None

    if all(
        c in row.index
        for c in ["bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    ):
        vals = [
            row["bbox_x"],
            row["bbox_y"],
            row["bbox_w"],
            row["bbox_h"]
        ]

        if all(pd.notna(v) for v in vals):
            x, y, w, h = [float(v) for v in vals]

            pred = (
                x,
                y,
                x + w,
                y + h
            )

    if pred is None:
        for cols in [
            ["x1", "y1", "x2", "y2"],
            ["pred_x1", "pred_y1", "pred_x2", "pred_y2"],
            ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]
        ]:
            if all(c in row.index for c in cols):
                vals = [row[c] for c in cols]

                if all(pd.notna(v) for v in vals):
                    pred = tuple(vals)
                    break

    if pred is not None:

        correct = boolish(
            row.get("correct_detection", False)
        )

        rect(
            out,
            pred,
            CORRECT_COLOR if correct else FALSE_COLOR
        )

    return out, gt


def draw_deep(detector, trial, sid, base):

    frames_path = require(
        STOP_DIRS[detector] /
        f"{prefix(trial)}_frames.csv"
    )

    boxes_path = require(
        STOP_DIRS[detector] /
        f"{prefix(trial)}_raw_boxes.csv"
    )

    frames = pd.read_csv(frames_path)

    q = frames[
        frames["sample_id"].astype(int) == sid
    ]

    if q.empty:
        raise RuntimeError(
            f"{detector}: frame {sid} missing"
        )

    row = q.iloc[0]

    out = base.copy()

    gt = deep_gt(row)

    if gt is not None:
        rect(out, gt, GT_COLOR)

    boxes = pd.read_csv(boxes_path)

    boxes = boxes[
        boxes["sample_id"].astype(int) == sid
    ].copy()

    if "passes_operating_threshold" in boxes.columns:
        boxes = boxes[
            boxes["passes_operating_threshold"].map(boolish)
        ]

    for _, b in boxes.iterrows():

        vals = [
            b["x1"], b["y1"],
            b["x2"], b["y2"]
        ]

        if not all(pd.notna(v) for v in vals):
            continue

        correct = boolish(
            b.get("correct_by_iou", False)
        )

        rect(
            out,
            vals,
            CORRECT_COLOR if correct else FALSE_COLOR,
            3
        )

    return out, gt


def add_zoom(img, gt, title, dist):

    out = img.copy()

    h, w = out.shape[:2]

    if gt is None:
        return out

    x1, y1, x2, y2 = map(float, gt)

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    side = int(
        max(
            240,
            5 * max(bw, bh)
        )
    )

    xa = max(0, cx - side // 2)
    ya = max(0, cy - side // 2)

    xb = min(w, xa + side)
    yb = min(h, ya + side)

    xa = max(0, xb - side)
    ya = max(0, yb - side)

    crop = out[ya:yb, xa:xb]

    inset_h = int(h * 0.50)

    inset_w = int(
        inset_h *
        crop.shape[1] /
        crop.shape[0]
    )

    inset_w = min(
        inset_w,
        int(w * 0.42)
    )

    crop = cv2.resize(
        crop,
        (inset_w, inset_h),
        interpolation=cv2.INTER_CUBIC
    )

    px = w - inset_w - 20
    py = 60

    cv2.rectangle(
        out,
        (px - 7, py - 7),
        (px + inset_w + 7,
         py + inset_h + 7),
        (255, 255, 255),
        -1
    )

    cv2.rectangle(
        out,
        (px - 7, py - 7),
        (px + inset_w + 7,
         py + inset_h + 7),
        (0, 0, 0),
        3
    )

    out[
        py:py + inset_h,
        px:px + inset_w
    ] = crop

    cv2.rectangle(
        out,
        (0, 0),
        (w, 48),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        out,
        f"{title} | Clear | {dist:.1f} m",
        (15, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return out


def find_lane_image(trial, sid):

    candidates = []

    for root in [
        ROOT / "results",
        ROOT / "outputs"
    ]:

        if not root.exists():
            continue

        for p in root.rglob("*.png"):

            low = str(p).lower()

            if (
                "lane" not in low
                and "ufld" not in low
            ):
                continue

            if (
                f"{sid:04d}" not in p.stem
                and str(sid) not in p.stem
            ):
                continue

            score = 0

            if (
                f"trial_{trial}" in low
                or f"trial{trial}" in low
            ):
                score += 5

            if "clear" in low:
                score += 3

            if (
                "overlay" in low
                or "vis" in low
                or "debug" in low
            ):
                score += 3

            candidates.append(
                (score, p.stat().st_size, p)
            )

    if not candidates:
        return None

    candidates.sort(reverse=True)

    return candidates[0][2]


def save(path, img):

    cv2.imwrite(
        str(path),
        img,
        [cv2.IMWRITE_PNG_COMPRESSION, 2]
    )

    print("Saved:", path)


def main():

    print("=" * 80)
    print("SELECTING CLOSE CLEAR FRAME")
    print("=" * 80)

    sel = choose_frame()

    trial = int(sel["trial"])
    sid = int(sel["sample_id"])
    dist = float(sel["distance_to_sign_m"])

    print("trial       :", trial)
    print("sample_id   :", sid)
    print("STOP range  :", f"{dist:.3f} m")

    if "route_s_m" in sel.index:
        print(
            "route_s_m   :",
            f"{float(sel['route_s_m']):.3f}"
        )

    base, raw_path = load_frame(
        trial,
        sid
    )

    print("raw frame   :", raw_path)
    print()

    # HSV
    img, gt = draw_classical(
        "HSV",
        trial,
        sid,
        base
    )

    img = add_zoom(
        img,
        gt,
        "HSV",
        dist
    )

    save(
        OUT / "02_stop_hsv.png",
        img
    )

    # Template
    img, gt = draw_classical(
        "Template",
        trial,
        sid,
        base
    )

    img = add_zoom(
        img,
        gt,
        "Template",
        dist
    )

    save(
        OUT / "03_stop_template.png",
        img
    )

    # YOLO
    img, gt = draw_deep(
        "YOLO26s",
        trial,
        sid,
        base
    )

    img = add_zoom(
        img,
        gt,
        "YOLO26s",
        dist
    )

    save(
        OUT / "04_stop_yolo26s.png",
        img
    )

    # RF-DETR
    img, gt = draw_deep(
        "RF-DETR-L",
        trial,
        sid,
        base
    )

    img = add_zoom(
        img,
        gt,
        "RF-DETR-L",
        dist
    )

    save(
        OUT / "05_stop_rfdetr_l.png",
        img
    )

    # Lane
    lane_source = find_lane_image(
        trial,
        sid
    )

    if lane_source is not None:

        print(
            "Found lane visualization:",
            lane_source
        )

        lane = cv2.imread(
            str(lane_source)
        )

        if lane is None:
            lane = base.copy()

    else:

        print(
            "No saved UFLD overlay found for "
            "this exact sample."
        )

        print(
            "Using the same clear raw frame as "
            "the lane qualitative placeholder."
        )

        lane = base.copy()

    h, w = lane.shape[:2]

    cv2.rectangle(
        lane,
        (0, 0),
        (w, 48),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        lane,
        f"UFLD-v2 lane example | Clear | sample {sid}",
        (15, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    save(
        OUT / "01_lane_ufld.png",
        lane
    )

    # Save selection
    info = {
        "weather": "clear",
        "severity": 0,
        "trial": trial,
        "sample_id": sid,
        "distance_to_sign_m": dist,
        "raw_frame": str(
            raw_path.relative_to(ROOT)
        ),
        "selection_rule":
            "all four STOP detectors correct; "
            "prefer 8-16 m; closest to 12 m"
    }

    if "route_s_m" in sel.index:
        info["route_s_m"] = float(
            sel["route_s_m"]
        )

    pd.DataFrame(
        [info]
    ).to_csv(
        OUT / "selection.csv",
        index=False
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)

    print("Output folder:")
    print(OUT)

    print()

    for p in sorted(
        OUT.glob("*.png")
    ):
        print(" ", p.name)


if __name__ == "__main__":
    main()
