#!/usr/bin/env python3
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
REC = ROOT / "data" / "recordings"

TRIALS = [1,2,3,4,5]
SEVS = [15,30,45,60,75,90]
EXPECTED_PHASE = {1:-8.0, 2:-4.0, 3:0.0, 4:4.0, 5:8.0}
EXPECTED_ROWS = 181

expected = []
for t in TRIALS:
    expected.append(("clear",0,t, REC / f"stop3_clear_0_drift_0p75_trial_{t}"))
for s in SEVS:
    for t in TRIALS:
        expected.append(("fog",s,t, REC / f"stop3_fog_{s}_drift_0p75_trial_{t}"))

problems = []
rows_out = []

core_required = [
    "sample_id","weather","severity","route_s_m",
    "ego_x","ego_y","ego_z",
    "ego_pitch","ego_yaw","ego_roll",
    "distance_to_sign_m","is_junction","lane_gt_x",
]

for weather, sev, trial, d in expected:
    if not d.is_dir():
        problems.append(f"MISSING DIR: {d}")
        continue

    csv = d / "frames.csv"
    frames_dir = d / "frames"

    if not csv.exists():
        problems.append(f"MISSING CSV: {csv}")
        continue
    if not frames_dir.is_dir():
        problems.append(f"MISSING FRAMES DIR: {frames_dir}")
        continue

    try:
        df = pd.read_csv(csv)
    except Exception as e:
        problems.append(f"READ ERROR {csv}: {e}")
        continue

    missing_cols = [c for c in core_required if c not in df.columns]
    if missing_cols:
        problems.append(f"{d.name}: missing columns {missing_cols}")
        continue

    n_csv = len(df)
    n_img = len(list(frames_dir.glob("*.png")))

    if n_csv != EXPECTED_ROWS:
        problems.append(f"{d.name}: CSV rows={n_csv}, expected {EXPECTED_ROWS}")
    if n_img != EXPECTED_ROWS:
        problems.append(f"{d.name}: PNG count={n_img}, expected {EXPECTED_ROWS}")

    sid = pd.to_numeric(df["sample_id"], errors="coerce")
    if sid.isna().any() or list(sid.astype(int)) != list(range(1, EXPECTED_ROWS+1)):
        problems.append(f"{d.name}: sample_id is not exactly 1..181")

    route = pd.to_numeric(df["route_s_m"], errors="coerce")
    if route.isna().any():
        problems.append(f"{d.name}: route_s_m contains NaN")
    elif not np.all(np.diff(route.to_numpy()) > 0):
        problems.append(f"{d.name}: route_s_m is not strictly increasing")

    wvals = set(df["weather"].astype(str))
    if wvals != {weather}:
        problems.append(f"{d.name}: weather values {wvals}, expected {weather}")

    svals = set(pd.to_numeric(df["severity"], errors="coerce").dropna().astype(int))
    if svals != {sev}:
        problems.append(f"{d.name}: severity values {svals}, expected {sev}")

    # Numeric fields that should never be missing.
    for c in [
        "ego_x","ego_y","ego_z",
        "ego_pitch","ego_yaw","ego_roll",
        "distance_to_sign_m","is_junction",
    ]:
        if pd.to_numeric(df[c], errors="coerce").isna().any():
            problems.append(f"{d.name}: {c} contains NaN/non-numeric values")

    # Lane GT may be blank at junctions only.
    junc = pd.to_numeric(df["is_junction"], errors="coerce").fillna(1).astype(int)
    lane = pd.to_numeric(df["lane_gt_x"], errors="coerce")
    nonj_missing = int(((junc == 0) & lane.isna()).sum())
    if nonj_missing != 0:
        problems.append(f"{d.name}: non-junction missing lane_gt_x = {nonj_missing}")

    # Optional final-sweep fields: validate if present.
    if "trial_id" in df.columns:
        vals = set(pd.to_numeric(df["trial_id"], errors="coerce").dropna().astype(int))
        if vals != {trial}:
            problems.append(f"{d.name}: trial_id values {vals}, expected {trial}")

    if "lateral_phase_deg" in df.columns:
        vals = pd.to_numeric(df["lateral_phase_deg"], errors="coerce")
        if vals.isna().any() or not np.allclose(vals.to_numpy(), EXPECTED_PHASE[trial], atol=1e-9):
            problems.append(
                f"{d.name}: lateral_phase_deg does not equal {EXPECTED_PHASE[trial]}"
            )

    rows_out.append({
        "recording": d.name,
        "csv_rows": n_csv,
        "pngs": n_img,
        "nonjunction_missing_lane_gt": nonj_missing,
        "route_start_m": float(route.iloc[0]),
        "route_end_m": float(route.iloc[-1]),
    })

print("="*80)
print("FINAL FOG SWEEP VALIDATION")
print("="*80)
print(f"Expected recordings: {len(expected)}")
print(f"Found validated candidates: {len(rows_out)}")

if rows_out:
    out = pd.DataFrame(rows_out)
    print("\nSummary:")
    print(out.to_string(index=False))

if problems:
    print("\nFAIL")
    print("-"*80)
    for p in problems:
        print(" -", p)
    sys.exit(1)

print("\nPASS: all 35 final recordings passed structural validation.")
print("Safe to start detector inference / final analysis.")
