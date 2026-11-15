#!/usr/bin/env bash
set -euo pipefail

# Final Fog-only acquisition sweep:
#   5 Clear trials
#   6 Fog severities x 5 trials = 30
#   Total = 35 recordings
#
# Assumes:
#   - CARLA server is already running
#   - you are launching this from ~/same-camera-different-failures
#   - conda environment "carla" is active
#   - frozen route already exists
#
# DO NOT add --reset-route during the final sweep.

DRIFT="0.75"
FOG_LEVELS=(15 30 45 60 75 90)
TRIALS=(1 2 3 4 5)

mkdir -p logs/final_fog_sweep

echo "============================================================"
echo "FINAL FOG SWEEP"
echo "Clear: 5 trials"
echo "Fog:   6 severities x 5 trials = 30"
echo "Total: 35 recordings"
echo "Lateral drift amplitude: ${DRIFT} m"
echo "============================================================"

# -----------------------------
# Clear baseline: 5 trials
# -----------------------------
for t in "${TRIALS[@]}"; do
    echo
    echo ">>> CLEAR | trial ${t}/5"
    python carla/collect_data.py \
        --weather clear \
        --severity 0 \
        --lateral-drift "${DRIFT}" \
        --trial "${t}" \
        2>&1 | tee "logs/final_fog_sweep/clear_trial_${t}.log"
done

# -----------------------------
# Fog: 6 severities x 5 trials
# -----------------------------
for s in "${FOG_LEVELS[@]}"; do
    for t in "${TRIALS[@]}"; do
        echo
        echo ">>> FOG ${s} | trial ${t}/5"
        python carla/collect_data.py \
            --weather fog \
            --severity "${s}" \
            --lateral-drift "${DRIFT}" \
            --trial "${t}" \
            2>&1 | tee "logs/final_fog_sweep/fog_${s}_trial_${t}.log"
    done
done

echo
echo "============================================================"
echo "SWEEP COMPLETE"
echo "============================================================"

echo
echo "Expected trial recording directories:"
find data/recordings -maxdepth 1 -type d \
    \( -name 'stop3_clear_0_drift_0p75_trial_*' \
    -o -name 'stop3_fog_*_drift_0p75_trial_*' \) \
    | sort

echo
echo "Count:"
find data/recordings -maxdepth 1 -type d \
    \( -name 'stop3_clear_0_drift_0p75_trial_*' \
    -o -name 'stop3_fog_*_drift_0p75_trial_*' \) \
    | wc -l
