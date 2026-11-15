#!/usr/bin/env python3

from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RECORDINGS = ROOT / "data" / "recordings"

OUT = ROOT / "results" / "final" / "fourier_monitor"

SEVERITIES = [0, 15, 30, 45, 60, 75, 90]
TRIALS = [1, 2, 3, 4, 5]

DRIFT_NAME = "drift_0p75"


# ============================================================
# FIND RECORDINGS
# ============================================================

def get_recording_dir(severity, trial):
    """
    Use the five drift recordings already collected.

    Expected:
      stop3_clear_0_drift_0p75_trial_1
      stop3_fog_15_drift_0p75_trial_1
      ...
    """

    weather = "clear" if severity == 0 else "fog"

    exact = (
        RECORDINGS
        / f"stop3_{weather}_{severity}_{DRIFT_NAME}_trial_{trial}"
    )

    if exact.is_dir():
        return exact

    # Conservative fallback:
    # only accept it if exactly ONE directory matches.
    candidates = sorted(
        p for p in RECORDINGS.glob(
            f"stop3_{weather}_{severity}*trial_{trial}"
        )
        if p.is_dir()
    )

    if len(candidates) == 1:
        return candidates[0]

    print("\nCould not uniquely identify recording.")
    print(f"severity={severity}, trial={trial}")
    print("Expected:")
    print(exact)

    print("\nCandidates:")
    for p in candidates:
        print(" ", p)

    raise RuntimeError(
        "Recording directory missing or ambiguous. "
        "Do not guess the recording."
    )


# ============================================================
# FOURIER PROFILE
# ============================================================

def radial_fourier_profile(image):
    """
    Frozen Fourier definition:

      BGR
        -> grayscale
        -> FFT2
        -> fftshift
        -> |F|^2
        -> log(1 + power)
        -> keep largest complete frequency circle
        -> normalize valid 2-D log-power to unit total
        -> radial average

    IMPORTANT:
    No second normalization after radial averaging.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float64)

    # 2-D Fourier transform
    spectrum = np.fft.fft2(gray)

    # Put DC / low frequencies in center
    spectrum = np.fft.fftshift(spectrum)

    # Power
    power = np.abs(spectrum) ** 2

    # Compress huge dynamic range
    log_power = np.log1p(power)

    h, w = gray.shape

    cy = h // 2
    cx = w // 2

    y, x = np.indices((h, w))

    radius = np.sqrt(
        (x - cx) ** 2
        + (y - cy) ** 2
    ).astype(np.int32)

    # Largest complete circle inside image
    max_radius = min(cx, cy)

    valid = radius < max_radius

    radius_valid = radius[valid].ravel()

    power_valid = log_power[valid].astype(
        np.float64
    )

    # --------------------------------------------------------
    # FROZEN ORDER:
    # normalize BEFORE radial averaging
    # --------------------------------------------------------

    total = power_valid.sum()

    if not np.isfinite(total) or total <= 0:
        raise RuntimeError(
            "Invalid Fourier power total."
        )

    power_valid = power_valid / total

    power_valid = power_valid.ravel()

    # Radial average
    radial_sum = np.bincount(
        radius_valid,
        weights=power_valid,
        minlength=max_radius,
    )

    radial_count = np.bincount(
        radius_valid,
        minlength=max_radius,
    )

    radial_profile = (
        radial_sum
        / np.maximum(radial_count, 1)
    )

    radial_profile = radial_profile[
        :max_radius
    ]

    if not np.all(
        np.isfinite(radial_profile)
    ):
        raise RuntimeError(
            "Non-finite radial Fourier profile."
        )

    return radial_profile


# ============================================================
# DISTANCE
# ============================================================

def spectral_distance_l1(profile, reference):
    """
    Frozen metric: L1 distance.
    """

    if profile.shape != reference.shape:
        raise RuntimeError(
            "Spectrum shape mismatch: "
            f"{profile.shape} vs {reference.shape}"
        )

    return float(
        np.sum(
            np.abs(
                profile - reference
            )
        )
    )


# ============================================================
# LOAD ONE RECORDING
# ============================================================

def load_recording(
    recording_dir,
    severity,
    trial,
):
    metadata_path = (
        recording_dir / "frames.csv"
    )

    frame_dir = (
        recording_dir / "frames"
    )

    if not metadata_path.is_file():
        raise RuntimeError(
            f"Missing metadata: {metadata_path}"
        )

    if not frame_dir.is_dir():
        raise RuntimeError(
            f"Missing frame directory: {frame_dir}"
        )

    metadata = pd.read_csv(
        metadata_path
    )

    if "sample_id" not in metadata.columns:
        raise RuntimeError(
            f"No sample_id in {metadata_path}"
        )

    if metadata["sample_id"].duplicated().any():
        raise RuntimeError(
            f"Duplicate sample IDs in {metadata_path}"
        )

    profiles = {}
    rows = {}

    print(
        f"\nLoading severity={severity:2d}, "
        f"trial={trial}:"
    )
    print(
        f"  {recording_dir.name}"
    )

    n = len(metadata)

    for i, row in metadata.iterrows():

        sample_id = int(
            row["sample_id"]
        )

        image_path = (
            frame_dir
            / f"{sample_id:04d}.png"
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Could not read: {image_path}"
            )

        profile = radial_fourier_profile(
            image
        )

        profiles[sample_id] = profile

        rows[sample_id] = row.to_dict()

        if (
            i == 0
            or (i + 1) % 25 == 0
            or i + 1 == n
        ):
            print(
                f"    {i + 1}/{n}"
            )

    return {
        "dir": recording_dir,
        "metadata": metadata,
        "profiles": profiles,
        "rows": rows,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 1. Resolve all 35 recordings
    # --------------------------------------------------------

    recording_dirs = {}

    selected_rows = []

    print("=" * 90)
    print("FOURIER MONITOR — FROZEN ANALYSIS")
    print("=" * 90)

    for severity in SEVERITIES:
        for trial in TRIALS:

            d = get_recording_dir(
                severity,
                trial,
            )

            recording_dirs[
                (severity, trial)
            ] = d

            selected_rows.append({
                "severity": severity,
                "trial": trial,
                "recording_dir": str(d),
            })

    selected_df = pd.DataFrame(
        selected_rows
    )

    selected_df.to_csv(
        OUT / "selected_recordings.csv",
        index=False,
    )

    print("\nSELECTED RECORDINGS")
    print(selected_df.to_string(index=False))

    # --------------------------------------------------------
    # 2. Compute Fourier profiles for all recordings
    # --------------------------------------------------------

    data = {}

    for severity in SEVERITIES:
        for trial in TRIALS:

            data[
                (severity, trial)
            ] = load_recording(
                recording_dirs[
                    (severity, trial)
                ],
                severity,
                trial,
            )

    # --------------------------------------------------------
    # 3. Frozen common-position intersection
    #
    # Exact same sample positions must exist in:
    #   7 severities x 5 trials
    # --------------------------------------------------------

    sample_sets = []

    for severity in SEVERITIES:
        for trial in TRIALS:

            ids = set(
                data[
                    (severity, trial)
                ]["profiles"].keys()
            )

            sample_sets.append(ids)

    common_ids = sorted(
        set.intersection(
            *sample_sets
        )
    )

    if len(common_ids) == 0:
        raise RuntimeError(
            "No common sample IDs across all "
            "severities and trials."
        )

    pd.DataFrame({
        "sample_id": common_ids
    }).to_csv(
        OUT / "common_sample_ids.csv",
        index=False,
    )

    print("\n" + "=" * 90)
    print("COMMON MATCHED ROUTE POSITIONS")
    print("=" * 90)

    print(
        f"Common positions: {len(common_ids)}"
    )

    print(
        f"sample_id range: "
        f"{common_ids[0]} -> "
        f"{common_ids[-1]}"
    )

    # --------------------------------------------------------
    # 4. Verify spectrum lengths
    # --------------------------------------------------------

    lengths = set()

    for severity in SEVERITIES:
        for trial in TRIALS:
            for sample_id in common_ids:

                lengths.add(
                    len(
                        data[
                            (severity, trial)
                        ]["profiles"][
                            sample_id
                        ]
                    )
                )

    if len(lengths) != 1:
        raise RuntimeError(
            "Fourier profile lengths differ: "
            f"{sorted(lengths)}"
        )

    print(
        f"Radial bins: {next(iter(lengths))}"
    )

    # --------------------------------------------------------
    # 5. Build POSITION-SPECIFIC leave-one-out
    #    CLEAR references
    #
    # For trial j and position p:
    #
    # ref(j,p) =
    #   mean clear profile at position p
    #   from the OTHER four trials
    #
    # Never include trial j in its own reference.
    # --------------------------------------------------------

    loo_reference = {}

    for trial in TRIALS:

        other_trials = [
            t for t in TRIALS
            if t != trial
        ]

        assert trial not in other_trials
        assert len(other_trials) == 4

        for sample_id in common_ids:

            stack = np.stack([
                data[
                    (0, other_trial)
                ]["profiles"][
                    sample_id
                ]
                for other_trial
                in other_trials
            ])

            reference = np.mean(
                stack,
                axis=0,
            )

            loo_reference[
                (trial, sample_id)
            ] = reference

    print("\nLeave-one-out reference audit:")
    for trial in TRIALS:
        others = [
            t for t in TRIALS
            if t != trial
        ]
        print(
            f"  trial {trial} "
            f"<- clear trials {others}"
        )

    # --------------------------------------------------------
    # 6. Position-level spectral distances
    # --------------------------------------------------------

    position_rows = []

    for severity in SEVERITIES:
        for trial in TRIALS:

            for sample_id in common_ids:

                profile = data[
                    (severity, trial)
                ]["profiles"][
                    sample_id
                ]

                reference = loo_reference[
                    (trial, sample_id)
                ]

                distance = (
                    spectral_distance_l1(
                        profile,
                        reference,
                    )
                )

                meta = data[
                    (severity, trial)
                ]["rows"][
                    sample_id
                ]

                position_rows.append({
                    "severity":
                        severity,

                    "trial":
                        trial,

                    "sample_id":
                        sample_id,

                    "route_s_m":
                        meta.get(
                            "route_s_m",
                            np.nan,
                        ),

                    "distance_to_sign_m":
                        meta.get(
                            "distance_to_sign_m",
                            np.nan,
                        ),

                    "is_junction":
                        meta.get(
                            "is_junction",
                            np.nan,
                        ),

                    "spectral_distance_l1":
                        distance,
                })

    position_df = pd.DataFrame(
        position_rows
    )

    position_df.to_csv(
        OUT / "position_distances.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 7. AGGREGATE POSITION FIRST
    #
    # One D(severity, trial) per trial.
    # --------------------------------------------------------

    trial_df = (
        position_df
        .groupby(
            ["severity", "trial"],
            as_index=False,
        )
        .agg(
            spectral_distance=(
                "spectral_distance_l1",
                "mean",
            ),
            position_std=(
                "spectral_distance_l1",
                lambda x:
                    float(
                        np.std(
                            x,
                            ddof=1,
                        )
                    )
                    if len(x) > 1
                    else 0.0,
            ),
            positions_n=(
                "sample_id",
                "nunique",
            ),
        )
        .sort_values(
            ["severity", "trial"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 8. CLEAR threshold
    #
    # IMPORTANT:
    # mu0 and sigma0 are calculated across the FIVE
    # position-aggregated CLEAR trial values.
    #
    # NOT pooled clear frames.
    # --------------------------------------------------------

    clear_trial = (
        trial_df[
            trial_df["severity"] == 0
        ]
        .sort_values("trial")
        .copy()
    )

    if len(clear_trial) != 5:
        raise RuntimeError(
            "Expected exactly 5 clear trial "
            f"aggregates, got {len(clear_trial)}"
        )

    clear_values = (
        clear_trial[
            "spectral_distance"
        ]
        .to_numpy(
            dtype=float
        )
    )

    mu0 = float(
        np.mean(
            clear_values
        )
    )

    sigma0 = float(
        np.std(
            clear_values,
            ddof=1,
        )
    )

    threshold = float(
        mu0
        + 3.0 * sigma0
    )

    trial_df["threshold"] = threshold

    trial_df[
        "degradation_alert"
    ] = (
        trial_df[
            "spectral_distance"
        ]
        > threshold
    ).astype(int)

    trial_df.to_csv(
        OUT / "trial_distances.csv",
        index=False,
    )

    threshold_df = pd.DataFrame([{
        "clear_trials_n": 5,
        "positions_per_trial":
            len(common_ids),
        "mu0":
            mu0,
        "sigma0":
            sigma0,
        "threshold_mu0_plus_3sigma0":
            threshold,
    }])

    threshold_df.to_csv(
        OUT / "threshold.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 9. Severity summary across five trials
    # --------------------------------------------------------

    severity_df = (
        trial_df
        .groupby(
            "severity",
            as_index=False,
        )
        .agg(
            mean_spectral_distance=(
                "spectral_distance",
                "mean",
            ),
            std_spectral_distance=(
                "spectral_distance",
                lambda x:
                    float(
                        np.std(
                            x,
                            ddof=1,
                        )
                    ),
            ),
            min_spectral_distance=(
                "spectral_distance",
                "min",
            ),
            max_spectral_distance=(
                "spectral_distance",
                "max",
            ),
            alert_trials=(
                "degradation_alert",
                "sum",
            ),
            trials_n=(
                "trial",
                "count",
            ),
        )
        .sort_values(
            "severity"
        )
        .reset_index(drop=True)
    )

    severity_df[
        "alert_rate_pct"
    ] = (
        100.0
        * severity_df[
            "alert_trials"
        ]
        / severity_df[
            "trials_n"
        ]
    )

    severity_df[
        "threshold"
    ] = threshold

    severity_df.to_csv(
        OUT / "severity_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 10. Important paired Fog30 vs Fog45 comparison
    # --------------------------------------------------------

    d30 = (
        trial_df[
            trial_df["severity"] == 30
        ][
            ["trial", "spectral_distance"]
        ]
        .rename(
            columns={
                "spectral_distance":
                    "fog30_distance"
            }
        )
    )

    d45 = (
        trial_df[
            trial_df["severity"] == 45
        ][
            ["trial", "spectral_distance"]
        ]
        .rename(
            columns={
                "spectral_distance":
                    "fog45_distance"
            }
        )
    )

    paired = d30.merge(
        d45,
        on="trial",
        how="inner",
        validate="one_to_one",
    )

    paired[
        "fog45_minus_fog30"
    ] = (
        paired["fog45_distance"]
        - paired["fog30_distance"]
    )

    paired[
        "fog45_more_degraded"
    ] = (
        paired[
            "fog45_minus_fog30"
        ]
        > 0
    ).astype(int)

    paired.to_csv(
        OUT / "fog30_vs_fog45_paired.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 11. Main figure
    # --------------------------------------------------------

    x = severity_df[
        "severity"
    ].to_numpy()

    y = severity_df[
        "mean_spectral_distance"
    ].to_numpy()

    yerr = severity_df[
        "std_spectral_distance"
    ].to_numpy()

    plt.figure(
        figsize=(7.5, 5.0)
    )

    plt.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=4,
        linewidth=2,
    )

    plt.axhline(
        threshold,
        linestyle="--",
        label="Clear threshold μ₀ + 3σ₀",
    )

    plt.xlabel(
        "Fog severity"
    )

    plt.ylabel(
        "Position-aggregated L1 spectral distance"
    )

    plt.title(
        "Fourier Image Degradation vs Fog Severity"
    )

    plt.xticks(
        SEVERITIES
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUT / "fourier_distance_vs_fog.png",
        dpi=200,
    )

    plt.close()

    # --------------------------------------------------------
    # PRINT FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CLEAR THRESHOLD")
    print("=" * 90)

    print(
        clear_trial[
            [
                "trial",
                "spectral_distance",
                "positions_n",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nmu0      = {mu0:.10f}"
    )
    print(
        f"sigma0   = {sigma0:.10f}"
    )
    print(
        f"threshold= {threshold:.10f}"
    )

    print("\n" + "=" * 90)
    print("FOURIER SUMMARY")
    print("=" * 90)

    print(
        severity_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 90)
    print("FOG30 VS FOG45 — PAIRED BY TRIAL")
    print("=" * 90)

    print(
        paired.to_string(
            index=False
        )
    )

    delta = paired[
        "fog45_minus_fog30"
    ].to_numpy(
        dtype=float
    )

    print(
        "\nMean Fog45 - Fog30 = "
        f"{delta.mean():.10f}"
    )

    print(
        "Trials where Fog45 is MORE "
        "spectrally degraded than Fog30: "
        f"{int((delta > 0).sum())}/5"
    )

    print(
        "Trials where Fog45 is LESS "
        "spectrally degraded than Fog30: "
        f"{int((delta < 0).sum())}/5"
    )

    print("\n" + "=" * 90)
    print("SAVED")
    print("=" * 90)

    for name in [
        "selected_recordings.csv",
        "common_sample_ids.csv",
        "position_distances.csv",
        "trial_distances.csv",
        "threshold.csv",
        "severity_summary.csv",
        "fog30_vs_fog45_paired.csv",
        "fourier_distance_vs_fog.png",
    ]:
        print(OUT / name)


if __name__ == "__main__":
    main()
