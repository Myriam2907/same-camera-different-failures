import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

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


OUTPUT_ROOT = "results/spectrum"

REFERENCE_PROFILE_PATH = os.path.join(
    OUTPUT_ROOT,
    "clear_reference_profile.csv",
)

SUMMARY_PATH = os.path.join(
    OUTPUT_ROOT,
    "summary.csv",
)

DISTANCE_PLOT_PATH = os.path.join(
    OUTPUT_ROOT,
    "spectral_distance_vs_severity.png",
)

PROFILE_PLOT_PATH = os.path.join(
    OUTPUT_ROOT,
    "mean_radial_spectra.png",
)


# ============================================================
# FOURIER SPECTRUM
# ============================================================

def normalized_radial_power_spectrum(image):
    """
    Compute a normalized radially averaged Fourier
    power spectrum.

    Pipeline:

        BGR
        -> grayscale
        -> FFT2
        -> fftshift
        -> |F|^2
        -> log(1 + power)
        -> radial average
        -> normalize profile to sum = 1

    The output describes how image energy is distributed
    across spatial frequencies while reducing dependence
    on orientation.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = gray.astype(
        np.float64
    )

    # --------------------------------------------------------
    # 2-D Fourier transform
    # --------------------------------------------------------

    spectrum = np.fft.fft2(
        gray
    )

    spectrum = np.fft.fftshift(
        spectrum
    )

    # --------------------------------------------------------
    # Power spectrum
    # --------------------------------------------------------

    power = np.abs(
        spectrum
    ) ** 2

    # Dynamic range of Fourier power is enormous.
    # log1p keeps weak and strong frequencies measurable.
    log_power = np.log1p(
        power
    )

    height, width = (
        gray.shape
    )

    center_y = height // 2
    center_x = width // 2

    y, x = np.indices(
        (height, width)
    )

    radius = np.sqrt(
        (x - center_x) ** 2
        +
        (y - center_y) ** 2
    )

    radius = radius.astype(
        np.int32
    )

    # Use frequencies only up to the largest complete
    # circle that fits inside the image.
    max_radius = min(
        center_x,
        center_y,
    )

    valid = (
        radius < max_radius
    )

    radius_valid = radius[
        valid
    ].ravel()

    power_valid = log_power[
        valid
    ].ravel()

    # --------------------------------------------------------
    # Radial average
    # --------------------------------------------------------

    radial_sum = np.bincount(
        radius_valid,
        weights=power_valid,
        minlength=max_radius,
    )

    radial_count = np.bincount(
        radius_valid,
        minlength=max_radius,
    )

    radial_profile = radial_sum / np.maximum(
        radial_count,
        1,
    )

    radial_profile = radial_profile[
        :max_radius
    ]

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    total = radial_profile.sum()

    if total <= 0:
        raise RuntimeError(
            "Invalid Fourier radial spectrum."
        )

    radial_profile = (
        radial_profile / total
    )

    return radial_profile


# ============================================================
# SPECTRAL DISTANCE
# ============================================================

def spectral_distance(
    profile,
    reference,
):
    """
    Euclidean distance between two normalized radial
    power spectra.
    """

    if len(profile) != len(reference):
        raise RuntimeError(
            "Spectrum length mismatch."
        )

    return float(
        np.linalg.norm(
            profile - reference
        )
    )


# ============================================================
# LOAD DATASET SPECTRA
# ============================================================

def load_dataset_profiles(
    dataset_name,
    dataset_info,
):
    """
    Compute one normalized Fourier profile for every frame.
    """

    root = dataset_info[
        "root"
    ]

    metadata_path = os.path.join(
        root,
        "frames.csv",
    )

    frame_dir = os.path.join(
        root,
        "frames",
    )

    if not os.path.exists(
        metadata_path
    ):
        raise RuntimeError(
            f"Missing metadata: {metadata_path}"
        )

    metadata = pd.read_csv(
        metadata_path
    )

    profiles = []
    rows = []

    print(
        f"Computing spectra for {dataset_name}..."
    )

    for index, row in metadata.iterrows():

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
                f"Could not read image: {image_path}"
            )

        profile = (
            normalized_radial_power_spectrum(
                image
            )
        )

        profiles.append(
            profile
        )

        rows.append({
            "sample_id":
                sample_id,

            "weather":
                row["weather"],

            "severity":
                int(row["severity"]),

            "route_s_m":
                float(row["route_s_m"]),

            "distance_to_sign_m":
                float(
                    row["distance_to_sign_m"]
                ),

            "is_junction":
                int(row["is_junction"]),
        })

        if (
            index == 0
            or (index + 1) % 20 == 0
            or index + 1 == len(metadata)
        ):
            print(
                f"  {index + 1}/{len(metadata)}"
            )

    profiles = np.stack(
        profiles,
        axis=0,
    )

    rows_df = pd.DataFrame(
        rows
    )

    return (
        rows_df,
        profiles,
    )


# ============================================================
# CLEAR REFERENCE
# ============================================================

def build_clear_reference(
    clear_profiles,
):
    """
    Reference spectrum = mean normalized radial spectrum
    across all clear-weather frames.

    Clear spectral distances provide normal scene variation.

    Threshold:

        T = μ0 + 3σ0
    """

    reference = np.mean(
        clear_profiles,
        axis=0,
    )

    # Re-normalize after averaging.
    reference = (
        reference
        / reference.sum()
    )

    clear_distances = np.array([
        spectral_distance(
            profile,
            reference,
        )
        for profile
        in clear_profiles
    ])

    mu0 = float(
        np.mean(
            clear_distances
        )
    )

    sigma0 = float(
        np.std(
            clear_distances,
            ddof=1,
        )
    )

    threshold = (
        mu0
        + 3.0 * sigma0
    )

    return (
        reference,
        clear_distances,
        mu0,
        sigma0,
        threshold,
    )


# ============================================================
# SAVE REFERENCE
# ============================================================

def save_reference_profile(
    reference,
):

    df = pd.DataFrame({
        "radius_bin":
            np.arange(
                len(reference)
            ),

        "normalized_power":
            reference,
    })

    df.to_csv(
        REFERENCE_PROFILE_PATH,
        index=False,
    )


# ============================================================
# PROCESS ONE CONDITION
# ============================================================

def evaluate_condition(
    name,
    metadata,
    profiles,
    reference,
    threshold,
):
    """
    Compute spectral distance and alert for all samples.
    """

    distances = np.array([
        spectral_distance(
            profile,
            reference,
        )
        for profile
        in profiles
    ])

    alerts = (
        distances > threshold
    )

    result = metadata.copy()

    result[
        "spectral_distance"
    ] = distances

    result[
        "threshold"
    ] = threshold

    result[
        "degradation_alert"
    ] = alerts.astype(
        int
    )

    output_path = os.path.join(
        OUTPUT_ROOT,
        f"{name}_per_frame.csv",
    )

    result.to_csv(
        output_path,
        index=False,
    )

    summary = {
        "dataset":
            name,

        "severity":
            int(
                metadata[
                    "severity"
                ].iloc[0]
            ),

        "frames":
            len(result),

        "mean_spectral_distance":
            float(
                distances.mean()
            ),

        "median_spectral_distance":
            float(
                np.median(
                    distances
                )
            ),

        "std_spectral_distance":
            float(
                distances.std(
                    ddof=1
                )
            ),

        "alert_count":
            int(
                alerts.sum()
            ),

        "alert_rate":
            float(
                alerts.mean()
            ),
    }

    return (
        result,
        summary,
        distances,
    )


# ============================================================
# PLOTS
# ============================================================

def save_distance_plot(
    summaries,
    threshold,
):
    """
    Plot mean spectral distance versus fog severity.
    """

    df = pd.DataFrame(
        summaries
    ).sort_values(
        "severity"
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        df["severity"],
        df["mean_spectral_distance"],
        marker="o",
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
        "Mean spectral distance"
    )

    plt.title(
        "Fourier Degradation Score vs Fog Severity"
    )

    plt.xticks(
        [0, 45, 90]
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        DISTANCE_PLOT_PATH,
        dpi=200,
    )

    plt.close()


def save_profile_plot(
    mean_profiles,
):
    """
    Plot average normalized radial spectrum
    for Clear, Fog45, and Fog90.
    """

    plt.figure(
        figsize=(8, 5)
    )

    for name in [
        "clear",
        "fog45",
        "fog90",
    ]:

        profile = mean_profiles[
            name
        ]

        plt.plot(
            np.arange(
                len(profile)
            ),
            profile,
            label=name,
        )

    plt.xlabel(
        "Radial frequency bin"
    )

    plt.ylabel(
        "Normalized log power"
    )

    plt.title(
        "Mean Radially Averaged Fourier Spectrum"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        PROFILE_PLOT_PATH,
        dpi=200,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Compute spectra for all three conditions
    # --------------------------------------------------------

    data = {}

    for (
        name,
        info,
    ) in DATASETS.items():

        (
            metadata,
            profiles,
        ) = load_dataset_profiles(
            name,
            info,
        )

        data[name] = {
            "metadata":
                metadata,

            "profiles":
                profiles,
        }

    # --------------------------------------------------------
    # Sanity check:
    # all three conditions must use identical sample IDs.
    # --------------------------------------------------------

    clear_ids = data[
        "clear"
    ][
        "metadata"
    ][
        "sample_id"
    ].to_numpy()

    for name in [
        "fog45",
        "fog90",
    ]:

        ids = data[
            name
        ][
            "metadata"
        ][
            "sample_id"
        ].to_numpy()

        if not np.array_equal(
            clear_ids,
            ids,
        ):

            raise RuntimeError(
                f"{name} sample IDs do not "
                f"match Clear."
            )

    # --------------------------------------------------------
    # Build normal Clear spectral reference
    # --------------------------------------------------------

    (
        reference,
        clear_distances,
        mu0,
        sigma0,
        threshold,
    ) = build_clear_reference(
        data[
            "clear"
        ][
            "profiles"
        ]
    )

    save_reference_profile(
        reference
    )

    print()
    print(
        "=" * 60
    )

    print(
        "CLEAR REFERENCE"
    )

    print(
        "=" * 60
    )

    print(
        f"mu0:       {mu0:.8f}"
    )

    print(
        f"sigma0:    {sigma0:.8f}"
    )

    print(
        f"threshold: {threshold:.8f}"
    )

    # --------------------------------------------------------
    # Evaluate all conditions
    # --------------------------------------------------------

    summaries = []

    mean_profiles = {}

    for name in [
        "clear",
        "fog45",
        "fog90",
    ]:

        metadata = data[
            name
        ][
            "metadata"
        ]

        profiles = data[
            name
        ][
            "profiles"
        ]

        (
            result,
            summary,
            distances,
        ) = evaluate_condition(
            name,
            metadata,
            profiles,
            reference,
            threshold,
        )

        summaries.append(
            summary
        )

        mean_profiles[name] = (
            profiles.mean(
                axis=0
            )
        )

        print()
        print(
            "=" * 60
        )

        print(
            "Condition:",
            name,
        )

        print(
            "=" * 60
        )

        print(
            "Frames:",
            summary[
                "frames"
            ],
        )

        print(
            "Mean spectral distance:",
            f'{summary["mean_spectral_distance"]:.8f}',
        )

        print(
            "Median spectral distance:",
            f'{summary["median_spectral_distance"]:.8f}',
        )

        print(
            "Alerts:",
            summary[
                "alert_count"
            ],
            "/",
            summary[
                "frames"
            ],
        )

        print(
            "Alert rate:",
            f'{summary["alert_rate"] * 100:.2f}%',
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_df = pd.DataFrame(
        summaries
    )

    # Add clear calibration information to all rows
    # so it is preserved with the experiment.
    summary_df[
        "clear_mu0"
    ] = mu0

    summary_df[
        "clear_sigma0"
    ] = sigma0

    summary_df[
        "alert_threshold"
    ] = threshold

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------

    save_distance_plot(
        summaries,
        threshold,
    )

    save_profile_plot(
        mean_profiles,
    )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "FOURIER MONITOR COMPLETE"
    )

    print(
        "Summary:",
        SUMMARY_PATH,
    )

    print(
        "Distance plot:",
        DISTANCE_PLOT_PATH,
    )

    print(
        "Spectrum plot:",
        PROFILE_PLOT_PATH,
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()