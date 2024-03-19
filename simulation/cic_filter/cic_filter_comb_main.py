import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from absl import app, flags

FLAGS = flags.FLAGS

# Filters to use for the overall averaging filter.
FILTERS = [
    np.array([1], dtype=np.float64),
    np.array([0.25, 0.75, 0.75, 0.25]),
    np.array([0.25, 1, 1, 0.25]),
    np.array([0, 1, 1, 0]),
]

# FFT size.
FFT_SIZE = 1048576


def _create_comb_filter_diff(coefficients: np.ndarray, R: int) -> np.ndarray:
    """Creates the difference comb filter coefficients.

    Args:
        coefficients: The desired filter coefficients without zero padding.
        R: Downsampling factor.

    Returns:
        The length-(R + 1) difference coefficients of the comb filter.
    """
    coefficients_with_zero_padding = np.concatenate(
        (np.array([0]), coefficients, np.array([0])))
    diff = np.diff(coefficients_with_zero_padding).astype(np.float64)
    # Scale the filter to integrate to 1.
    diff /= np.mean(coefficients)

    comb = np.zeros(R + 1)
    coefficient_step = R // len(coefficients)
    for coefficient_index, coefficient in enumerate(diff):
        comb[coefficient_step * coefficient_index] = coefficient
    return comb


def compare_crc_filter_combs(fs: float, T: float, R: int) -> None:
    """Compares different comb filters for a CIC filter stage.

    Args:
        fs: Sampling frequency.
        T: Maximum time to simulate.
        R: Downsampling factor.
    """
    length = int(fs * T)
    delta = np.zeros(length)
    delta[0] = 1

    # Integrate the signal.
    integrated = np.cumsum(delta)

    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    omega = np.linspace(0, 2 * np.pi, length, endpoint=False)

    for coefficients in FILTERS:
        # Apply a comb filter to the signal.
        comb_filter_diff = _create_comb_filter_diff(coefficients, R)
        combed = np.convolve(integrated, comb_filter_diff)

        # Plot the spectrum.
        combed_fft = np.fft.fft(combed, length)
        combed_fft_abs = np.abs(combed_fft)
        ax.plot(omega, 20 * np.log10(combed_fft_abs), label=coefficients)

    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("FFT magnitude [dB]")
    ax.set_xlim([0, 2 * np.pi / R])
    ax.set_ylim([50, 62])
    ax.axvline(np.pi / R, linestyle="--", linewidth=2, zorder=0)
    ax.legend()
    plt.show()


def plot_comb_filter_spectrum(R: int) -> None:
    """Plots the spectrum of each comb filter.

    Args:
        R: Downsampling factor.
    """
    # Plot the DTFT of the comb filters directly.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    omega = np.linspace(0, 2 * np.pi, FFT_SIZE, endpoint=False)

    for coefficients in FILTERS:
        comb_filter_diff = _create_comb_filter_diff(coefficients, R)
        comb_filter = np.cumsum(comb_filter_diff)
        combed_fft = np.fft.fft(comb_filter, FFT_SIZE)
        combed_fft_abs = np.abs(combed_fft)
        ax.plot(omega, 20 * np.log10(combed_fft_abs), label=coefficients)

    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("FFT magnitude [dB]")
    ax.set_xlim([0, 2 * np.pi / R])
    ax.set_ylim([50, 62])
    ax.axvline(np.pi / R, linestyle="--", linewidth=2, zorder=0)
    ax.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    compare_crc_filter_combs(FLAGS.fs, FLAGS.T, FLAGS.R)
    plot_comb_filter_spectrum(FLAGS.R)


if __name__ == "__main__":
    flags.DEFINE_float("fs", 1000000, "Sampling frequency.")
    flags.DEFINE_float("T", 1, "Maximum time to simulate.")
    flags.DEFINE_integer("R", 1024, "Downsampling factor.")

    app.run(main)
