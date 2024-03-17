import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import scipy.signal
from absl import app, flags

FLAGS = flags.FLAGS

# Comb filter.
COMB_FILTER = np.array([0, 1, 1, 0])
COMB_NUM_TAPS = len(COMB_FILTER)


def _simulate_cic_filter(signal: np.ndarray, fs: float, T: float,
                         R: int) -> None:
    """Simulates a single stage of a CIC filter.

    Args:
        signal: Signal vector.
        fs: Sampling frequency.
        T: Maximum time to simulate.
        R: Decimation factor.
    """
    # Integrate the signal.
    integrated = np.cumsum(signal)

    # Decimate the signal twice to compare against the custom comb filter.
    decimated = scipy.signal.decimate(integrated,
                                      R // COMB_NUM_TAPS,
                                      ftype="fir")
    decimated = scipy.signal.decimate(decimated, COMB_NUM_TAPS, ftype="fir")

    # Apply a comb filter to the signal.
    combed = np.convolve(decimated, np.array([1, -1]))
    return combed


def _calculate_spectrum_magnitude(signal: np.ndarray,
                                  length: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the magnitude of the spectrum of the signal.

    Args:
        signal: Signal.
        length: FFT length.

    Returns:
        A 2-tuple consisting of the omega axis and the magnitude vector.
    """
    signal_fft = np.fft.fft(signal, length)
    signal_fft_abs = np.abs(signal_fft)
    omega = np.linspace(0, 2 * np.pi, length, endpoint=False)
    return omega, signal_fft_abs


def _create_comb_filter_diff(coefficients: np.ndarray) -> np.ndarray:
    """Creates the difference comb filter coefficients.

    Args:
        coefficients: The desired filter coefficients without zero padding.

    Returns:
        The difference coefficients of the comb filter.
    """
    coefficients_with_zero_padding = np.concatenate(
        (np.array([0]), coefficients, np.array([0])))
    diff = np.diff(coefficients_with_zero_padding).astype(np.float64)
    diff /= np.mean(coefficients)
    return diff


def plot_cic_filter_output(signal: np.ndarray, fs: float, T: float,
                           R: int) -> None:
    """Plots the output of a single stage of a CIC filter.

    Args:
        signal: Signal vector.
        fs: Sampling frequency.
        T: Maximum time to simulate.
        R: Decimation factor.
    """
    output = _simulate_cic_filter(signal, fs, T, R)

    # Plot the signal.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(output)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Output")
    plt.show()

    # Plot the spectrum.
    fig, ax = plt.subplots(figsize=(12, 8))
    decimated_length = int(fs * T) // R
    omega, output_fft_abs = _calculate_spectrum_magnitude(
        output, decimated_length)
    ax.plot(omega, 20 * np.log10(output_fft_abs))
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("FFT magnitude [dB]")
    plt.show()


def plot_cic_filter_comb_filter(signal: np.ndarray, fs: float, T: float,
                                R: int) -> None:
    """Plots the output of a single stage of a CIC filter with a different comb
    filter.

    Args:
        signal: Signal vector.
        fs: Sampling frequency.
        T: Maximum time to simulate.
        R: Decimation factor.
    """
    # Integrate the signal.
    integrated = np.cumsum(signal)

    # Decimate the signal.
    decimated = scipy.signal.decimate(integrated,
                                      R // COMB_NUM_TAPS,
                                      ftype="fir")

    # Apply a comb filter to the signal.
    comb_filter_diff = _create_comb_filter_diff(COMB_FILTER)
    combed = np.convolve(decimated, comb_filter_diff)

    # Decimate the signal a second time.
    output = scipy.signal.decimate(combed, COMB_NUM_TAPS, ftype="fir")

    # Simulate the standard CIC filter stage.
    output_standard = _simulate_cic_filter(signal, fs, T, R)

    # Plot the spectrum.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    decimated_length = int(fs * T) // R
    omega, output_fft_abs = _calculate_spectrum_magnitude(
        output, decimated_length)
    ax.plot(omega,
            20 * np.log10(output_fft_abs / np.max(output_fft_abs)),
            label=f"Comb filter: {COMB_FILTER}")
    omega_standard, output_standard_fft_abs = _calculate_spectrum_magnitude(
        output_standard, decimated_length)
    ax.plot(omega_standard,
            20 *
            np.log10(output_standard_fft_abs / np.max(output_standard_fft_abs)),
            "--",
            label="Standard comb filter")
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("FFT magnitude [dB]")
    ax.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    if FLAGS.fin is None:
        signal = np.zeros(int(FLAGS.fs * FLAGS.T))
        signal[0] = 1
    else:
        signal = np.cos(2 * np.pi * FLAGS.fin / FLAGS.fs *
                        np.arange(int(FLAGS.fs * FLAGS.T)))
    if FLAGS.noise is not None:
        signal += np.random.normal(scale=FLAGS.noise, size=signal.shape)

    plot_cic_filter_output(signal, FLAGS.fs, FLAGS.T, FLAGS.R)
    plot_cic_filter_comb_filter(signal, FLAGS.fs, FLAGS.T, FLAGS.R)


if __name__ == "__main__":
    flags.DEFINE_float("fs", 1000000, "Sampling frequency.")
    flags.DEFINE_float("T", 1, "Maximum time to simulate.")
    flags.DEFINE_integer("R", 1024, "Decimation factor.")
    flags.DEFINE_float("fin", None, "Input frequency.")
    flags.DEFINE_float("noise", None, "Noise amplitude.")

    app.run(main)
