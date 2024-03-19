import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from absl import app, flags

FLAGS = flags.FLAGS

# Comb filter.
COMB_FILTER = np.array([0.25, 0.75, 0.75, 0.25])


def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convolves two signals but only includes boundary effects at the beginning.

    Args:
        a: First input signal.
        b: Second input signal.

    Returns:
        The convolution of a and b with boundary effects only at the beginning.
    """
    max_length = max(len(a), len(b))
    return np.convolve(a, b)[:max_length]


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
    return diff


def _simulate_cic_filter(signal: np.ndarray, fs: float, T: float, R: int,
                         N: int, comb_filter: np.ndarray) -> None:
    """Simulates a CIC filter.

    Args:
        signal: Signal vector.
        fs: Sampling frequency.
        T: Maximum time to simulate.
        R: Downsampling factor.
        N: Number of stages.
    """
    # Integrate the signal N times.
    integrated = signal
    for _ in range(N):
        integrated = np.cumsum(integrated)

    # Downsample the signal.
    num_comb_filter_taps = len(comb_filter)
    downsampled = integrated[::R // num_comb_filter_taps]

    # Apply a comb filter to the signal N times.
    comb_filter_diff = _create_comb_filter_diff(comb_filter)
    combed = downsampled
    for _ in range(N):
        combed = _convolve(combed, comb_filter_diff)

    # Downsample the signal a second time.
    output = combed[::num_comb_filter_taps]
    return output


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


def plot_cic_filter_output(signal: np.ndarray, fs: float, T: float, R: int,
                           N: int) -> None:
    """Plots the output of a standard CIC filter.

    Args:
        signal: Signal vector.
        fs: Sampling frequency.
        T: Maximum time to simulate.
        R: Downsampling factor.
        N: Number of stages.
    """
    output = _simulate_cic_filter(signal, fs, T, R, N, np.ones(1))

    # Plot the signal.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.stem(output)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Output")
    plt.show()

    # Plot the spectrum.
    fig, ax = plt.subplots(figsize=(12, 8))
    fft_length = int(fs * T)
    omega, output_fft_abs = _calculate_spectrum_magnitude(output, fft_length)
    ax.plot(omega, 20 * np.log10(output_fft_abs))
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("FFT magnitude [dB]")
    plt.show()


def plot_cic_filter_comb_filter(signal: np.ndarray, fs: float, T: float, R: int,
                                N: int) -> None:
    """Plots the output of a CIC filter with a custom comb filter.

    Args:
        signal: Signal vector.
        fs: Sampling frequency.
        T: Maximum time to simulate.
        R: Downsampling factor.
        N: Number of stages.
    """
    # Simulate the CIC filter with a custom comb filter.
    output = _simulate_cic_filter(signal, fs, T, R, N, COMB_FILTER)

    # Simulate the standard CIC filter.
    output_standard = _simulate_cic_filter(signal, fs, T, R, N, np.ones(1))

    # Plot the spectrum.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    fft_length = int(fs * T)
    omega, output_fft_abs = _calculate_spectrum_magnitude(output, fft_length)
    omega_standard, output_standard_fft_abs = _calculate_spectrum_magnitude(
        output_standard, fft_length)
    ax.plot(omega,
            20 * np.log10(output_fft_abs / np.max(output_fft_abs)),
            label=f"Comb filter: {COMB_FILTER}")
    ax.plot(omega_standard,
            20 *
            np.log10(output_standard_fft_abs / np.max(output_standard_fft_abs)),
            "--",
            label=f"Standard comb filter: {np.ones(1)}")
    ax.set_xlabel(r"$\omega$ [rad]")
    ax.set_ylabel("Normalized FFT magnitude [dB]")
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

    plot_cic_filter_output(signal, FLAGS.fs, FLAGS.T, FLAGS.R, FLAGS.N)
    plot_cic_filter_comb_filter(signal, FLAGS.fs, FLAGS.T, FLAGS.R, FLAGS.N)


if __name__ == "__main__":
    flags.DEFINE_float("fs", 1000000, "Sampling frequency.")
    flags.DEFINE_float("T", 1, "Maximum time to simulate.")
    flags.DEFINE_integer("R", 1024, "Downsampling factor.")
    flags.DEFINE_integer("N", 1, "Number of stages.")
    flags.DEFINE_float("fin", None, "Input frequency.")
    flags.DEFINE_float("noise", None, "Noise amplitude.")

    app.run(main)
