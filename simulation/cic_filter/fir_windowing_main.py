import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from absl import app, flags, logging

FLAGS = flags.FLAGS

# Windows to simulate.
WINDOWS = {
    "Rectangular": np.ones,
    "Hann": np.hanning,
    "Hamming": np.hamming,
    "Blackman": np.blackman,
}


def _calculate_averaging_snr(coefficients: np.ndarray) -> float:
    """Calculates the SNR after applying the filter.

    The filter is scaled, such that the signal amplitude stays constant.

    Args:
        filter: Filter coefficients.

    Returns:
        The SNR increase in dB after applying the averaging filter.
    """
    normalized_filter = coefficients / np.linalg.norm(coefficients, 1)
    return 20 * np.log10(1 / np.linalg.norm(normalized_filter))


def plot_windows(fir_length: int, fft_length: int) -> None:
    """Plots different windows.

    Args:
        fir_length: FIR filter length.
        fft_length: FFT length.
    """
    plt.style.use(["science", "grid"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    omega = np.linspace(-np.pi, np.pi, fft_length, endpoint=False)
    for name, window in WINDOWS.items():
        window_coefficients = window(fir_length + 2)[1:-1]
        window_coefficients /= np.linalg.norm(window_coefficients, 1)
        logging.info("%s: %f dB", name,
                     _calculate_averaging_snr(window_coefficients))
        ax1.plot(window_coefficients, label=name)
        window_fft = np.fft.fftshift(np.fft.fft(window_coefficients,
                                                fft_length))
        window_fft_abs = np.abs(window_fft)
        ax2.plot(omega, 20 * np.log10(window_fft_abs), label=name)
    ax1.set_xlabel("Sample")
    ax1.set_ylabel("Window amplitude")
    ax2.set_xlabel("Frequency [rad]")
    ax2.set_ylabel("FFT magnitude [dB]")
    ax2.set_ylim(bottom=-100)
    ax1.legend()
    ax2.legend()
    plt.show()


def plot_fir_with_windowing(corner_frequency: float, fir_length: int,
                            fft_length: int) -> None:
    """Plots the FIR filter spectrum with different windows.

    Args:
        corner_frequency: Corner frequency in units of pi.
        fir_length: FIR filter length.
        fft_length: FFT length.
    """
    # Generate the frequency response.
    sinc_fft = np.zeros(fir_length)
    sinc_fft[:int(fir_length // 2 * corner_frequency)] = 1
    sinc_fft[-int(fir_length // 2 * corner_frequency):] = 1
    sinc = np.real(np.fft.fftshift(np.fft.ifft(sinc_fft)))

    # Plot the spectrum after applying different windows.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    omega = np.linspace(0, 2 * np.pi, fft_length, endpoint=False)
    for name, window in WINDOWS.items():
        window_coefficients = window(fir_length + 2)[1:-1]
        windowed_sinc = sinc * window_coefficients
        windowed_sinc_fft = np.fft.fft(windowed_sinc, fft_length)
        windowed_sinc_fft_abs = np.abs(windowed_sinc_fft)
        ax.plot(omega, 20 * np.log10(windowed_sinc_fft_abs), label=name)
    ax.set_xlabel("Frequency [rad]")
    ax.set_ylabel("FFT magnitude [dB]")
    ax.set_xlim([0, np.pi])
    ax.set_ylim(bottom=-120)
    ax.axvline(np.pi * corner_frequency, linestyle="--", linewidth=2, zorder=0)
    ax.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_windows(FLAGS.fir_length, FLAGS.fft_length)
    plot_fir_with_windowing(FLAGS.corner_frequency, FLAGS.fir_length,
                            FLAGS.fft_length)


if __name__ == "__main__":
    flags.DEFINE_float("corner_frequency", 0.25,
                       "Corner frequency in units of pi.")
    flags.DEFINE_integer("fir_length", 32, "FIR filter length.")
    flags.DEFINE_integer("fft_length", 1048576, "FFT length.")

    app.run(main)
