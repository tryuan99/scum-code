import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from absl import app, flags

FLAGS = flags.FLAGS

# Maximum number of averaging stages to simulate.
MAX_AVERAGING_STAGES = 20


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


def plot_averaging_snr(R: int) -> None:
    """Plots the SNR as a function of the number of averaging stages.

    Args:
        R: Number of samples to average over.
    """
    # Simulate the SNR as a function of the number of averaging stages.
    snr = np.zeros(MAX_AVERAGING_STAGES + 1)
    averaging_response = np.ones(R) / R
    response = np.ones(1)
    for num_averaging_stages in range(1, MAX_AVERAGING_STAGES + 1):
        response = np.convolve(response, averaging_response)
        snr[num_averaging_stages] = _calculate_averaging_snr(response)

    # Plot the SNR.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(snr)
    ax.set_xlabel("Number of averaging stages")
    ax.set_ylabel("SNR increase [dB]")
    ax.set_title(f"Averaging over {R} samples")
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_averaging_snr(FLAGS.R)


if __name__ == "__main__":
    flags.DEFINE_integer("R", 4, "Number of samples to average over.")

    app.run(main)
