import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging

FLAGS = flags.FLAGS

# Tau values to plot.
TAU = [0, 0.5, 1, 1.5, 2, 2.5, 3]

# Empirically determined scaling factor of the exponential measured by SCuM.
EXPONENTIAL_SCALING_FACTOR = 870

# Standard deviation of SCuM's ADC.
SIGMA = 5


def plot_sample_distribution() -> None:
    """Plots the distribution of the log ADC samples as a function of tau."""
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.linspace(-0.25, 0.25, 10000)
    for tau in TAU:
        pdf = 1 / (np.sqrt(2 * np.pi) * SIGMA) * np.exp(
            -(EXPONENTIAL_SCALING_FACTOR**2 * np.exp(-2 * tau) *
              (np.exp(x) - 1)**2) /
            (2 * SIGMA**2)) * EXPONENTIAL_SCALING_FACTOR * np.exp(x - tau)
        pdf_plot = plt.plot(x, pdf, label=f"tau={tau}")

        # Approximate the pdf.
        stddev = SIGMA * np.exp(tau) / EXPONENTIAL_SCALING_FACTOR
        pdf_approximated = 1 / (np.sqrt(2 * np.pi) * stddev) * np.exp(
            -1 / 2 * x**2 / stddev**2)
        plt.plot(x,
                 pdf_approximated,
                 "--",
                 c=pdf_plot[0].get_color(),
                 label=f"tau={tau} (approximated)")
    ax.set_title("Probability density of the log ADC samples")
    ax.set_xlabel("Difference from actual value")
    ax.set_ylabel("Probability density")
    plt.legend()
    plt.show()


def plot_sample_stddev() -> None:
    """Plots the standard deviation of the log ADC samples as a function of tau."""
    fig, ax = plt.subplots(figsize=(12, 8))
    tau = np.linspace(np.min(TAU), np.max(TAU), 10000)
    stddev = SIGMA * np.exp(tau) / EXPONENTIAL_SCALING_FACTOR
    plt.plot(tau, stddev, label="Standard deviation")
    ax.set_title("Standard deviation of the log ADC samples")
    ax.set_xlabel("tau")
    ax.set_ylabel("Standard deviation [bits]")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_sample_distribution()
    plot_sample_stddev()


if __name__ == "__main__":
    app.run(main)
