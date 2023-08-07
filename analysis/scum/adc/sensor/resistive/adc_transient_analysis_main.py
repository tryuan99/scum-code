import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging

from analysis.scum.adc.sensor.resistive.adc_data import (
    EXPONENTIAL_SCALING_FACTOR, SIGMA, ExponentialAdcData)

FLAGS = flags.FLAGS

# Offset of the decaying exponential.
EXPONENTIAL_OFFSET = 127

# Tau values to plot.
TAUS = [0, 0.5, 1, 1.5, 2, 2.5, 3]


def _generate_transient_adc_data(tau: float,
                                 sampling_rate: float,
                                 num_taus: float = 5) -> np.ndarray:
    """Generates an exponentially decaying ADC data with the given time constant.

    Args:
        tau: Time constant in seconds.
        sampling_rate: Sampling rate in Hz.
        num_taus: Duration of the generated ADC data in number of taus.

    Returns:
        The ADC samples of a decaying exponential.
    """
    # Generate the time axis.
    t_axis = np.arange(0, num_taus * tau, 1 / sampling_rate)

    # Generate a decaying exponential with an offset.
    exponential = EXPONENTIAL_SCALING_FACTOR * np.exp(
        -t_axis / tau) + EXPONENTIAL_OFFSET

    # Generate some noise.
    noise = np.random.normal(scale=SIGMA, size=exponential.shape)

    # Round the ADC samples to the nearest integer.
    return np.round(exponential + noise)


def plot_log_noise_distribution() -> None:
    """Plots the distribution of the noise of the log ADC samples."""
    # Plot the distribution of the noise.
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.linspace(-0.25, 0.25, 10000)
    for tau in TAUS:
        pdf = 1 / (np.sqrt(2 * np.pi) * SIGMA) * np.exp(
            -(EXPONENTIAL_SCALING_FACTOR**2 * np.exp(-2 * tau) *
              (np.exp(x) - 1)**2) /
            (2 * SIGMA**2)) * EXPONENTIAL_SCALING_FACTOR * np.exp(x - tau)
        pdf_plot = plt.plot(x, pdf, label=f"t={tau}tau")

        # Approximate the pdf.
        stddev = SIGMA * np.exp(tau) / EXPONENTIAL_SCALING_FACTOR
        pdf_approximated = 1 / (np.sqrt(2 * np.pi) * stddev) * np.exp(
            -1 / 2 * x**2 / stddev**2)
        plt.plot(x,
                 pdf_approximated,
                 "--",
                 c=pdf_plot[0].get_color(),
                 label=f"t={tau}tau (approximated)")
    ax.set_title("Probability density of the noise of the log ADC samples")
    ax.set_xlabel("Noise value [bits]")
    ax.set_ylabel("Probability density")
    plt.legend()
    plt.show()

    # Plot the difference between the actual distribution and the approximated
    # distribution.
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.linspace(-0.25, 0.25, 10000)
    for tau in TAUS:
        pdf = 1 / (np.sqrt(2 * np.pi) * SIGMA) * np.exp(
            -(EXPONENTIAL_SCALING_FACTOR**2 * np.exp(-2 * tau) *
              (np.exp(x) - 1)**2) /
            (2 * SIGMA**2)) * EXPONENTIAL_SCALING_FACTOR * np.exp(x - tau)

        # Approximate the pdf.
        stddev = SIGMA * np.exp(tau) / EXPONENTIAL_SCALING_FACTOR
        pdf_approximated = 1 / (np.sqrt(2 * np.pi) * stddev) * np.exp(
            -1 / 2 * x**2 / stddev**2)
        plt.plot(x, pdf_approximated - pdf, label=f"t={tau}tau")
    ax.set_title(
        "Approximation error of the probability density of the noise of the log ADC samples"
    )
    ax.set_xlabel("Noise value [bits]")
    ax.set_ylabel("Difference in the probability density")
    plt.legend()
    plt.show()


def plot_log_noise_stddev() -> None:
    """Plots the standard deviation of the noise of the log ADC samples."""
    fig, ax = plt.subplots(figsize=(12, 8))
    tau = np.linspace(np.min(TAUS), np.max(TAUS), 10000)
    stddev = SIGMA * np.exp(tau) / EXPONENTIAL_SCALING_FACTOR
    plt.plot(tau, stddev, label="Standard deviation")
    ax.set_title("Standard deviation of the noise of the log ADC samples")
    ax.set_xlabel("tau")
    ax.set_ylabel("Standard deviation [bits]")
    plt.legend()
    plt.show()


def plot_slope_stddev(sampling_rate: float) -> None:
    """Plots the standard deviation of the estimated slope.

    Args:
        sampling_rate: Sampling rate in Hz.
    """
    taus = np.linspace(0.1, 20, 500)
    slope_stddevs = []
    for tau in taus:
        adc_output = _generate_transient_adc_data(tau, sampling_rate)
        adc_data = ExponentialAdcData(adc_output, sampling_rate)
        weighted_linear_regression = adc_data.perform_weighted_linear_regression(
        )
        slope_stddevs.append(np.sqrt(weighted_linear_regression.slope_variance))

    # Plot the standard deviation of the estimated slope.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(taus, slope_stddevs, label="Empirical")
    plt.plot(taus,
             np.sqrt(8 * SIGMA**2 /
                     (EXPONENTIAL_SCALING_FACTOR**2 * sampling_rate * taus**3)),
             label="Approximated")
    ax.set_title("Standard deviation of the slope")
    ax.set_xlabel("Slope [1/s]")
    ax.set_ylabel("Standard deviation [1/s]")
    plt.legend()
    plt.show()

    # Plot the difference between the actual standard deviation and the
    # approximated standard deviation.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(
        taus,
        np.sqrt(8 * SIGMA**2 /
                (EXPONENTIAL_SCALING_FACTOR**2 * sampling_rate * taus**3)) -
        slope_stddevs,
        label="Approximated - actual")
    ax.set_title("Approximation error of the standard deviation of the slope")
    ax.set_xlabel("Slope [1/s]")
    ax.set_ylabel("Difference in the standard deviation [1/s]")
    plt.legend()
    plt.show()


def plot_time_constant_stddev(sampling_rate: float) -> None:
    """Plots the standard deviation of the estimated time constant.

    Args:
        sampling_rate: Sampling rate in Hz.
    """
    taus = np.linspace(0.1, 20, 500)
    tau_stddevs = []
    for tau in taus:
        adc_output = _generate_transient_adc_data(tau, sampling_rate)
        adc_data = ExponentialAdcData(adc_output, sampling_rate)
        weighted_linear_regression = adc_data.perform_weighted_linear_regression(
        )
        tau_stddev = np.sqrt(weighted_linear_regression.slope_variance /
                             np.abs(weighted_linear_regression.slope)**4)
        tau_stddevs.append(tau_stddev)

    # Plot the standard deviation of the estimated time constant.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(taus, tau_stddevs, label="Empirical")
    plt.plot(taus,
             np.sqrt(8 * SIGMA**2 * taus /
                     (EXPONENTIAL_SCALING_FACTOR**2 * sampling_rate)),
             label="Approximated")
    ax.set_title("Standard deviation of the time constant")
    ax.set_xlabel("Time constant [s]")
    ax.set_ylabel("Standard deviation [s]")
    plt.legend()
    plt.show()

    # Plot the difference between the actual time constant and the approximated
    # time constant.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(taus,
             np.sqrt(8 * SIGMA**2 * taus /
                     (EXPONENTIAL_SCALING_FACTOR**2 * sampling_rate)) -
             tau_stddevs,
             label="Approximated - actual")
    ax.set_title(
        "Approximation error of the standard deviation of the time constant")
    ax.set_xlabel("Time constant [s]")
    ax.set_ylabel("Difference in the standard deviation [s]")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_log_noise_distribution()
    plot_log_noise_stddev()
    plot_slope_stddev(FLAGS.sampling_rate)
    plot_time_constant_stddev(FLAGS.sampling_rate)


if __name__ == "__main__":
    flags.DEFINE_float("sampling_rate", 100, "Sampling rate in Hz.")

    app.run(main)
