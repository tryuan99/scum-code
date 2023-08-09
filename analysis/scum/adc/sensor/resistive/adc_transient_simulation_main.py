import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging

from analysis.scum.adc.sensor.resistive.adc_data import (
    EXPONENTIAL_SCALING_FACTOR, SIGMA, ExponentialAdcData)
from utils.regression.linear_regression import WeightedLinearRegression

FLAGS = flags.FLAGS

# Offset of the decaying exponential.
EXPONENTIAL_OFFSET = 127

# Number of standard deviations to plot for distributions.
NUM_STDDEVS_TO_PLOT = 5

# Time constants to simulate.
TAUS = np.arange(1, 51) * 0.5

# Number of simulations per time constant.
NUM_SIMULATIONS_PER_TAU = 100


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


def _analyze_weighted_linear_regression(
        weighted_linear_regression: WeightedLinearRegression) -> None:
    """Analyzes the noise of the weighted linear regression.

    Args:
        weighted_linear_regression: Weighted linear regression.
    """
    logging.info("Weighted linear regression:")

    # Plot the distribution of the slope of the weighted linear regression.
    # The slope is 1/time constant.
    slope_mean = np.abs(weighted_linear_regression.slope)
    slope_stddev = np.sqrt(weighted_linear_regression.slope_variance)
    logging.info("Slope: mean = %f, stddev = %f", slope_mean, slope_stddev)

    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.linspace(slope_mean - NUM_STDDEVS_TO_PLOT * slope_stddev,
                    slope_mean + NUM_STDDEVS_TO_PLOT * slope_stddev, 10000)
    slope_pdf = 1 / (np.sqrt(2 * np.pi) * slope_stddev) * np.exp(
        -1 / 2 * (x - slope_mean)**2 / slope_stddev**2)
    plt.plot(x,
             slope_pdf,
             label=f"Mean={slope_mean:.3f}, stddev={slope_stddev:.3f}")
    ax.set_title("Distribution of the slope (1 / time constant)")
    ax.set_xlabel("Slope")
    ax.set_ylabel("Probability density")
    plt.legend()
    plt.show()

    # Plot the distribution of the time constant of the weighted linear regression.
    tau_mean = 1 / slope_mean
    tau_variance = weighted_linear_regression.slope_variance / slope_mean**4
    tau_stddev = np.sqrt(tau_variance)
    logging.info("Time constant: mean = %f, stddev = %f", tau_mean, tau_stddev)

    fig, ax = plt.subplots(figsize=(12, 8))
    x, step = np.linspace(tau_mean - NUM_STDDEVS_TO_PLOT * tau_stddev,
                          tau_mean + NUM_STDDEVS_TO_PLOT * tau_stddev,
                          10000,
                          retstep=True)
    tau_pdf = 1 / (np.sqrt(2 * np.pi) * slope_stddev) * np.exp(
        -1 / 2 * (1 / x - slope_mean)**2 / slope_stddev**2)
    tau_pdf /= np.sum(tau_pdf) * step
    plt.plot(x, tau_pdf, label=f"Mean={tau_mean:.3f}, stddev={tau_stddev:.3f}")
    tau_pdf_approximated = 1 / (np.sqrt(2 * np.pi) * tau_stddev) * np.exp(
        -1 / 2 * (x - tau_mean)**2 / tau_stddev**2)
    plt.plot(
        x,
        tau_pdf_approximated,
        label=f"Mean={tau_mean:.3f}, stddev={tau_stddev:.3f} (approximated)")
    ax.set_title("Distribution of the time constant")
    ax.set_xlabel("Time constant")
    ax.set_ylabel("Probability density")
    plt.legend()
    plt.show()


def plot_single_transient_adc_data(tau: float, sampling_rate: float) -> None:
    """Plots a single transient ADC data.

    Args:
        tau: Time constant in seconds.
        sampling_rate: Sampling rate in Hz.
    """
    adc_output = _generate_transient_adc_data(tau, sampling_rate)
    adc_data = ExponentialAdcData(adc_output, sampling_rate)
    logging.info("Estimated tau: tau = %f", adc_data.estimate_tau())

    # Perform an exponential regression.
    exponential_regression = adc_data.perform_exponential_regression()
    tau_exponential = exponential_regression.time_constant
    logging.info("Exponential regression: tau = %f", tau_exponential)

    # Perform a linear regression in log space.
    linear_regression = adc_data.perform_linear_regression()
    tau_linear = -1 / linear_regression.slope
    logging.info("Linear regression: tau = %f", tau_linear)

    # Perform a weighted linear regression in log space.
    weighted_linear_regression = adc_data.perform_weighted_linear_regression()
    tau_weighted_linear = -1 / weighted_linear_regression.slope
    logging.info("Weighted linear regression: tau = %f", tau_weighted_linear)

    # Perform a polynomial regression.
    polynomial_regression = adc_data.perform_polynomial_regression()
    tau_polynomial = -polynomial_regression.coefficients[
        0] / polynomial_regression.coefficients[1]
    logging.info("Polynomial regression: tau = %f", tau_polynomial)

    # Plot the transient ADC data in linear and log space.
    n = np.arange(len(adc_output))
    t = adc_data.t_axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax1.plot(n, adc_data.samples, label="ADC data")
    ax1.plot(n, exponential_regression.evaluate(t), label="Exponential fit")
    ax1.plot(n,
             np.exp(linear_regression.evaluate(t)) + adc_data.min_adc_output,
             label="Linear fit")
    ax1.plot(n,
             np.exp(weighted_linear_regression.evaluate(t)) +
             adc_data.min_adc_output,
             label="Weighted linear fit")
    ax1.plot(n,
             polynomial_regression.evaluate(t) + adc_data.min_adc_output,
             label="Polynomial fit")
    ax1.set_title("Transient ADC output in linear space")
    ax1.set_ylabel("ADC output [LSB]")
    ax1.legend()

    ax2.plot(n,
             np.log(adc_data.samples - adc_data.min_adc_output),
             label="Log ADC data minus offset")
    ax2.plot(
        n,
        np.log(exponential_regression.evaluate(t) - adc_data.min_adc_output),
        label="Exponential fit")
    ax2.plot(n, linear_regression.evaluate(t), label="Linear fit")
    ax2.plot(n,
             weighted_linear_regression.evaluate(t),
             label="Weighted linear fit")
    ax2.plot(n,
             np.log(polynomial_regression.evaluate(t)),
             label="Polynomial fit")
    ax2.set_title("Transient ADC output in log space minus offset")
    ax2.set_xlabel("ADC sample")
    ax2.set_ylabel("Log ADC output minus offset [bits]")
    ax2.legend()
    plt.show()

    # Analyze the weighted linear regression.
    _analyze_weighted_linear_regression(weighted_linear_regression)


def plot_multiple_transient_adc_data_over_tau(sampling_rate: float) -> None:
    """Plots multiple transient ADC data as a function of the time constant.

    Args:
        sampling_rate: Sampling rate in Hz.
    """
    tau_means_exponential, tau_stddevs_exponential = np.zeros(
        len(TAUS)), np.zeros(len(TAUS))
    tau_means_linear, tau_stddevs_linear = np.zeros(len(TAUS)), np.zeros(
        len(TAUS))
    tau_means_weighted_linear, tau_stddevs_weighted_linear = np.zeros(
        len(TAUS)), np.zeros(len(TAUS))
    tau_means_polynomial, tau_stddevs_polynomial = np.zeros(
        len(TAUS)), np.zeros(len(TAUS))
    slope_means_exponential, slope_stddevs_exponential = np.zeros(
        len(TAUS)), np.zeros(len(TAUS))
    slope_means_linear, slope_stddevs_linear = np.zeros(len(TAUS)), np.zeros(
        len(TAUS))
    slope_means_weighted_linear, slope_stddevs_weighted_linear, slope_stddevs_weighted_linear_exact = np.zeros(
        len(TAUS)), np.zeros(len(TAUS)), np.zeros(len(TAUS))
    slope_means_polynomial, slope_stddevs_polynomial = np.zeros(
        len(TAUS)), np.zeros(len(TAUS))
    for tau_index, tau in enumerate(TAUS):
        tau_estimates_exponential = np.zeros(NUM_SIMULATIONS_PER_TAU)
        tau_estimates_linear = np.zeros(NUM_SIMULATIONS_PER_TAU)
        tau_estimates_weighted_linear = np.zeros(NUM_SIMULATIONS_PER_TAU)
        tau_estimates_polynomial = np.zeros(NUM_SIMULATIONS_PER_TAU)
        slope_estimates_stddevs_weighted_linear = np.zeros(
            NUM_SIMULATIONS_PER_TAU)

        for i in range(NUM_SIMULATIONS_PER_TAU):
            adc_output = _generate_transient_adc_data(tau, sampling_rate)
            adc_data = ExponentialAdcData(adc_output, sampling_rate)

            # Perform an exponential regression.
            exponential_regression = adc_data.perform_exponential_regression()
            tau_exponential = exponential_regression.time_constant
            tau_estimates_exponential[i] = tau_exponential

            # Perform a linear regression in log space.
            linear_regression = adc_data.perform_linear_regression()
            tau_linear = -1 / linear_regression.slope
            tau_estimates_linear[i] = tau_linear

            # Perform a weighted linear regression in log space.
            weighted_linear_regression = adc_data.perform_weighted_linear_regression(
            )
            tau_weighted_linear = -1 / weighted_linear_regression.slope
            tau_estimates_weighted_linear[i] = tau_weighted_linear
            slope_estimates_stddevs_weighted_linear[i] = np.sqrt(
                weighted_linear_regression.slope_variance)

            # Perform a polynomial regression.
            polynomial_regression = adc_data.perform_polynomial_regression()
            tau_polynomial = -polynomial_regression.coefficients[
                0] / polynomial_regression.coefficients[1]
            tau_estimates_polynomial[i] = tau_polynomial

        tau_means_exponential[tau_index] = np.mean(tau_estimates_exponential)
        tau_stddevs_exponential[tau_index] = np.std(tau_estimates_exponential)
        tau_means_linear[tau_index] = np.mean(tau_estimates_linear)
        tau_stddevs_linear[tau_index] = np.std(tau_estimates_linear)
        tau_means_weighted_linear[tau_index] = np.mean(
            tau_estimates_weighted_linear)
        tau_stddevs_weighted_linear[tau_index] = np.std(
            tau_estimates_weighted_linear)
        slope_stddevs_weighted_linear_exact[tau_index] = np.mean(
            slope_estimates_stddevs_weighted_linear)
        tau_means_polynomial[tau_index] = np.mean(tau_estimates_polynomial)
        tau_stddevs_polynomial[tau_index] = np.std(tau_estimates_polynomial)
        slope_means_exponential[tau_index] = np.mean(1 /
                                                     tau_estimates_exponential)
        slope_stddevs_exponential[tau_index] = np.std(1 /
                                                      tau_estimates_exponential)
        slope_means_linear[tau_index] = np.mean(1 / tau_estimates_linear)
        slope_stddevs_linear[tau_index] = np.std(1 / tau_estimates_linear)
        slope_means_weighted_linear[tau_index] = np.mean(
            1 / tau_estimates_weighted_linear)
        slope_stddevs_weighted_linear[tau_index] = np.std(
            1 / tau_estimates_weighted_linear)
        slope_means_polynomial[tau_index] = np.mean(1 /
                                                    tau_estimates_polynomial)
        slope_stddevs_polynomial[tau_index] = np.std(1 /
                                                     tau_estimates_polynomial)

    # Plot the mean error and standard deviation of the estimated slope.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax1.plot(1 / TAUS,
             slope_means_exponential - 1 / TAUS,
             label="Exponential fit")
    ax1.plot(1 / TAUS, slope_means_linear - 1 / TAUS, label="Linear fit")
    ax1.plot(1 / TAUS,
             slope_means_weighted_linear - 1 / TAUS,
             label="Weighted linear fit")
    ax1.plot(1 / TAUS,
             slope_means_polynomial - 1 / TAUS,
             label="Polynomial fit")
    ax1.set_title("Mean error of the estimated slope")
    ax1.set_ylabel("Mean error [1/s]")
    ax1.legend()

    ax2.plot(1 / TAUS, slope_stddevs_exponential, label="Exponential fit")
    ax2.plot(1 / TAUS, slope_stddevs_linear, label="Linear fit")
    ax2.plot(1 / TAUS,
             slope_stddevs_weighted_linear,
             label="Weighted linear fit")
    ax2.plot(1 / TAUS, slope_stddevs_polynomial, label="Polynomial fit")
    ax2.set_title("Standard deviation of the estimated slope")
    ax2.set_xlabel("Slope [1/s]")
    ax2.set_ylabel("Standard deviation [1/s]")
    ax2.legend()
    plt.show()

    # Compare the standard deviation of the estimated slope with a weighted
    # linear regression with its approximation.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(1 / TAUS, slope_stddevs_weighted_linear, label="Simulated")
    plt.plot(1 / TAUS,
             np.sqrt(8 * SIGMA**2 /
                     (EXPONENTIAL_SCALING_FACTOR**2 * sampling_rate * TAUS**3)),
             label="Theoretical (approximated)")
    plt.plot(1 / TAUS,
             slope_stddevs_weighted_linear_exact,
             label="Thereotical (exact)")
    ax.set_title(
        "Standard deviation of the estimated slope with a weighted linear regression"
    )
    ax.set_xlabel("Slope [1/s]")
    ax.set_ylabel("Standard deviation [1/s]")
    ax.legend()
    plt.show()

    # Plot the mean error and standard deviation of the estimated time constant.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax1.plot(TAUS, tau_means_exponential - TAUS, label="Exponential fit")
    ax1.plot(TAUS, tau_means_linear - TAUS, label="Linear fit")
    ax1.plot(TAUS,
             tau_means_weighted_linear - TAUS,
             label="Weighted linear fit")
    ax1.plot(TAUS, tau_means_polynomial - TAUS, label="Polynomial fit")
    ax1.set_title("Mean error of the estimated time constant")
    ax1.set_ylabel("Mean error [s]")
    ax1.legend()

    ax2.plot(TAUS, tau_stddevs_exponential, label="Exponential fit")
    ax2.plot(TAUS, tau_stddevs_linear, label="Linear fit")
    ax2.plot(TAUS, tau_stddevs_weighted_linear, label="Weighted linear fit")
    ax2.plot(TAUS, tau_stddevs_polynomial, label="Polynomial fit")
    ax2.set_title("Standard deviation of the estimated time constant")
    ax2.set_xlabel("tau [s]")
    ax2.set_ylabel("Standard deviation [s]")
    ax2.legend()
    plt.show()

    # Compare the standard deviation of the estimated time constant with a
    # weighted linear regression with its approximation.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(TAUS, tau_stddevs_weighted_linear, label="Simulated")
    plt.plot(TAUS,
             np.sqrt(8 * SIGMA**2 * TAUS /
                     (EXPONENTIAL_SCALING_FACTOR**2 * sampling_rate)),
             label="Theoretical (approximated)")
    ax.set_title(
        "Standard deviation of the estimated time constant with a weighted linear regression"
    )
    ax.set_xlabel("tau [s]")
    ax.set_ylabel("Standard deviation [s]")
    ax.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_single_transient_adc_data(FLAGS.tau, FLAGS.sampling_rate)
    plot_multiple_transient_adc_data_over_tau(FLAGS.sampling_rate)


if __name__ == "__main__":
    flags.DEFINE_float("tau", 2, "Time constant in seconds.")
    flags.DEFINE_float("sampling_rate", 100, "Sampling rate in Hz.")

    app.run(main)
