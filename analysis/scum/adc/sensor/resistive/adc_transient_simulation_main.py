import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging
from matplotlib.ticker import FuncFormatter

from analysis.scum.adc.sensor.resistive.adc_data import (
    EXPONENTIAL_SCALING_FACTOR, SIGMA, ExponentialAdcData)
from utils.regression.linear_regression import (LinearRegression,
                                                WeightedLinearRegression)

FLAGS = flags.FLAGS

# Number of taus for which to generate ADC samples.
NUM_TAUS = 7

# Offset of the decaying exponential.
EXPONENTIAL_OFFSET = 127

# Number of standard deviations to plot for distributions.
NUM_STDDEVS_TO_PLOT = 5

# Number of samples for which to calculate the residuals.
NUM_SAMPLES_PER_RESIDUALS = 5

# Number of simulations for the PDF.
NUM_SIMULATIONS_FOR_PDF = 50000

# Step size when simulating the PDF.
PDF_STEP = 0.0001

# Maximum absolute value of the PDF sample.
MAX_ABSOLUTE_PDF_SAMPLE = 10

# Time constants to simulate.
TAUS = np.arange(0.5, 10.5, 0.5)

# Number of simulations per time constant.
NUM_SIMULATIONS_PER_TAU = 100


def _generate_transient_adc_data(tau: float,
                                 sampling_rate: float,
                                 num_taus: float = NUM_TAUS) -> np.ndarray:
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
    ax.plot(x,
            slope_pdf,
            label=f"Mean={slope_mean:.3f}, stddev={slope_stddev:.3f}")
    ax.set_title("Distribution of the slope (1 / time constant)")
    ax.set_xlabel("Slope")
    ax.set_ylabel("Probability density")
    ax.legend()
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
    ax.plot(x, tau_pdf, label=f"Mean={tau_mean:.3f}, stddev={tau_stddev:.3f}")
    tau_pdf_approximated = 1 / (np.sqrt(2 * np.pi) * tau_stddev) * np.exp(
        -1 / 2 * (x - tau_mean)**2 / tau_stddev**2)
    ax.plot(
        x,
        tau_pdf_approximated,
        label=f"Mean={tau_mean:.3f}, stddev={tau_stddev:.3f} (approximated)")
    ax.set_title("Distribution of the time constant")
    ax.set_xlabel("Time constant")
    ax.set_ylabel("Probability density")
    ax.legend()
    plt.show()


# Bin axis formatter.
def _bin_axis_formatter(x: float, position: float) -> str:
    """Formats the bin tick labels.

    Args:
        x: Tick value.
        position: Position.

    Returns:
        The string corresponding to the bin.
    """
    return (f"[{int(x) * NUM_SAMPLES_PER_RESIDUALS}, "
            f"{int(x + 1)*NUM_SAMPLES_PER_RESIDUALS})")


def _compare_linear_regressions_sse(
        linear_regression: LinearRegression,
        weighted_linear_regression: WeightedLinearRegression,
        adc_data: ExponentialAdcData) -> None:
    """Compares the sum of squared errors between the linear regression and the
    weighted linear regression.

    Args:
        linear_regression: Linear regression.
        weighted_linear_regression: Weighted linear regression.
        adc_data: ADC samples.
    """
    t = adc_data.t_axis
    y = adc_data.samples
    y_predicted_linear = np.exp(
        linear_regression.evaluate(t)) + adc_data.min_adc_output
    y_predicted_weighted_linear = np.exp(
        weighted_linear_regression.evaluate(t)) + adc_data.min_adc_output
    residuals_linear = np.linalg.norm(y - y_predicted_linear)**2
    residuals_weighted_linear = np.linalg.norm(y -
                                               y_predicted_weighted_linear)**2
    logging.info("Residuals: linear: %f, weighted linear: %f", residuals_linear,
                 residuals_weighted_linear)

    # Calculate the binned residuals.
    bin_indices = np.arange(len(y) // NUM_SAMPLES_PER_RESIDUALS)
    residuals_bins_linear = np.zeros(len(bin_indices))
    residuals_bins_weighted_linear = np.zeros(len(bin_indices))
    for bin_index in bin_indices:
        residuals_bin_linear = np.linalg.norm(
            y[bin_index * NUM_SAMPLES_PER_RESIDUALS:(bin_index + 1) *
              NUM_SAMPLES_PER_RESIDUALS] -
            y_predicted_linear[bin_index * NUM_SAMPLES_PER_RESIDUALS:
                               (bin_index + 1) * NUM_SAMPLES_PER_RESIDUALS])**2
        residuals_bin_weighted_linear = np.linalg.norm(
            y[bin_index * NUM_SAMPLES_PER_RESIDUALS:(bin_index + 1) *
              NUM_SAMPLES_PER_RESIDUALS] -
            y_predicted_weighted_linear[bin_index * NUM_SAMPLES_PER_RESIDUALS:
                                        (bin_index + 1) *
                                        NUM_SAMPLES_PER_RESIDUALS])**2
        residuals_bins_linear[bin_index] = residuals_bin_linear
        residuals_bins_weighted_linear[
            bin_index] = residuals_bin_weighted_linear

    # Plot the binned residuals.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(bin_indices,
            residuals_bins_linear,
            label="Linear regression residuals")
    ax.plot(bin_indices,
            residuals_bins_weighted_linear,
            label="Weighted linear regression residuals")
    ax.set_title("Linear regression residuals of the binned ADC samples")
    ax.set_xlabel("ADC sample bins")
    ax.set_ylabel("Residuals")
    ax.xaxis.set_major_formatter(FuncFormatter(_bin_axis_formatter))
    ax.legend()
    plt.show()

    # Plot the difference between the binned residuals.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        bin_indices,
        residuals_bins_linear - residuals_bins_weighted_linear,
        label=
        "Linear regression residuals - weighted linear regression residuals")
    plt.axhline(y=0, linestyle="--", c="C1")
    ax.set_title(
        "Difference in linear regression residuals of the binned ADC samples")
    ax.set_xlabel("ADC sample bins")
    ax.set_ylabel("Difference in residuals")
    ax.xaxis.set_major_formatter(FuncFormatter(_bin_axis_formatter))
    ax.legend()
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
    r_squared_exponential = exponential_regression.r_squared
    logging.info("Exponential regression: tau = %f, r^2 = %f", tau_exponential,
                 r_squared_exponential)

    # Perform a linear regression in log space.
    linear_regression = adc_data.perform_linear_regression()
    tau_linear = -1 / linear_regression.slope
    r_squared_linear = linear_regression.r_squared
    logging.info("Linear regression: tau = %f, r^2 = %f", tau_linear,
                 r_squared_linear)

    # Perform a weighted linear regression in log space.
    weighted_linear_regression = adc_data.perform_weighted_linear_regression()
    tau_weighted_linear = -1 / weighted_linear_regression.slope
    r_squared_weighted_linear = weighted_linear_regression.r_squared
    logging.info("Weighted linear regression: tau = %f, r^2 = %f",
                 tau_weighted_linear, r_squared_weighted_linear)

    # Perform a polynomial regression.
    polynomial_regression = adc_data.perform_polynomial_regression()
    tau_polynomial = -polynomial_regression.coefficients[
        0] / polynomial_regression.coefficients[1]
    r_squared_polynomial = polynomial_regression.r_squared
    logging.info("Polynomial regression: tau = %f, r^2 = %f", tau_polynomial,
                 r_squared_polynomial)

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

    # Compare the sum of squared errors between the linear regression and the
    # weighted linear regression.
    _compare_linear_regressions_sse(linear_regression,
                                    weighted_linear_regression, adc_data)


def plot_transient_adc_data_distribution(tau: float,
                                         sampling_rate: float) -> None:
    """Plots the distribution of the ADC data samples

    Args:
        tau: Time constant in seconds.
        sampling_rate: Sampling rate in Hz.
    """
    adc_output_length = int(NUM_TAUS * sampling_rate * tau)
    adc_output = np.zeros((NUM_SIMULATIONS_FOR_PDF, adc_output_length))
    log_adc_output = np.zeros((NUM_SIMULATIONS_FOR_PDF, adc_output_length))
    for i in range(NUM_SIMULATIONS_FOR_PDF):
        adc_output[i] = _generate_transient_adc_data(tau, sampling_rate)
        adc_data = ExponentialAdcData(adc_output[i], sampling_rate)
        log_adc_output[i] = adc_data.get_linear_regression_samples()
    t_axis = ExponentialAdcData(
        _generate_transient_adc_data(tau, sampling_rate), sampling_rate).t_axis
    tau_axis = t_axis / tau
    tau_axis_indices = tau_axis < 3

    # Plot the standard deviation of the ADC data.
    adc_output_stddev = np.std(adc_output, axis=0)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(tau_axis[tau_axis_indices], adc_output_stddev[tau_axis_indices])
    ax.set_title("Standard deviation of the ADC samples")
    ax.set_xlabel("Time [tau]")
    ax.set_ylabel("Standard deviation [LSB]")
    plt.show()

    # Calculate the the theoretical standard deviation of the log ADC data.
    regression_length = int(3 * sampling_rate * tau)
    log_adc_output_stddev_theoretical = np.zeros(regression_length)
    for tau_index, tau in enumerate(tau_axis[tau_axis_indices]):
        x = np.arange(-MAX_ABSOLUTE_PDF_SAMPLE, MAX_ABSOLUTE_PDF_SAMPLE,
                      PDF_STEP)
        pdf = 1 / (np.sqrt(2 * np.pi) * SIGMA) * np.exp(
            -(EXPONENTIAL_SCALING_FACTOR**2 * np.exp(-2 * tau) *
              (np.exp(x) - 1)**2) /
            (2 * SIGMA**2)) * EXPONENTIAL_SCALING_FACTOR * np.exp(x - tau)
        mean_pdf = np.sum(x * pdf * PDF_STEP)
        mean_pdf_squared = np.sum(x**2 * pdf * PDF_STEP)
        variance_pdf = mean_pdf_squared - mean_pdf**2
        log_adc_output_stddev_theoretical[tau_index] = np.sqrt(variance_pdf)

    # Plot the standard deviation of the log ADC data and its differences.
    log_adc_output_stddev = np.std(log_adc_output, axis=0)
    log_adc_output_stddev_approximated = SIGMA / EXPONENTIAL_SCALING_FACTOR * np.exp(
        tau_axis[tau_axis_indices])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharex=True)
    ax1.plot(tau_axis[tau_axis_indices],
             log_adc_output_stddev[tau_axis_indices],
             label="Simulated")
    ax1.plot(tau_axis[tau_axis_indices],
             log_adc_output_stddev_theoretical,
             label="Theoretical")
    ax1.plot(tau_axis[tau_axis_indices],
             log_adc_output_stddev_approximated,
             label="Approximated")
    ax1.set_title("Standard deviation of the log ADC samples")
    ax1.set_xlabel("Time [tau]")
    ax1.set_ylabel("Standard deviation")
    ax1.legend()
    ax2.plot(tau_axis[tau_axis_indices],
             np.abs(log_adc_output_stddev[tau_axis_indices] -
                    log_adc_output_stddev_theoretical),
             label="Simulated - theoretical")
    ax2.plot(tau_axis[tau_axis_indices],
             np.abs(log_adc_output_stddev_approximated -
                    log_adc_output_stddev_theoretical),
             label="Approximated - theoretical")
    ax2.set_title(
        "Absolute difference in standard deviation of the log ADC samples")
    ax2.set_xlabel("Time [tau]")
    ax2.set_ylabel("Difference in standard deviation")
    ax2.legend()
    plt.show()


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
    slope_means_weighted_linear, slope_stddevs_weighted_linear, slope_stddevs_weighted_linear_theoretical = np.zeros(
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
        slope_stddevs_weighted_linear_theoretical[tau_index] = np.mean(
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
    ax.plot(1 / TAUS, slope_stddevs_weighted_linear, label="Simulated")
    ax.plot(1 / TAUS,
            np.sqrt(8 * SIGMA**2 /
                    (EXPONENTIAL_SCALING_FACTOR**2 * sampling_rate * TAUS**3)),
            label="Approximated")
    ax.plot(1 / TAUS,
            slope_stddevs_weighted_linear_theoretical,
            label="Theoretical")
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
    ax.plot(TAUS, tau_stddevs_weighted_linear, label="Simulated")
    ax.plot(TAUS,
            np.sqrt(8 * SIGMA**2 * TAUS /
                    (EXPONENTIAL_SCALING_FACTOR**2 * sampling_rate)),
            label="Approximated")
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
    plot_transient_adc_data_distribution(FLAGS.tau, FLAGS.sampling_rate)
    plot_multiple_transient_adc_data_over_tau(FLAGS.sampling_rate)


if __name__ == "__main__":
    flags.DEFINE_float("tau", 2, "Time constant in seconds.")
    flags.DEFINE_float("sampling_rate", 100, "Sampling rate in Hz.")

    app.run(main)
