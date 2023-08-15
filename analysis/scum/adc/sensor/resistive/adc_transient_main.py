import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

from analysis.scum.adc.sensor.resistive.adc_data import ExponentialAdcData

FLAGS = flags.FLAGS


def plot_transient_adc_data(data: str, sampling_rate: float,
                            capacitance: float) -> None:
    """Plots the transient ADC data.

    Args:
        data: Data filename.
        sampling_rate: Sampling rate in Hz.
        capacitance: Fixed capacitance in F.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    (adc_output_column,) = df.columns
    logging.info(df.describe())

    adc_output = df[adc_output_column]
    adc_data = ExponentialAdcData(adc_output, sampling_rate)
    adc_data.disambiguate_msb_9()
    logging.info("Estimated tau: tau = %f", adc_data.estimate_tau())

    # Perform an exponential regression.
    exponential_regression = adc_data.perform_exponential_regression()
    tau_exponential = exponential_regression.time_constant
    r_squared_exponential = exponential_regression.r_squared
    logging.info("Exponential regression:")
    logging.info("tau = %f, r^2 = %f", tau_exponential, r_squared_exponential)
    logging.info("C = %g, R = %g", capacitance, tau_exponential / capacitance)

    # Perform a linear regression in log space.
    linear_regression = adc_data.perform_linear_regression()
    tau_linear = -1 / linear_regression.slope
    r_squared_linear = linear_regression.r_squared
    logging.info("Linear regression:")
    logging.info("tau = %f, r^2 = %f", tau_linear, r_squared_linear)
    logging.info("C = %g, R = %g", capacitance, tau_linear / capacitance)

    # Perform a weighted linear regression in log space.
    weighted_linear_regression = adc_data.perform_weighted_linear_regression()
    tau_weighted_linear = -1 / weighted_linear_regression.slope
    r_squared_weighted_linear = weighted_linear_regression.r_squared
    logging.info("Weighted linear regression:")
    logging.info("tau = %f, r^2 = %f", tau_weighted_linear,
                 r_squared_weighted_linear)
    logging.info("C = %g, R = %g", capacitance,
                 tau_weighted_linear / capacitance)

    # Perform a polynomial regression.
    polynomial_regression = adc_data.perform_polynomial_regression()
    tau_polynomial = -polynomial_regression.coefficients[
        0] / polynomial_regression.coefficients[1]
    r_squared_polynomial = polynomial_regression.r_squared
    logging.info("Polynomial regression:")
    logging.info("tau = %f, r^2 = %f", tau_polynomial, r_squared_polynomial)
    logging.info("C = %g, R = %g", capacitance, tau_polynomial / capacitance)

    # Plot the transient ADC data in linear and log space.
    t = adc_data.t_axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    ax1.plot(adc_output.index, adc_data.samples, label="ADC data")
    ax1.plot(adc_output.index,
             exponential_regression.evaluate(t),
             label="Exponential fit")
    ax1.plot(adc_output.index,
             np.exp(linear_regression.evaluate(t)) + adc_data.min_adc_output,
             label="Linear fit")
    ax1.plot(adc_output.index,
             np.exp(weighted_linear_regression.evaluate(t)) +
             adc_data.min_adc_output,
             label="Weighted linear fit")
    ax1.plot(adc_output.index,
             polynomial_regression.evaluate(t) + adc_data.min_adc_output,
             label="Polynomial fit")
    ax1.set_title("Transient ADC output in linear space")
    ax1.set_ylabel("ADC output [LSB]")
    ax1.legend()

    ax2.plot(adc_output.index,
             np.log(adc_data.samples - adc_data.min_adc_output),
             label="Log ADC data minus offset")
    ax2.plot(
        adc_output.index,
        np.log(exponential_regression.evaluate(t) - adc_data.min_adc_output),
        label="Exponential fit")
    ax2.plot(adc_output.index,
             linear_regression.evaluate(t),
             label="Linear fit")
    ax2.plot(adc_output.index,
             weighted_linear_regression.evaluate(t),
             label="Weighted linear fit")
    ax2.plot(adc_output.index,
             np.log(polynomial_regression.evaluate(t)),
             label="Polynomial fit")
    ax2.set_title("Transient ADC output in log space minus offset")
    ax2.set_xlabel("ADC sample")
    ax2.set_ylabel("Log ADC output minus offset [bits]")
    ax2.legend()
    plt.show()


def plot_multiple_transient_adc_data(data: str, sampling_rate: float,
                                     capacitance: float) -> None:
    """Plots multiple transient ADC data.

    Args:
        data: Data filename.
        sampling_rate: Sampling rate in Hz.
        capacitance: Fixed capacitance in F.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    (
        iteration_column,
        adc_output_column,
    ) = df.columns
    logging.info(df.describe())

    tau_exponential = []
    tau_linear = []
    tau_weighted_linear = []
    fig, ax = plt.subplots(figsize=(12, 8))
    iterations = df.groupby(iteration_column)
    for _, group in iterations:
        adc_data = ExponentialAdcData(group[adc_output_column], sampling_rate)
        adc_data.disambiguate_msb_9()

        # Plot the ADC data.
        plt.plot(group.reset_index().index, adc_data.samples)

        # Estimate tau using an exponential regression, a linear regression in
        # log space, and a weighted linear regression in log space.
        exponential_regression = adc_data.perform_exponential_regression()
        tau_exponential.append(exponential_regression.time_constant)
        linear_regression = adc_data.perform_linear_regression()
        tau_linear.append(-1 / linear_regression.slope)
        weighted_linear_regression = adc_data.perform_weighted_linear_regression(
        )
        tau_weighted_linear.append(-1 / weighted_linear_regression.slope)
    ax.set_title("Transient ADC output")
    ax.set_xlabel("ADC samples")
    ax.set_ylabel("ADC output [LSB]")
    plt.show()

    # Calculate the mean and standard deviation of the estimated taus.
    logging.info("Num iterations: %d", len(iterations))
    logging.info("Exponential regression: mean tau = %f, stddev = %f",
                 np.mean(tau_exponential), np.std(tau_exponential))
    logging.info("Linear regression: mean tau = %f, stddev = %f",
                 np.mean(tau_linear), np.std(tau_linear))
    logging.info("Weighted linear regression: mean tau = %f, stddev = %f",
                 np.mean(tau_weighted_linear), np.std(tau_weighted_linear))


def main(argv):
    assert len(argv) == 1
    if FLAGS.multiple:
        plot_multiple_transient_adc_data(FLAGS.data, FLAGS.sampling_rate,
                                         FLAGS.capacitance)
    else:
        plot_transient_adc_data(FLAGS.data, FLAGS.sampling_rate,
                                FLAGS.capacitance)


if __name__ == "__main__":
    flags.DEFINE_string(
        "data",
        "analysis/scum/adc/sensor/resistive/data/adc_data_thermistor_3a_3.csv",
        "Data filename.")
    flags.DEFINE_float("sampling_rate", 100, "Sampling rate in Hz.")
    flags.DEFINE_float("capacitance", 50e-9, "Fixed capacitance in F.")
    flags.DEFINE_bool("multiple", False,
                      "If true, process multiple transients simultaneously.")

    app.run(main)
