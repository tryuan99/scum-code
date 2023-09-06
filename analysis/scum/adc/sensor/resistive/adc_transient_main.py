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
    plt.rcParams.update({"font.size": 16})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    select = adc_data.t_axis < 2
    ax1.plot(adc_data.t_axis[select], adc_data.samples[select])
    before = adc_data.samples[select].copy()
    adc_data.disambiguate_msb_9()
    ax2.plot(adc_data.t_axis[select], adc_data.samples[select])
    after = adc_data.samples[select]
    for i in range(len(adc_data.t_axis[select])):
        print(f"{adc_data.t_axis[select][i]},{before[i]},{after[i]}")
    ax1.set_title("Raw ADC data")
    ax2.set_title("ADC data with MSB corrected")
    ax1.set_xlabel("Time [s]")
    ax2.set_xlabel("Time [s]")
    ax1.set_ylabel("ADC output [LSB]")
    ax2.set_ylabel("ADC output [LSB]")
    plt.show()
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

    iterations = df.groupby(iteration_column)
    tau_exponential = np.zeros(len(iterations))
    tau_linear = np.zeros(len(iterations))
    tau_weighted_linear = np.zeros(len(iterations))
    tau_polynomial = np.zeros(len(iterations))
    r_squared_exponential = np.zeros(len(iterations))
    r_squared_linear = np.zeros(len(iterations))
    r_squared_weighted_linear = np.zeros(len(iterations))
    r_squared_polynomial = np.zeros(len(iterations))
    fig, ax = plt.subplots(figsize=(12, 8))
    for iteration_index, (_, group) in enumerate(iterations):
        adc_data = ExponentialAdcData(group[adc_output_column], sampling_rate)
        adc_data.disambiguate_msb_9()

        # Plot the ADC data.
        plt.plot(group.reset_index().index, adc_data.samples)

        # Estimate tau using an exponential regression, a linear regression in
        # log space, and a weighted linear regression in log space.
        exponential_regression = adc_data.perform_exponential_regression()
        tau_exponential[iteration_index] = exponential_regression.time_constant
        r_squared_exponential[
            iteration_index] = exponential_regression.r_squared
        linear_regression = adc_data.perform_linear_regression()
        tau_linear[iteration_index] = -1 / linear_regression.slope
        r_squared_linear = linear_regression.r_squared
        weighted_linear_regression = adc_data.perform_weighted_linear_regression(
        )
        tau_weighted_linear[
            iteration_index] = -1 / weighted_linear_regression.slope
        r_squared_weighted_linear = weighted_linear_regression.r_squared
        polynomial_regression = adc_data.perform_polynomial_regression()
        tau_polynomial[iteration_index] = -polynomial_regression.coefficients[
            0] / polynomial_regression.coefficients[1]
        r_squared_polynomial[iteration_index] = polynomial_regression.r_squared
    ax.set_title("Transient ADC output")
    ax.set_xlabel("ADC samples")
    ax.set_ylabel("ADC output [LSB]")
    plt.show()

    # Calculate the mean and standard deviation of the estimated taus.
    logging.info("Num iterations: %d", len(iterations))
    logging.info(
        "Exponential regression: mean tau = %f, stddev tau = %f, mean r^2 = %f",
        np.mean(tau_exponential), np.std(tau_exponential),
        np.mean(r_squared_exponential))
    logging.info(
        "Linear regression: mean tau = %f, stddev tau = %f, mean r^2 = %f",
        np.mean(tau_linear), np.std(tau_linear), np.mean(r_squared_linear))
    logging.info(
        "Weighted linear regression: mean tau = %f, stddev tau = %f, mean r^2 = %f",
        np.mean(tau_weighted_linear), np.std(tau_weighted_linear),
        np.mean(r_squared_weighted_linear))
    logging.info(
        "Polynomial regression: mean tau = %f, stddev tau = %f, mean r^2 = %f",
        np.mean(tau_polynomial), np.std(tau_polynomial),
        np.mean(r_squared_polynomial))


def main(argv):
    assert len(argv) == 1

    # Compare regression types.
    plt.rcParams.update({"font.size": 16})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    nominal_taus = np.array(
        [0.020916, 0.0462, 0.110, 0.234, 0.404, 0.5, 1.058, 2.09, 4.599])
    ax1.loglog(nominal_taus,
               np.abs(
                   np.array([
                       0.021575, 0.045671, 0.111, 0.233, 0.407, 0.499, 1.01,
                       2.099187, 4.549780
                   ]) - nominal_taus),
               label="Exponential",
               linestyle="--",
               marker="^")
    ax1.loglog(nominal_taus,
               np.abs(
                   np.array([
                       0.022884, 0.049767, 0.117, 0.245, 0.417, 0.516, 1.03,
                       2.188900, 4.409927
                   ]) - nominal_taus),
               label="Linear",
               linestyle=":",
               marker="v")
    ax1.loglog(nominal_taus,
               np.abs(
                   np.array([
                       0.022236, 0.048159, 0.115, 0.241, 0.414, 0.512, 1.03,
                       2.167384, 4.507946
                   ]) - nominal_taus),
               label="Weighted linear",
               marker="o")
    ax1.loglog(nominal_taus,
               np.abs(
                   np.array([
                       0.022240, 0.043677, 0.104, 0.199, 0.331, 0.409, 0.826,
                       1.802195, 3.939816
                   ]) - nominal_taus),
               label="Polynomial",
               linestyle="-.",
               marker="s")
    ax2.loglog(nominal_taus,
               np.array([
                   0.000604, 0.000472, 0.00189, 0.00202, 0.00301, 0.002800,
                   0.00955, 0.024469, 0.031757
               ]),
               label="Exponential",
               linestyle="--",
               marker="^")
    ax2.loglog(nominal_taus,
               np.array([
                   0.000647, 0.000845, 0.00163, 0.00230, 0.00317, 0.00476,
                   0.0133, 0.049733, 0.073143
               ]),
               label="Linear",
               linestyle="--",
               marker="v")
    ax2.loglog(nominal_taus,
               np.array([
                   0.000184, 0.000413, 0.00098, 0.00161, 0.00178, 0.00301,
                   0.0103, 0.037237, 0.048494
               ]),
               label="Weighted linear",
               marker="o")
    ax2.loglog(nominal_taus,
               np.array([
                   0.000804, 0.001264, 0.00735, 0.00498, 0.00399, 0.00324,
                   0.0136, 0.023325, 0.032916
               ]),
               label="Polynomial",
               linestyle="-.",
               marker="s")
    # ax2.loglog(nominal_taus, np.sqrt(8 * 5**2 * nominal_taus / 870**2 / 100))
    print(
        "tau [s],Exponential mean,Exponential std,Linear mean,Linear std,Weighted linear mean,Weighted linear std,Polynomial mean,Polynomial std"
    )
    for i in range(len(nominal_taus)):
        print(f"""{nominal_taus[i]},{np.abs(
                   np.array([
                       0.021575, 0.045671, 0.111, 0.233, 0.407, 0.499, 1.01,
                       2.099187, 4.549780
                   ]) - nominal_taus)[i]},{np.array([
                   0.000604, 0.000472, 0.00189, 0.00202, 0.00301, 0.002800,
                   0.00955, 0.024469, 0.031757
               ])[i]},{np.abs(
                   np.array([
                       0.022884, 0.049767, 0.117, 0.245, 0.417, 0.516, 1.03,
                       2.188900, 4.409927
                   ]) - nominal_taus)[i]},{np.array([
                   0.000647, 0.000845, 0.00163, 0.00230, 0.00317, 0.00476,
                   0.0133, 0.049733, 0.073143
               ])[i]},{np.abs(
                   np.array([
                       0.022236, 0.048159, 0.115, 0.241, 0.414, 0.512, 1.03,
                       2.167384, 4.507946
                   ]) - nominal_taus)[i]},{np.array([
                   0.000184, 0.000413, 0.00098, 0.00161, 0.00178, 0.00301,
                   0.0103, 0.037237, 0.048494
               ])[i]},{np.abs(
                   np.array([
                       0.022240, 0.043677, 0.104, 0.199, 0.331, 0.409, 0.826,
                       1.802195, 3.939816
                   ]) - nominal_taus)[i]},{np.array([
                   0.000804, 0.001264, 0.00735, 0.00498, 0.00399, 0.00324,
                   0.0136, 0.023325, 0.032916
               ])[i]}""")
    ax1.set_title("Mean absolute error of the estimated time constant")
    ax1.set_xlabel("tau [s]")
    ax1.set_ylabel("Mean absolute error [s]")
    ax1.legend()
    ax2.set_title("Standard error of the estimated time constant")
    ax2.set_xlabel("tau [s]")
    ax2.set_ylabel("Standard error [s]")
    ax2.legend()
    plt.show()

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
