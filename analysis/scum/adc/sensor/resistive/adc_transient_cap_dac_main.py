import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from absl import app, flags, logging

FLAGS = flags.FLAGS

# Actual time constant columm name.
ACTUAL_TIME_CONSTANT_COLUMN = "Actual time constant [s]"

# Estimated time constant columm name.
ESTIMATED_TIME_CONSTANT_COLUMN = "Estimated time constant [s]"


def analyze_cap_dac_data(data: str, sampling_rate: float) -> None:
    """Analyzes the capacitive DAC data.

    Args:
        data: Data filename.
        sampling_rate: Sampling rate in Hz.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    (
        resistance_column,
        capacitance_column,
        num_adc_samples_column,
        three_tau_index_column,
        time_constant_column,
        time_constant_scaling_factor_column,
    ) = df.columns
    df[ACTUAL_TIME_CONSTANT_COLUMN] = df[resistance_column] * df[
        capacitance_column]
    df[ESTIMATED_TIME_CONSTANT_COLUMN] = (
        df[time_constant_column] / df[time_constant_scaling_factor_column])
    logging.info(df.describe())

    time_constants = df.groupby(ACTUAL_TIME_CONSTANT_COLUMN)
    logging.info(time_constants.describe())

    # Plot the mean estimated time constant with error bars.
    fig, ax = plt.subplots(figsize=(12, 8))
    time_constants.mean().plot.line(
        y=ESTIMATED_TIME_CONSTANT_COLUMN,
        yerr=time_constants.std()[ESTIMATED_TIME_CONSTANT_COLUMN],
        ax=ax,
        label="Estimated")
    actual_time_constants = df[ACTUAL_TIME_CONSTANT_COLUMN].unique()
    ax.plot(actual_time_constants, actual_time_constants, "--", label="Actual")
    ax.set_title("Estimated vs. actual time constant")
    ax.set_ylabel("Estimated time constant [s]")
    plt.legend()
    plt.show()

    # Plot the error of the mean estimated time constant.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(time_constants.mean()[ESTIMATED_TIME_CONSTANT_COLUMN] -
            time_constants.mean().index,
            label="Estimated - actual")
    ax.set_title("Error of the mean estimated time constant")
    ax.set_xlabel("Actual time constant [s]")
    ax.set_ylabel("Difference in time constant [s]")
    plt.legend()
    plt.show()


def run_cap_dac_sanity_check(data: str) -> None:
    """Analyzes the capacitive DAC data.

    Args:
        data: Data filename.
        sampling_rate: Sampling rate in Hz.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    (
        resistance_column,
        capacitance_column,
        num_adc_samples_column,
        three_tau_index_column,
        time_constant_column,
        time_constant_scaling_factor_column,
    ) = df.columns
    df[ESTIMATED_TIME_CONSTANT_COLUMN] = (
        df[time_constant_column] / df[time_constant_scaling_factor_column])
    logging.info(df.describe())

    # The two lowest capacitor values should add to the capacitance of the
    # largest capacitor.
    capacitances = sorted(df[capacitance_column].unique())
    expected_time_constant_sum = [
        sum(estimated_time_constant)
        for estimated_time_constant in itertools.product(
            df[df[capacitance_column] ==
               capacitances[0]][ESTIMATED_TIME_CONSTANT_COLUMN], df[
                   df[capacitance_column] == capacitances[1]]
            [ESTIMATED_TIME_CONSTANT_COLUMN])
    ]
    actual_time_constant_sum = df[df[capacitance_column] == capacitances[2]][
        ESTIMATED_TIME_CONSTANT_COLUMN]

    logging.info("Expected time constant sum: mean = %f, stddev = %f",
                 np.mean(expected_time_constant_sum),
                 np.std(expected_time_constant_sum))
    logging.info("Actual time constant sum:, mean = %f, stddev = %f",
                 actual_time_constant_sum.mean(),
                 actual_time_constant_sum.std())

    # Plot a histogram of the time constant sums.
    fig, ax = plt.subplots(figsize=(12, 6))
    minimum_time_constant = np.min(
        (*expected_time_constant_sum, *actual_time_constant_sum))
    maximum_time_constant = np.max(
        (*expected_time_constant_sum, *actual_time_constant_sum))
    bins = np.arange(minimum_time_constant - 0.005,
                     maximum_time_constant + 0.005, 0.0003)
    ax.hist(expected_time_constant_sum,
            bins=bins,
            density=True,
            histtype="step",
            linestyle="--",
            color="C0",
            label="Expected time constant sum")
    actual_time_constant_sum.hist(ax=ax,
                                  bins=bins,
                                  density=True,
                                  histtype="step",
                                  linestyle="--",
                                  color="C1",
                                  label="Actual time constant sum")
    ax.plot(bins,
            scipy.stats.gaussian_kde(expected_time_constant_sum)(bins),
            color="C0",
            label="Estimated density of expected sum")
    ax.plot(bins,
            scipy.stats.gaussian_kde(actual_time_constant_sum)(bins),
            color="C1",
            label="Estimated density of actual sum")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1
    analyze_cap_dac_data(FLAGS.data, FLAGS.sampling_rate)
    run_cap_dac_sanity_check(FLAGS.data)


if __name__ == "__main__":
    flags.DEFINE_string(
        "data",
        "analysis/scum/adc/sensor/resistive/data/adc_data_resistor_47k_cap_dac.csv",
        "Data filename.")
    flags.DEFINE_float("sampling_rate", 100, "Sampling rate in Hz.")

    app.run(main)
