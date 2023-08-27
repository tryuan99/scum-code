import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

FLAGS = flags.FLAGS

# Actual time constant columm name.
ACTUAL_TIME_CONSTANT_COLUMN = "Actual time constant [s]"

# Estimated time constant columm name.
ESTIMATED_TIME_CONSTANT_COLUMN = "Estimated time constant [s]"


def plot_time_constants(data: str, sampling_rate: float,
                        capacitance: float) -> None:
    """Plots the time constants.

    Args:
        data: Data filename.
        sampling_rate: Sampling rate in Hz.
        capacitance: Fixed capacitance in F.
    """
    # Open the time constants data file.
    df = pd.read_csv(data, comment="#")
    (
        resistance_column,
        num_adc_samples_column,
        three_tau_index_column,
        time_constant_column,
        time_constant_scaling_factor_column,
    ) = df.columns
    df[ACTUAL_TIME_CONSTANT_COLUMN] = df[resistance_column] * 1e6 * capacitance
    df[ESTIMATED_TIME_CONSTANT_COLUMN] = (
        df[time_constant_column] / df[time_constant_scaling_factor_column])
    logging.info(df.describe())

    time_constants = df.groupby(ACTUAL_TIME_CONSTANT_COLUMN)
    logging.info(time_constants.describe())

    # Plot the mean estimated time constant.
    fig, ax = plt.subplots(figsize=(12, 8))
    time_constants.mean().plot.line(y=ESTIMATED_TIME_CONSTANT_COLUMN,
                                    ax=ax,
                                    label="Estimated")
    actual_time_constants = df[ACTUAL_TIME_CONSTANT_COLUMN].unique()
    ax.plot(actual_time_constants, actual_time_constants, "--", label="Actual")
    ax.set_title("Mean estimated time constant")
    ax.set_ylabel("Time constant [s]")
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

    # Plot the standard deviation of the estimated time constant.
    fig, ax = plt.subplots(figsize=(12, 8))
    time_constants.std().plot.line(y=ESTIMATED_TIME_CONSTANT_COLUMN, ax=ax)
    ax.set_title("Standard deviation of the estimated time constant")
    ax.set_ylabel("Standard deviation [s]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_time_constants(FLAGS.data, FLAGS.sampling_rate, FLAGS.capacitance)


if __name__ == "__main__":
    flags.DEFINE_string(
        "data",
        "analysis/scum/adc/sensor/resistive/data/time_constants_resistor.csv",
        "Data filename.")
    flags.DEFINE_float("sampling_rate", 100, "Sampling rate in Hz.")
    flags.DEFINE_float("capacitance", 50e-9, "Fixed capacitance in F.")

    app.run(main)
