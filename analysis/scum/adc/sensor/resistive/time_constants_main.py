import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

FLAGS = flags.FLAGS

# Time constant columm name.
TIME_CONSTANT_COLUMN = "Time constant [s]"


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
    df[TIME_CONSTANT_COLUMN] = (df[time_constant_column] /
                                df[time_constant_scaling_factor_column])
    logging.info(df.describe())

    time_constant_by_resistance = df.groupby(resistance_column)
    logging.info(time_constant_by_resistance.describe())

    # Plot the mean time constant as a function of the resistance.
    fig, ax = plt.subplots(figsize=(12, 8))
    time_constant_by_resistance.mean().plot.line(y=TIME_CONSTANT_COLUMN,
                                                 ax=ax,
                                                 label="Measured (mean)")
    resistances = df[resistance_column].unique()
    ax.plot(resistances, resistances * 1e6 * capacitance, label="Ideal")
    ax.set_title(f"Mean estimated time constant (C={capacitance * 1e9:.1f} nF)")
    ax.set_ylabel("Time constant [s]")
    plt.legend()
    plt.show()

    # Plot the standard deviation of the estimated time constant as a function
    # of the resistance.
    fig, ax = plt.subplots(figsize=(12, 8))
    time_constant_by_resistance.std().plot.line(y=TIME_CONSTANT_COLUMN,
                                                ax=ax,
                                                label="Measured")
    ax.set_title(f"Standard deviation of the estimated time constant "
                 f"(C={capacitance * 1e9:.1f} nF)")
    ax.set_ylabel("Standard deviation [s]")
    plt.legend()
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
