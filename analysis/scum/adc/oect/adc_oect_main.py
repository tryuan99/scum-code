import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

from analysis.scum.adc.adc_config import ADC_CONFIGS, AdcConfig
from utils.regression.exponential_regression import ExponentialRegression

FLAGS = flags.FLAGS

# Regex pattern for the collection time.
COLLECTION_TIME_REGEX_PATTERN = r"Collection time: (\d+.?\d*) s"


def read_collection_time(data: str) -> float:
    """Reads the collection time from the data file.

    Args:
        data: Data filename.

    Returns:
        The collection time in seconds. Returns 0 if no collection time can be found.
    """
    with open(data) as f:
        for line in f:
            match = re.search(COLLECTION_TIME_REGEX_PATTERN, line)
            if match is not None:
                return float(match.group(1))


def plot_adc_samples(data: str, adc_config: AdcConfig) -> None:
    """Plots the ADC samples over time.

    Args:
        data: Data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    iteration_column, adc_output_column = df.columns
    logging.info(df.describe())

    # Read the collection time.
    collection_time = read_collection_time(data) or len(df)
    logging.info("Collection time = %f s", collection_time)

    # Plot the ADC samples as a function of time.
    time_axis = np.linspace(0, collection_time, len(df))
    fig, ax = plt.subplots(figsize=(12, 8))
    cutoff_indices = np.concatenate(
        ([0], np.where(np.diff(df[iteration_column]) != 0)[0] + 1, [len(df)]))
    for cutoff_index in range(1, len(cutoff_indices)):
        start_index, end_index = (
            cutoff_indices[cutoff_index - 1],
            cutoff_indices[cutoff_index],
        )
        df_iteration = df[start_index:end_index]
        data = df_iteration[adc_output_column]
        (iteration,) = df_iteration[iteration_column].unique()
        time_axis_iteration = time_axis[start_index:end_index]
        plt.plot(time_axis_iteration, data)

        # Perform an exponential regression to find the time constant.
        exponential_regression = ExponentialRegression(
            time_axis_iteration, adc_config.lsb2volt(data))
        logging.info("Iteration %d: tau = %f", iteration,
                     exponential_regression.time_constant)
    secax = ax.secondary_yaxis("right",
                               functions=(adc_config.lsb2volt,
                                          adc_config.volt2lsb))
    ax.set_title("ADC output over time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("ADC output [LSB]")
    secax.set_ylabel("ADC output [V]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_adc_samples(FLAGS.data, ADC_CONFIGS[FLAGS.board])


if __name__ == "__main__":
    flags.DEFINE_string("data", "analysis/scum/adc/oect/data/dd_oect_1.csv",
                        "Data filename.")
    flags.DEFINE_enum("board", "l35", ADC_CONFIGS.keys(), "SCuM board.")

    app.run(main)
