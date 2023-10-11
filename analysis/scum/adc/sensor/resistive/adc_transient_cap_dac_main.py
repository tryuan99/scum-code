import matplotlib.pyplot as plt
import pandas as pd
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


def main(argv):
    assert len(argv) == 1
    analyze_cap_dac_data(FLAGS.data, FLAGS.sampling_rate)


if __name__ == "__main__":
    flags.DEFINE_string(
        "data",
        "analysis/scum/adc/sensor/resistive/data/adc_data_resistor_220k_cap_dac.csv",
        "Data filename.")
    flags.DEFINE_float("sampling_rate", 100, "Sampling rate in Hz.")

    app.run(main)
