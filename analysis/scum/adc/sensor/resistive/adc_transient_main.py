import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

from utils.regression.exponential_regression import ExponentialRegression

FLAGS = flags.FLAGS

# Number of bits in an ADC sample.
NUM_ADC_SAMPLE_BITS = 10

# Maximum difference in LSBs between consecutive ADC samples.
MAX_DIFF_BETWEEN_CONSECUTIVE_ADC_SAMPLES = 64  # LSBs


def _rectify_transient_adc_data(data: pd.Series) -> None:
    """Rectifies the transient ADC data.

    This function assumes a decaying exponential and fixes any discontinuities
    in the ADC data caused by the stuck MSB.

    Args:
        data: ADC data column.
    """
    # Correct the ADC samples by the value of the MSB.
    correction = 2**(NUM_ADC_SAMPLE_BITS - 1)

    # Fix any discontinuities caused by the MSB bit.
    diffs = np.squeeze(
        np.argwhere(
            np.abs(np.diff(data)) > MAX_DIFF_BETWEEN_CONSECUTIVE_ADC_SAMPLES))
    data[:np.min(diffs)] += correction

    # Debounce the ADC data at the discontinuities.
    for i in range(np.min(diffs), np.max(diffs) + 1):
        if data[i] - data[i - 1] < -MAX_DIFF_BETWEEN_CONSECUTIVE_ADC_SAMPLES:
            data[i] += correction


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

    # Rectify the ADC data.
    adc_output = df[adc_output_column]
    _rectify_transient_adc_data(adc_output)

    # Perform an exponential regression.
    t = adc_output.index / sampling_rate
    exponential_regression = ExponentialRegression(t, adc_output.values)
    logging.info("tau = %f", exponential_regression.time_constant)
    logging.info("C = %g, R = %g", capacitance,
                 exponential_regression.time_constant / capacitance)

    # Plot the transient ADC data in linear and log space.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    adc_output.plot(ax=ax1)
    ax1.plot(adc_output.index,
             exponential_regression.evaluate(t),
             label="Exponential fit")
    ax1.set_title("Transient ADC output in linear space")
    ax1.set_xlabel("ADC sample")
    ax1.set_ylabel("ADC output [LSB]")
    ax1.legend()

    np.log(adc_output).plot(ax=ax2)
    ax2.plot(adc_output.index,
             np.log(exponential_regression.evaluate(t)),
             label="Exponential fit")
    ax2.set_title("Transient ADC output in log space")
    ax2.set_xlabel("ADC sample")
    ax2.set_ylabel("Log ADC output [bits]")
    ax2.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_transient_adc_data(FLAGS.data, FLAGS.sampling_rate, FLAGS.capacitance)


if __name__ == "__main__":
    flags.DEFINE_string(
        "data",
        "analysis/scum/adc/sensor/resistive/data/adc_data_thermistor_3a_3.csv",
        "Data filename.")
    flags.DEFINE_float("sampling_rate", 100, "Sampling rate in Hz.")
    flags.DEFINE_float("capacitance", 50e-9, "Fixed capacitance in F.")

    app.run(main)
