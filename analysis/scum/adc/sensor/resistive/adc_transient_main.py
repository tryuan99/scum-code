import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal
from absl import app, flags, logging

from utils.regression.exponential_regression import ExponentialRegression
from utils.regression.linear_regression import LinearRegression

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


def _filter_adc_data(data: np.ndarray,
                     sampling_rate: float,
                     cutoff_frequency: float = 10) -> np.ndarray:
    """Filters the noise from the ADC data.

    Args:
        data: ADC data.
        sampling_rate: Sampling rate in Hz.
        cutoff_frequency: Cutoff frequency in Hz.

    Returns:
        The filtered ADC data.
    """
    # Use a Butterworth filter.
    butter = scipy.signal.butter(3,
                                 cutoff_frequency,
                                 fs=sampling_rate,
                                 output="sos")
    return scipy.signal.sosfiltfilt(butter, data)


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

    # Find the index at 3tau.
    # Find the ADC output corresponding to 0 V by averaging the last ten ADC samples.
    min_adc_output = np.mean(adc_output[-10:])
    max_adc_output = np.max(adc_output)
    # At 3tau, the exponential has decayed by 95%.
    three_tau_index = np.argmax(adc_output < min_adc_output + 0.05 *
                                (max_adc_output - min_adc_output))

    # Perform an exponential regression.
    t = adc_output.index / sampling_rate
    adc_data = adc_output.values
    exponential_regression = ExponentialRegression(t[:three_tau_index],
                                                   adc_data[:three_tau_index])
    tau_exponential = exponential_regression.time_constant
    logging.info("Exponential regression:")
    logging.info("tau = %f", tau_exponential)
    logging.info("C = %g, R = %g", capacitance, tau_exponential / capacitance)

    # Perform a linear regression in log space.
    linear_regression = LinearRegression(
        t[:three_tau_index],
        np.log(adc_data[:three_tau_index] - min_adc_output))
    tau_linear = -1 / linear_regression.slope
    logging.info("Linear regression:")
    logging.info("tau = %f", tau_linear)
    logging.info("C = %g, R = %g", capacitance, tau_linear / capacitance)

    # Plot the transient ADC data in linear and log space.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    adc_output.plot(ax=ax1)
    ax1.plot(adc_output.index,
             exponential_regression.evaluate(t),
             label="Exponential fit")
    ax1.plot(adc_output.index,
             np.exp(linear_regression.evaluate(t)) + min_adc_output,
             label="Linear fit")
    ax1.set_title("Transient ADC output in linear space")
    ax1.set_xlabel("ADC sample")
    ax1.set_ylabel("ADC output [LSB]")
    ax1.legend()

    np.log(adc_output).plot(ax=ax2)
    ax2.plot(adc_output.index,
             np.log(exponential_regression.evaluate(t)),
             label="Exponential fit")
    ax2.plot(adc_output.index,
             np.log(np.exp(linear_regression.evaluate(t)) + min_adc_output),
             label="Linear fit")
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
