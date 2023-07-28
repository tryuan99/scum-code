import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags, logging

from analysis.scum.adc.sensor.resistive.adc_data import (
    EXPONENTIAL_SCALING_FACTOR, SIGMA, ExponentialAdcData)

FLAGS = flags.FLAGS

# Offset of the decaying exponential.
EXPONENTIAL_OFFSET = 127


def _generate_transient_adc_data(tau: float,
                                 sampling_rate: float,
                                 num_taus: float = 5) -> np.ndarray:
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


def plot_example_transient_adc_data(tau: float, sampling_rate: float) -> None:
    """Plots an example transient ADC data.

    Args:
        tau: Time constant in seconds.
        sampling_rate: Sampling rate in Hz.
    """
    adc_output = _generate_transient_adc_data(tau, sampling_rate)
    adc_data = ExponentialAdcData(adc_output, sampling_rate)

    # Perform an exponential regression.
    exponential_regression = adc_data.perform_exponential_regression()
    tau_exponential = exponential_regression.time_constant
    logging.info("Exponential regression: tau = %f", tau_exponential)

    # Perform a linear regression in log space.
    linear_regression = adc_data.perform_linear_regression()
    tau_linear = -1 / linear_regression.slope
    logging.info("Linear regression: tau = %f", tau_linear)

    # Perform a weighted linear regression in log space.
    weighted_linear_regression = adc_data.perform_weighted_linear_regression()
    tau_weighted_linear = -1 / weighted_linear_regression.slope
    logging.info("Weighted linear regression: tau = %f", tau_weighted_linear)

    # Plot the transient ADC data in linear and log space.
    n = np.arange(len(adc_output))
    t = adc_data.t_axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(n, adc_data.samples, label="ADC data")
    ax1.plot(n, exponential_regression.evaluate(t), label="Exponential fit")
    ax1.plot(n,
             np.exp(linear_regression.evaluate(t)) + adc_data.min_adc_output,
             label="Linear fit")
    ax1.plot(n,
             np.exp(weighted_linear_regression.evaluate(t)) +
             adc_data.min_adc_output,
             label="Weighted linear fit")
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
    ax2.set_title("Transient ADC output in log space minus offset")
    ax2.set_xlabel("ADC sample")
    ax2.set_ylabel("Log ADC output minus offset [bits]")
    ax2.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_example_transient_adc_data(FLAGS.tau, FLAGS.sampling_rate)


if __name__ == "__main__":
    flags.DEFINE_float("tau", 2, "Time constant in seconds.")
    flags.DEFINE_float("sampling_rate", 100, "Sampling rate in Hz.")

    app.run(main)
