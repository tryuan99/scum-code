import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from absl import app, flags, logging

FLAGS = flags.FLAGS


def plot_adc_data_histogram(data: str) -> None:
    """Plots the histogram of the ADC data.

    Args:
        data: Data filename.
    """
    # Open the ADC data file.
    df = pd.read_csv(data, comment="#")
    (adc_output_column,) = df.columns
    logging.info(df.describe())

    # Plot the ADC data over time.
    adc_output = df[adc_output_column]
    fig, ax = plt.subplots(figsize=(12, 8))
    adc_output.plot(ax=ax)
    ax.set_title("ADC output")
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("ADC output [LSB]")
    plt.show()

    # Plot a histogram of the ADC data.
    fig, ax = plt.subplots(figsize=(12, 8))
    minimum_adc_output = adc_output.min()
    maximum_adc_output = adc_output.max()
    bins = np.arange(minimum_adc_output - 0.5, maximum_adc_output + 1)
    adc_output.hist(ax=ax, bins=bins)

    # Plot the Gaussian fit.
    secax = ax.twinx()
    gaussian_fit = scipy.stats.norm.pdf(bins, adc_output.mean(),
                                        adc_output.std())
    secax.plot(bins, gaussian_fit, "r")
    secax.set_ylim(bottom=0)
    ax.set_title("Histogram of the ADC data")
    ax.set_xlabel("ADC data [LSB]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_adc_data_histogram(FLAGS.data)


if __name__ == "__main__":
    flags.DEFINE_string("data",
                        "analysis/scum/adc/noise/data/adc_noise_data_1.csv",
                        "Data filename.")

    app.run(main)
