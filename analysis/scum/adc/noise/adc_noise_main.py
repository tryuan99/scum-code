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
    fig, ax = plt.subplots(figsize=(10, 4))
    adc_output.plot(ax=ax)
    ax.set_title("ADC output")
    ax.set_xlabel("Measurement index")
    ax.set_ylabel("ADC output [LSB]")
    plt.show()

    # Plot a histogram of the ADC data.
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(12, 5))
    minimum_adc_output = adc_output.min()
    maximum_adc_output = adc_output.max()
    bins = np.arange(minimum_adc_output - 0.5, maximum_adc_output + 1)

    # Plot the Gaussian fit.
    secax = ax.twinx()
    gaussian_fit = scipy.stats.norm.pdf(bins, adc_output.mean(),
                                        adc_output.std())
    for i in range(len(bins)):
        print(f"{bins[i]},{gaussian_fit[i]}")
    secax.plot(bins, gaussian_fit, "r", linestyle="--")
    secax.set_ylim(bottom=0)
    plt.ylabel("PDF")
    ax.set_title(
        "Histogram of the ADC data (mean=234.9 LSBs, stddev=4.54 LSBs)")
    ax.set_xlabel("ADC data [LSB]")
    ax.set_ylabel("Count")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_adc_data_histogram(FLAGS.data)


if __name__ == "__main__":
    flags.DEFINE_string("data",
                        "analysis/scum/adc/noise/data/adc_noise_data_1.csv",
                        "Data filename.")

    app.run(main)
