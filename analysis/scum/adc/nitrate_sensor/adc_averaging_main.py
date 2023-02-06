import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging

from analysis.scum.adc.adc_config import ADC_CONFIGS, AdcConfig

FLAGS = flags.FLAGS

# Number of ADC samples to average over.
NUM_SAMPLES_TO_AVERAGE = np.arange(1, 101)
NUM_SAMPLES_TO_AVERAGE_AND_PLOT = [1, 5, 10, 20, 50, 100]


def plot_adc_averaging(data: str, adc_config: AdcConfig) -> None:
    """Plots the output of averaging over ADC samples.

    Args:
        data: Data filename.
        adc_config: ADC configuration.
    """
    # Open the ADC characterization data file.
    df = pd.read_csv(data, comment="#")
    concentration_column, adc_output_column = df.columns
    df_by_input = df.groupby(concentration_column)
    logging.info(df_by_input.describe())

    cutoff_indices = np.concatenate(
        ([0], np.where(np.diff(df[concentration_column]) != 0)[0] + 1,
         [len(df)]))
    longest_sequence_index = np.argmax(np.diff(cutoff_indices))
    averaging_data = df[cutoff_indices[longest_sequence_index]:cutoff_indices[
        longest_sequence_index + 1]]
    (concentration,) = averaging_data[concentration_column].unique()
    logging.info("Number of ADC samples: %d", len(averaging_data))
    logging.info("Concentration: %f M", concentration)

    # Average over the given number of samples and plot the resulting averaged samples.
    fig, ax = plt.subplots(figsize=(12, 8))
    averaging_stddev = []
    for num_samples_to_average in NUM_SAMPLES_TO_AVERAGE:
        averaging_data_slice = averaging_data[adc_output_column][:(
            len(averaging_data) //
            num_samples_to_average) * num_samples_to_average].values
        averaged_data = np.mean(averaging_data_slice.reshape(
            -1, num_samples_to_average),
                                axis=1)

        # Calculate the standard deviation after averaging.
        averaged_stddev = np.std(averaged_data)
        averaging_stddev.append(averaged_stddev)

        if num_samples_to_average in NUM_SAMPLES_TO_AVERAGE_AND_PLOT:
            logging.info(
                "Averaging over %d samples: stddev = %f",
                num_samples_to_average,
                averaged_stddev,
            )
            plt.plot(
                np.arange(len(averaged_data)) * num_samples_to_average,
                averaged_data,
                label=f"Averaging over {num_samples_to_average} samples",
            )
    secax = ax.secondary_yaxis("right",
                               functions=(adc_config.lsb2volt,
                                          adc_config.volt2lsb))
    ax.set_title(f"Averaging over ADC samples at {concentration} M")
    ax.set_xlabel("Averaged ADC sample")
    ax.set_ylabel("ADC output [LSB]")
    secax.set_ylabel("ADC output [V]")
    plt.legend()
    plt.show()

    # Plot the standard deviations as a function of the number of ADC samples to average over.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(NUM_SAMPLES_TO_AVERAGE, averaging_stddev)
    plt.axhline(y=adc_config.volt2lsb_stddev(0.001), linestyle="--")
    secax = ax.secondary_yaxis("right",
                               functions=(adc_config.lsb2volt_stddev,
                                          adc_config.volt2lsb_stddev))
    ax.set_title(
        f"Standard deviation after averaging over ADC samples at {concentration} M"
    )
    ax.set_xlabel("Number of ADC samples to average over")
    ax.set_ylabel("Standard deviation [LSB]")
    secax.set_ylabel("Standard deviation [V]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_adc_averaging(FLAGS.data, ADC_CONFIGS[FLAGS.board])


if __name__ == "__main__":
    flags.DEFINE_string(
        "data",
        "analysis/scum/adc/nitrate_sensor/data/adc_data_vi_2.csv",
        "Data filename.",
    )
    flags.DEFINE_enum("board", "l35", ADC_CONFIGS.keys(), "SCuM board.")

    app.run(main)
