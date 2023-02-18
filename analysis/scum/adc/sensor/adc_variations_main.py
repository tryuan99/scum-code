import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags

from analysis.scum.adc.adc_config import ADC_CONFIGS, AdcConfig

FLAGS = flags.FLAGS


def plot_adc_variations(data: list[str], adc_config: AdcConfig) -> None:
    """Plots the ADC output variations over time at different concentrations.

    Args:
        data: List of data filename.
        adc_config: ADC configuration.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Open the ADC data files and plot the ADC output variations from each data file.
    for file_index, filename in enumerate(data):
        df = pd.read_csv(filename, comment="#")
        concentration_column, adc_output_column = df.columns

        cutoff_indices = np.concatenate(
            ([0], np.where(np.diff(df[concentration_column]) != 0)[0] + 1,
             [len(df)]))
        for cutoff_index in range(1, len(cutoff_indices)):
            start_index, end_index = (
                cutoff_indices[cutoff_index - 1],
                cutoff_indices[cutoff_index],
            )
            df_concentration = df[start_index:end_index]
            (concentration,) = df_concentration[concentration_column].unique()
            adc_data = df_concentration[adc_output_column]
            label = (f"{concentration} M"
                     if len(data) == 1 else f"{filename}: {concentration} M")
            if len(data) == 1:
                adc_data.reset_index()[adc_output_column].plot.line(ax=ax)
            else:
                adc_data.reset_index()[adc_output_column].plot.line(
                    ax=ax, color=f"C{file_index}")

    secax = ax.secondary_yaxis("right",
                               functions=(adc_config.lsb2volt,
                                          adc_config.volt2lsb))
    ax.set_title("ADC output variations")
    ax.set_xlabel("ADC sample")
    ax.set_ylabel("ADC output [LSB]")
    secax.set_ylabel("ADC output [V]")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_adc_variations(FLAGS.data, ADC_CONFIGS[FLAGS.board])


if __name__ == "__main__":
    flags.DEFINE_list(
        "data",
        [
            "analysis/scum/adc/sensor/data/adc_data_nitrate_vi_1.csv",
            "analysis/scum/adc/sensor/data/adc_data_nitrate_vi_2.csv",
        ],
        "Data filenames.",
    )
    flags.DEFINE_enum("board", "l35", ADC_CONFIGS.keys(), "SCuM board.")

    app.run(main)
