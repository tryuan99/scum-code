import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging
from matplotlib.ticker import FuncFormatter

from analysis.scum.radio.tuning.tuning_code import TuningCode

FLAGS = flags.FLAGS

# Tuning code column.
TUNING_CODE_COLUMN = "Tuning code"


def plot_tuning_codes(data: str) -> None:
    """Plots the TX and RX tuning codes by 802.15.4 channel.

    Args:
        data: Data filename.
    """
    # Open the tuning codes file.
    df = pd.read_csv(data, comment="#")
    tx_rx_column, channel_column, coarse_column, mid_column, fine_column = df.columns
    df[TUNING_CODE_COLUMN] = TuningCode.coarse_mid_fine_to_tuning_code(
        df[coarse_column], df[mid_column], df[fine_column])
    logging.info(df.describe())

    # Plot the TX and RX tuning codes.
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, (tx_rx, tx_rx_group) in enumerate(df.groupby(tx_rx_column)):
        tx_rx_group.plot.scatter(TUNING_CODE_COLUMN,
                                 channel_column,
                                 ax=ax,
                                 c=f"C{index}",
                                 label=tx_rx,
                                 legend=True)
    ax.set_title("TX and RX tuning codes")

    # Format the tuning code tick labels.
    def tuning_code_formatter(x: int, position: float) -> str:
        """Formats the tuning code tick labels.

        Args:
            x: Tick value.
            position: Position.

        Returns:
            The string containing the coarse, mid, and fine codes.
        """
        return str(TuningCode.tuning_code_to_coarse_mid_fine(int(x)))

    ax.xaxis.set_major_formatter(FuncFormatter(tuning_code_formatter))
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_tuning_codes(FLAGS.data)


if __name__ == "__main__":
    flags.DEFINE_string("data",
                        "analysis/scum/radio/tuning/data/tuning_codes_1.csv",
                        "Data filename.")

    app.run(main)
