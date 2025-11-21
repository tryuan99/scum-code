import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from absl import app, flags, logging
from matplotlib.ticker import FuncFormatter

import utils.visualization.mpl_config
from analysis.scum.radio.tuning.tuning_code import TuningCode

FLAGS = flags.FLAGS

# Tuning code column.
TUNING_CODE_COLUMN = "Tuning code"

# TX/RX label.
TX_LABEL = "TX"
RX_LABEL = "RX"


def _tuning_code_formatter(x: float, position: float) -> str:
    """Formats the tuning code tick labels.

    Args:
        x: Tick value.
        position: Position.

    Returns:
        The string containing the coarse, mid, and fine codes.
    """
    tuning_code = TuningCode.tuning_code_to_coarse_mid_fine(int(np.abs(x)))
    if x < 0:
        return rf"$-{tuning_code}$"
    return rf"${tuning_code}$"


def plot_tx_rx_tuning_codes(df: pd.DataFrame) -> None:
    """Plots the TX and RX tuning codes by 802.15.4 channel.

    Args:
        data: Tuning codes data.
    """
    (
        tx_rx_column,
        channel_column,
        coarse_column,
        mid_column,
        fine_column,
        tuning_code_column,
    ) = df.columns

    # Plot the TX and RX tuning codes.
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, (tx_rx, tx_rx_group) in enumerate(df.groupby(tx_rx_column)):
        tx_rx_group.plot.scatter(
            tuning_code_column,
            channel_column,
            ax=ax,
            c=f"C{index}",
            alpha=0.5,
            label=tx_rx,
            legend=True,
        )
    ax.set_title("TX and RX tuning codes")
    ax.xaxis.set_major_formatter(FuncFormatter(_tuning_code_formatter))
    plt.show()

    # Plot the difference between the TX and RX tuning codes.
    fig, ax = plt.subplots(figsize=(12, 6))
    for (channel, channel_group) in df.groupby(channel_column):
        tx = channel_group.groupby(tx_rx_column).get_group(TX_LABEL)
        rx = channel_group.groupby(tx_rx_column).get_group(RX_LABEL)
        tx_rx_differences = (tx[tuning_code_column].to_numpy()[:, None] -
                             rx[tuning_code_column].to_numpy()).flatten()
        plt.scatter(
            np.full(tx_rx_differences.shape, channel),
            tx_rx_differences,
        )
    ax.axhline(
        TuningCode(0, -1, 31).tuning_code(),
        color="red",
        linestyle="--",
    )
    ax.axhline(
        TuningCode(0, -1, 0).tuning_code(),
        color="red",
        linestyle="--",
    )
    ax.set_title("Difference between TX and RX tuning codes")
    ax.set_xlabel("Channel")
    ax.set_ylabel(r"TX $-$ RX difference")
    ax.yaxis.set_major_formatter(FuncFormatter(_tuning_code_formatter))
    plt.show()


def plot_rx_tuning_code_extrapolation(
        df: pd.DataFrame, num_mid_codes_between_channels: int) -> None:
    """Plots the RX tuning code extrapolation.

    Args:
        data: Tuning codes data.
        num_mid_codes_between_channels: Number of mid codes between channels.
    """
    (
        tx_rx_column,
        channel_column,
        coarse_column,
        mid_column,
        fine_column,
        tuning_code_column,
    ) = df.columns
    df_rx = df[df[tx_rx_column] == "RX"]

    # Extrapolate the preceding RX channel.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 6))
    df_rx.plot.scatter(
        tuning_code_column,
        channel_column,
        ax=ax,
        c="C0",
        alpha=0.8,
    )
    for channel in df[channel_column].unique():
        channel_data = df_rx[df_rx[channel_column] == channel]
        min_mid_code = channel_data[mid_column].min()
        max_mid_code = channel_data[mid_column].max()
        ax.axvline(
            TuningCode(
                channel_data[coarse_column].min(),
                min_mid_code - num_mid_codes_between_channels,
                0,
            ).tuning_code(),
            color="red",
            linestyle="--",
            alpha=0.5,
        )
        ax.axvline(
            TuningCode(
                channel_data[coarse_column].max(),
                max_mid_code - num_mid_codes_between_channels,
                31,
            ).tuning_code(),
            color="red",
            linestyle="--",
            alpha=0.5,
        )
    ax.set_title("Extrapolating the preceding RX channel")
    ax.xaxis.set_major_formatter(FuncFormatter(_tuning_code_formatter))
    plt.show()

    # Extrapolate the succeeding RX channel.
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 6))
    df_rx.plot.scatter(
        tuning_code_column,
        channel_column,
        ax=ax,
        c="C0",
        alpha=0.8,
    )
    for channel in df[channel_column].unique():
        channel_data = df_rx[df_rx[channel_column] == channel]
        min_mid_code = channel_data[mid_column].min()
        max_mid_code = channel_data[mid_column].max()
        ax.axvline(
            TuningCode(
                channel_data[coarse_column].min(),
                min_mid_code + num_mid_codes_between_channels,
                0,
            ).tuning_code(),
            color="green",
            linestyle="--",
            alpha=0.5,
        )
        ax.axvline(
            TuningCode(
                channel_data[coarse_column].max(),
                max_mid_code + num_mid_codes_between_channels,
                31,
            ).tuning_code(),
            color="green",
            linestyle="--",
            alpha=0.5,
        )
    ax.set_title("Extrapolating the succeeding RX channel")
    ax.xaxis.set_major_formatter(FuncFormatter(_tuning_code_formatter))
    plt.show()


def main(argv):
    assert len(argv) == 1

    # Open the tuning codes file.
    df = pd.read_csv(FLAGS.data, comment="#")
    (
        tx_rx_column,
        channel_column,
        coarse_column,
        mid_column,
        fine_column,
    ) = df.columns
    df[TUNING_CODE_COLUMN] = TuningCode.coarse_mid_fine_to_tuning_code(
        df[coarse_column], df[mid_column], df[fine_column])
    logging.info(df.describe())

    plot_tx_rx_tuning_codes(df)
    plot_rx_tuning_code_extrapolation(df, FLAGS.num_mid_codes_between_channels)


if __name__ == "__main__":
    flags.DEFINE_string("data",
                        "analysis/scum/radio/tuning/data/tuning_codes_1.csv",
                        "Data filename.")
    flags.DEFINE_integer("num_mid_codes_between_channels", 6,
                         "Number of mid codes between channels.")

    app.run(main)
