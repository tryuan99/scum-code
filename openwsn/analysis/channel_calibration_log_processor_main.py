import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

import utils.visualization.mpl_config
from openwsn.analysis.channel_calibration_log_processor import (
    CHANNEL_CALIBRATION_MAX_CHANNEL, CHANNEL_CALIBRATION_MIN_CHANNEL,
    ChannelCalibrationLogProcessor, ChannelCalibrationTxRxSuccess)

FLAGS = flags.FLAGS


def plot_channel_histogram(
    successes: list[ChannelCalibrationTxRxSuccess],
    label: str,
) -> None:
    """Plots a histogram of the channels.

    Args:
        successes: List of successes.
        label: Label.
    """
    # Plot a histogram of the channels.
    channels = [success.channel for success in successes]
    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.arange(
        CHANNEL_CALIBRATION_MIN_CHANNEL - 0.5,
        CHANNEL_CALIBRATION_MAX_CHANNEL + 1,
    )
    counts, bins, patches = ax.hist(channels, bins=bins)
    ax.bar_label(
        patches,
        labels=[int(count) for count in counts],
        label_type="edge",
    )
    ax.set_title(f"Channel histogram of the {label} successes")
    ax.set_xlabel("Channel")
    plt.show()


def plot_success_timeline(
    tx_successes: list[ChannelCalibrationTxRxSuccess],
    rx_successes: list[ChannelCalibrationTxRxSuccess],
) -> None:
    """Plots a timeline of TX and RX successes for each channel.

    Args:
        tx_successes: List of TX successes.
        rx_successes: List of RX successes.
    """
    # Plot a timeline of the TX and RX successes.
    fig, ax = plt.subplots(figsize=(12, 6))
    for index, (successes, label) in enumerate([
        (rx_successes, "RX"),
        (tx_successes, "TX"),
    ]):
        x = [success.time.total_seconds() for success in successes]
        y = [success.channel for success in successes]
        ax.scatter(
            x,
            y,
            c=f"C{index + 1}",
            alpha=0.25,
            label=label,
        )
    ax.set_title("Channel calibration timeline")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Channel")
    ax.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    log_processor = ChannelCalibrationLogProcessor(FLAGS.log)
    plot_channel_histogram(log_processor.tx_successes, "TX")
    plot_channel_histogram(log_processor.rx_successes, "RX")
    plot_success_timeline(
        log_processor.tx_successes,
        log_processor.rx_successes,
    )


if __name__ == "__main__":
    flags.DEFINE_string("log", None, "Channel calibration log.")
    flags.mark_flag_as_required("log")

    app.run(main)
