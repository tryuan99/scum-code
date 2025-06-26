import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from absl import app, flags

from openwsn.analysis.channel_calibration_log_processor import (
    CHANNEL_CALIBRATION_MAX_CHANNEL, CHANNEL_CALIBRATION_MIN_CHANNEL,
    ChannelCalibrationLogProcessor, ChannelCalibrationTxRxSuccess)

FLAGS = flags.FLAGS


def plot_channel_histogram(
    successes: list[ChannelCalibrationTxRxSuccess],
    label: str,
    ax: matplotlib.axes.Axes,
) -> None:
    """Plots a histogram of the channels.

    Args:
        successes: List of successes.
        label: Label.
    """
    # Plot a histogram of the channels.
    channels = [success.channel for success in successes]
    bins = np.arange(
        CHANNEL_CALIBRATION_MIN_CHANNEL - 0.5,
        CHANNEL_CALIBRATION_MAX_CHANNEL + 1,
    )
    counts, bins, patches = ax.hist(
        channels,
        bins=bins,
        color="C1" if label == "RX" else "C0",
        rwidth=0.8,
    )
    ax.bar_label(
        patches,
        labels=[int(count) for count in counts],
        label_type="edge",
    )
    ax.set_title(f"{label} successes")
    ax.set_xlabel("IEEE 802.15.4 channel")
    ax.set_ylabel("Count")


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
    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(20, 4))
    for index, (successes, label) in enumerate([
        (rx_successes, "RX"),
        (tx_successes, "TX"),
    ]):
        x = [success.time.total_seconds() for success in successes]
        y = [success.channel for success in successes]
        ax.scatter(
            x,
            np.array(y) + (index * 2 - 1) * 0.05,
            c=f"C{index + 1}",
            alpha=0.25,
            label=label,
        )
        first_x, first_y = [], []
        for channel in set(y):
            first_index = y.index(channel)
            first_x.append(x[first_index])
            first_y.append(y[first_index])
        ax.scatter(
            first_x,
            first_y,
            c=["blue", "red"][index],
            label=f"Channel {label} done",
            marker="*",
        )
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("IEEE 802.15.4 channel")
    ax.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    log_processor = ChannelCalibrationLogProcessor(FLAGS.log)
    plt.style.use(["science", "grid"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    plot_channel_histogram(log_processor.tx_successes, "TX", ax1)
    plot_channel_histogram(log_processor.rx_successes, "RX", ax2)
    plt.show()
    plot_success_timeline(
        log_processor.tx_successes,
        log_processor.rx_successes,
    )


if __name__ == "__main__":
    flags.DEFINE_string("log", None, "Channel calibration log.")
    flags.mark_flag_as_required("log")

    app.run(main)
