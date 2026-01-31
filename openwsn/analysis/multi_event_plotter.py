"""The multi-event plotter plots events from multiple devices within the OpenWSN network."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from matplotlib.ticker import FuncFormatter

from analysis.scum.radio.tuning.tuning_code import TuningCode
from openwsn.analysis.event_plotter import EventPlotter


class MultiEventPlotter:
    """OpenWSN multi-event plotter.

    Args:
        event_plotters: Individual event plotters for each device.
        start_time: Log start time.
    """

    def __init__(self, dfs: list[pd.DataFrame]) -> None:
        self.event_plotters = [EventPlotter(df) for df in dfs]
        self.start_time = min([
            event_plotter.timestamps.min()
            for event_plotter in self.event_plotters
        ])

    def plot_timeline(self) -> None:
        """Plots a timeline of the events.

        The timeline plots the following events for every device:
         - RX tuning code
         - Successful receives and successful and missed acknowledgments
        """
        plt.style.use(["science", "grid"])
        fig, axes = plt.subplots(
            1 + len(self.event_plotters),
            1,
            figsize=(12, 6),
            sharex=True,
            height_ratios=[1] + [0.5] * len(self.event_plotters),
        )

        for event_plotter_index, event_plotter in enumerate(
                self.event_plotters):
            # Plot the RX tuning code.
            rx_tuning_code_timestamps, rx_tuning_codes = (
                event_plotter.get_rx_tuning_codes())
            coarse_column, mid_column, fine_column = rx_tuning_codes.columns
            tuning_codes = TuningCode.coarse_mid_fine_to_tuning_code(
                rx_tuning_codes[coarse_column], rx_tuning_codes[mid_column],
                rx_tuning_codes[fine_column])
            # Subtract the starting tuning code.
            tuning_codes -= tuning_codes[0]
            # Modify the tuning codes array to plot a stairstep line.
            tuning_codes = pd.concat([
                pd.Series([tuning_codes.iloc[0]]),
                np.repeat(tuning_codes, 2),
            ])
            tuning_code_timestamps = np.repeat(rx_tuning_code_timestamps, 2)
            tuning_code_timestamps.iloc[0] = self._calculate_timestamps(
                event_plotter.timestamps.min())
            tuning_code_timestamps = pd.concat([
                tuning_code_timestamps,
                pd.Series([
                    self._calculate_timestamps(event_plotter.timestamps.max())
                ]),
            ])
            axes[0].plot(
                tuning_code_timestamps.dt.total_seconds(),
                tuning_codes,
                linestyle=(0, (5, 1)),
                label=f"Device {event_plotter_index + 1}",
            )

            # Plot successful receives.
            successful_rx_timestamps = (
                event_plotter.get_successful_rx_timestamps())
            axes[event_plotter_index + 1].scatter(
                successful_rx_timestamps.dt.total_seconds(),
                np.repeat(1, len(successful_rx_timestamps)),
                label="Successful RX",
                color="C0",
            )

            # Plot successful and missed acknowledgments.
            successful_ack_timestamps = (
                event_plotter.get_successful_ack_timestamps())
            axes[event_plotter_index + 1].scatter(
                successful_ack_timestamps.dt.total_seconds(),
                np.repeat(0, len(successful_ack_timestamps)),
                label="Successful acknowledgments",
                color="C1",
            )
            missed_ack_timestamps = event_plotter.get_missed_ack_timestamps()
            axes[event_plotter_index + 1].scatter(
                missed_ack_timestamps.dt.total_seconds(),
                np.repeat(-1, len(missed_ack_timestamps)),
                label="Missed acknowledgments",
                color="C2",
            )

            axes[event_plotter_index + 1].set_yticks([])
            axes[event_plotter_index + 1].set_ylim([-1.5, 1.5])
            axes[event_plotter_index +
                 1].set_ylabel(f"Device {event_plotter_index + 1} events")
            axes[event_plotter_index + 1].legend()

        axes[0].yaxis.set_major_formatter(
            FuncFormatter(self._format_tuning_code))
        axes[0].set_ylabel("RX tuning code offset from initial")
        axes[0].legend()
        plt.xlabel("Time [s]")
        plt.show()

    def _calculate_timestamps(self, timestamps: pd.Series) -> pd.Series:
        """Subtract the start time from the timestamps to find the timestamps
        since the beginning of the log.

        Args:
            timestamps: Timestamps.

        Returns:
            The formatted timestamps.
        """
        return timestamps - self.start_time

    @staticmethod
    def _format_tuning_code(y: float, position: float) -> str:
        """Formats the tuning code tick labels.

        Args:
            y: Tick value.
            position: Position.

        Returns:
            The string containing the coarse, mid, and fine codes.
        """
        tuning_code = TuningCode.tuning_code_to_coarse_mid_fine(int(np.abs(y)))
        return f"{tuning_code}"
