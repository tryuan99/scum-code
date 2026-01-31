"""The event plotter plots events during the OpenWSN network over time."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

import utils.visualization.mpl_config
from analysis.scum.radio.tuning.tuning_code import TuningCode

# Time correction regex pattern.
TIME_CORRECTION_REGEX_PATTERN = r"\*([\d-]+)"

# RX tuning code regex pattern.
TUNING_CODE_REGEX_PATTERN = r"([\d]+).([\d]+).([\d]+)"
RX_REGEX_PATTERN = r"RX"

# Successful RX regex pattern.
SUCCESSFUL_RX_REGEX_PATTERN = r"^R$"

# Successful acknolwedgment regex pattern.
SUCCESSFUL_ACK_REGEX_PATTERN = r"^A$"

# Missed acknowledgment regex pattern.
MISSED_ACK_REGEX_PATTERN = r"^E$"


class EventPlotter:
    """OpenWSN event plotter.

    Args:
        timestamps: Column of timestamps.
        logs: Column of logs.
        start_time: Log start time.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        timestamp_column, log_column = df.columns
        self.timestamps = pd.to_datetime(df[timestamp_column])
        self.logs = df[log_column]
        self.start_time = self.timestamps.min()

    def plot_timeline(self) -> None:
        """Plots a timeline of the events.

        The timeline plots the following events:
         - Time correction in us
         - RX tuning code
         - Successful receives and successful and missed acknowledgments
        """
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 6),
            sharex=True,
        )

        # Plot the time correction in us.
        time_correction_rows = self.logs.str.startswith("*")
        time_correction_timestamps = (
            self._get_timestamps_from_indices(time_correction_rows))
        time_corrections = pd.to_numeric(
            self.logs[time_correction_rows].str.extract(
                TIME_CORRECTION_REGEX_PATTERN).squeeze())
        axes[0].plot(time_correction_timestamps, time_corrections, color="C0")
        axes[0].set_ylim([-200, 200])
        axes[0].set_ylabel("Time correction [µs]")

        # Plot the RX tuning code.
        rx_tuning_code_rows = (self.logs.str.contains(TUNING_CODE_REGEX_PATTERN)
                               & self.logs.str.contains(RX_REGEX_PATTERN))
        rx_tuning_code_timestamps = (
            self._get_timestamps_from_indices(rx_tuning_code_rows))
        rx_tuning_codes = self.logs[rx_tuning_code_rows].str.extract(
            TUNING_CODE_REGEX_PATTERN).apply(pd.to_numeric)
        coarse_column, mid_column, fine_column = rx_tuning_codes.columns
        tuning_codes = TuningCode.coarse_mid_fine_to_tuning_code(
            rx_tuning_codes[coarse_column], rx_tuning_codes[mid_column],
            rx_tuning_codes[fine_column])
        # Modify the tuning codes array to plot a stairstep line.
        tuning_codes = pd.concat([
            pd.Series([tuning_codes.iloc[0]]),
            np.repeat(tuning_codes, 2),
        ])
        tuning_code_timestamps = np.repeat(rx_tuning_code_timestamps, 2)
        tuning_code_timestamps.iloc[0] = self._calculate_timestamps(
            self.timestamps.min())
        tuning_code_timestamps = pd.concat([
            tuning_code_timestamps,
            pd.Series([self._calculate_timestamps(self.timestamps.max())]),
        ])
        axes[1].plot(tuning_code_timestamps, tuning_codes, color="C0")
        axes[1].yaxis.set_major_formatter(
            FuncFormatter(self._format_tuning_code))
        axes[1].set_ylabel("RX tuning code")

        # Plot successful receives.
        successful_rx_rows = (
            self.logs.str.contains(SUCCESSFUL_RX_REGEX_PATTERN))
        successful_rx_timestamps = (
            self._get_timestamps_from_indices(successful_rx_rows))
        axes[2].scatter(
            successful_rx_timestamps,
            pd.Series([1]).repeat(len(successful_rx_timestamps)),
            label="Successful RX",
            color="C0",
        )

        # Plot successful and missed acknowledgments.
        successful_ack_rows = (
            self.logs.str.contains(SUCCESSFUL_ACK_REGEX_PATTERN))
        successful_ack_timestamps = (
            self._get_timestamps_from_indices(successful_ack_rows))
        axes[2].scatter(
            successful_ack_timestamps,
            pd.Series([0]).repeat(len(successful_ack_timestamps)),
            label="Successful acknowledgments",
            color="C1",
        )
        missed_ack_rows = self.logs.str.contains(MISSED_ACK_REGEX_PATTERN)
        missed_ack_timestamps = self._get_timestamps_from_indices(
            missed_ack_rows)
        axes[2].scatter(
            missed_ack_timestamps,
            pd.Series([-1]).repeat(len(missed_ack_timestamps)),
            label="Missed acknowledgments",
            color="C2",
        )
        axes[2].set_yticks([])
        axes[2].set_ylim([-1.5, 1.5])
        axes[2].set_ylabel("TX/RX")
        axes[2].legend()

        axes[2].xaxis.set_major_formatter(self._format_timestamp)
        plt.xlabel("Time [s]")
        plt.show()

    def _get_timestamps_from_indices(self, row_indices: pd.Series) -> pd.Series:
        """Get the timestamps from the row indices.

        Args:
            row_indices: Row indices

        Returns:
            The timestamps since the beginning of the log in seconds.
        """
        return self._calculate_timestamps(self.timestamps[row_indices])

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
    def _format_timestamp(x: float, position: float) -> str:
        """Formats the timestamp tick labels.

        Args:
            x: Tick value.
            position: Position.

        Returns:
            The timestamp string.
        """
        return f"{int(x / 1e9)}"

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
        if y < 0:
            return rf"$-{tuning_code}$"
        return rf"${tuning_code}$"
