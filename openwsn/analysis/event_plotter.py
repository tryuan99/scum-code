"""The event plotter plots events from a single device within the OpenWSN network over time."""

import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from matplotlib.ticker import FuncFormatter

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

    def get_time_correction(self) -> tuple[pd.Series, pd.Series]:
        """Get the time correction events.

        Returns:
            A tuple consisting of the time correction timestamps and the time
            correction data.
        """
        time_correction_rows = self.logs.str.startswith("*")
        time_correction_timestamps = (
            self._get_timestamps_from_indices(time_correction_rows))
        time_corrections = pd.to_numeric(
            self.logs[time_correction_rows].str.extract(
                TIME_CORRECTION_REGEX_PATTERN).squeeze())
        return time_correction_timestamps, time_corrections

    def get_rx_tuning_codes(self) -> tuple[pd.Series, pd.DataFrame]:
        """Get the RX tuning codes.

        Returns:
            A tuple consisting of the RX tuning code timestamps and RX tuning
            codes in separate coarse, mid, and fine columns.
        """
        rx_tuning_code_rows = (self.logs.str.contains(TUNING_CODE_REGEX_PATTERN)
                               & self.logs.str.contains(RX_REGEX_PATTERN))
        rx_tuning_code_timestamps = (
            self._get_timestamps_from_indices(rx_tuning_code_rows))
        rx_tuning_codes = self.logs[rx_tuning_code_rows].str.extract(
            TUNING_CODE_REGEX_PATTERN).apply(pd.to_numeric)
        return rx_tuning_code_timestamps, rx_tuning_codes

    def get_successful_rx_timestamps(self) -> pd.Series:
        """Get the successful RX timestamps.

        Returns:
            A series containing the successful RX timestamps.
        """
        successful_rx_rows = (
            self.logs.str.contains(SUCCESSFUL_RX_REGEX_PATTERN))
        successful_rx_timestamps = (
            self._get_timestamps_from_indices(successful_rx_rows))
        return successful_rx_timestamps

    def get_successful_ack_timestamps(self) -> pd.Series:
        """Get the successful acknowledgment timestamps.

        Returns:
            A series containing the successful acknowledgment timestamps.
        """
        successful_ack_rows = (
            self.logs.str.contains(SUCCESSFUL_ACK_REGEX_PATTERN))
        successful_ack_timestamps = (
            self._get_timestamps_from_indices(successful_ack_rows))
        return successful_ack_timestamps

    def get_missed_ack_timestamps(self) -> pd.Series:
        """Get the missed acknowledgment timestamps.

        Returns:
            A series containing the missed acknowledgment timestamps.
        """
        missed_ack_rows = self.logs.str.contains(MISSED_ACK_REGEX_PATTERN)
        missed_ack_timestamps = (
            self._get_timestamps_from_indices(missed_ack_rows))
        return missed_ack_timestamps

    def plot_timeline(self) -> None:
        """Plots a timeline of the events.

        The timeline plots the following events:
         - Time correction in us
         - RX tuning code
         - Successful receives and successful and missed acknowledgments
        """
        plt.style.use(["science", "grid"])
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 6),
            sharex=True,
        )

        # Plot the time correction in us.
        time_correction_timestamps, time_corrections = self.get_time_correction(
        )
        axes[0].plot(
            time_correction_timestamps.dt.total_seconds(),
            time_corrections,
            color="C0",
        )
        axes[0].set_ylim([-200, 200])
        axes[0].set_ylabel("Time correction [µs]")

        # Plot the RX tuning code.
        rx_tuning_code_timestamps, rx_tuning_codes = self.get_rx_tuning_codes()
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
        axes[1].plot(
            tuning_code_timestamps.dt.total_seconds(),
            tuning_codes,
            color="C0",
        )
        axes[1].yaxis.set_major_formatter(
            FuncFormatter(self._format_tuning_code))
        axes[1].set_ylabel("RX tuning code")

        # Plot successful receives.
        successful_rx_timestamps = self.get_successful_rx_timestamps()
        axes[2].scatter(
            successful_rx_timestamps.dt.total_seconds(),
            np.repeat(1, len(successful_rx_timestamps)),
            label="Successful RX",
            color="C0",
        )

        # Plot successful and missed acknowledgments.
        successful_ack_timestamps = self.get_successful_ack_timestamps()
        axes[2].scatter(
            successful_ack_timestamps.dt.total_seconds(),
            np.repeat(0, len(successful_ack_timestamps)),
            label="Successful acknowledgments",
            color="C1",
        )
        missed_ack_timestamps = self.get_missed_ack_timestamps()
        axes[2].scatter(
            missed_ack_timestamps.dt.total_seconds(),
            np.repeat(-1, len(missed_ack_timestamps)),
            label="Missed acknowledgments",
            color="C2",
        )
        axes[2].set_yticks([])
        axes[2].set_ylim([-1.5, 1.5])
        axes[2].set_ylabel("TX/RX events")
        axes[2].legend()
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
