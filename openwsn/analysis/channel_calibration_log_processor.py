"""The channel calibration log processor processes the log outputted by SCuM
during channel calibration and records the time and channel of a successful TX
or RX.
"""

import re
from datetime import datetime

# Minimum channel.
CHANNEL_CALIBRATION_MIN_CHANNEL = 11

# Maximum channel.
CHANNEL_CALIBRATION_MAX_CHANNEL = 26

# Channel calibration log prefix pattern.
CHANNEL_CALIBRATION_LOG_PREFIX_PATTERN = r"\w\d{4} ([\d:\.]+) \d+ [\w\.]+:\d+\]"

# Channel calibration log start pattern.
CHANNEL_CALIBRATION_LOG_START_PATTERN = r"Cal complete"

# Channel calibration TX start pattern.
CHANNEL_CALIBRATION_LOG_TX_PATTERN = r"TX (\d+) \*"

# Channel calibration RX pattern.
CHANNEL_CALIBRATION_LOG_RX_PATTERN = r"RX (\d+) \*"

# Channel calibration log date format.
CHANNEL_CALIBRATION_LOG_DATE_FORMAT = "%H:%M:%S.%f"


class ChannelCalibrationTxRxSuccess:
    """Channel calibration TX or RX success.

    Attributes:
        channel: Channel.
        time: Time since the start time.
    """

    def __init__(self, channel: int, time: datetime) -> None:
        self.channel = channel
        self.time = time


class ChannelCalibrationLogProcessor:
    """Channel calibration log processor.

    Attributes:

        start_time: Start time of the log.
        tx_successes: List of TX successes.
        rx_successes: List of RX successes.
    """

    def __init__(self, log_file: str) -> None:
        (
            self.start_time,
            self.tx_successes,
            self.rx_successes,
        ) = self.process_log(log_file)

    @staticmethod
    def process_log(
        log_file: str
    ) -> tuple[datetime, list[ChannelCalibrationTxRxSuccess],
               list[ChannelCalibrationTxRxSuccess]]:
        """Processes the log.

        Attributes:
            log_file: Path to the log file.

        Returns:
            A tuple consisting of the start time, a list of TX successes, and a
            list of RX successes.
        """
        start_time: datetime = None
        tx_successes: ChannelCalibrationTxRxSuccess = []
        rx_successes: ChannelCalibrationTxRxSuccess = []
        with open(log_file, "r") as log:
            for line in log:
                if start_time is None:
                    # Search for the start pattern.
                    match = re.search(
                        rf"{CHANNEL_CALIBRATION_LOG_PREFIX_PATTERN} "
                        rf"{CHANNEL_CALIBRATION_LOG_START_PATTERN}",
                        line,
                    )
                    if match is not None:
                        start_time_str = match.group(1)
                        start_time = (ChannelCalibrationLogProcessor.
                                      parse_timestamp(start_time_str))
                else:
                    # Search for TX successes.
                    tx_success = ChannelCalibrationLogProcessor.parse_success(
                        line,
                        CHANNEL_CALIBRATION_LOG_TX_PATTERN,
                        start_time,
                    )
                    if tx_success is not None:
                        tx_successes.append(tx_success)

                    # Seach for RX successes.
                    rx_success = ChannelCalibrationLogProcessor.parse_success(
                        line,
                        CHANNEL_CALIBRATION_LOG_RX_PATTERN,
                        start_time,
                    )
                    if rx_success is not None:
                        rx_successes.append(rx_success)
        return start_time, tx_successes, rx_successes

    @staticmethod
    def parse_timestamp(timestamp: str) -> datetime:
        """Parses the timestamp string.

        Args:
            timestamp: Timestamp string.

        Returns:
            The parsed datetime object.
        """
        return datetime.strptime(timestamp, CHANNEL_CALIBRATION_LOG_DATE_FORMAT)

    @staticmethod
    def parse_success(
        line: str,
        pattern: str,
        start_time: datetime,
    ) -> ChannelCalibrationTxRxSuccess:
        """Parses the line for any TX or RX successes.

        Args:
            line: Line to parse.
            pattern: The pattern for a TX or RX success.
            start_time: Start time.

        Returns:
            The channel calibration success if the line matches the pattern.
            Otherwise returns None.
        """
        match = re.search(
            rf"{CHANNEL_CALIBRATION_LOG_PREFIX_PATTERN} {pattern}", line)
        if match is not None:
            time_str = match.group(1)
            time = ChannelCalibrationLogProcessor.parse_timestamp(time_str)
            channel = int(match.group(2))
            return ChannelCalibrationTxRxSuccess(channel, time - start_time)
        return None
