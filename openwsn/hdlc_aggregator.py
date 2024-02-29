"""The HDLC aggregator aggregates bytes until the HDLC frame is complete."""

from enum import Enum, auto

from absl import logging

from openwsn.hdlc import HDLC_FLAG, Hdlc


class HdlcAggregatorState(Enum):
    """HDLC aggregator state enumeration."""
    READY = auto()
    RECEIVING = auto()
    DONE = auto()


class HdlcAggregator:
    """Aggregator for HDLC frames.

    Attributes:
        frame: Data from the HDLC frame.
        buffer: Raw data from the HDLC frame.
        state: Aggregator state.
    """

    def __init__(self):
        self.reset()

    def receive_byte(self, byte: bytes) -> None:
        """Receives a single byte to aggregate.

        Args:
            byte: Byte to aggregate.
        """
        if self.state == HdlcAggregatorState.READY:
            if byte == HDLC_FLAG:
                self.frame += byte
                self.state = HdlcAggregatorState.RECEIVING
        elif self.state == HdlcAggregatorState.RECEIVING:
            self.frame += byte
            if byte == HDLC_FLAG:
                try:
                    self.buffer = Hdlc.dehdlcify(self.frame)
                    self.state = HdlcAggregatorState.DONE
                except Exception as e:
                    logging.exception(e)
                    self.reset()

    def has_received_data(self) -> bool:
        """Returns whether a complete HDLC frame has been recevied."""
        return self.state == HdlcAggregatorState.DONE

    def reset(self) -> None:
        """Resets all variables for the subsequent HDLC frame."""
        self.frame = bytearray()
        self.buffer = bytearray()
        self.state = HdlcAggregatorState.READY
