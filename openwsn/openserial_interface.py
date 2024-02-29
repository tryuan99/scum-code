"""The OpenSerial interface is used to interface with an OpenWSN network."""

from absl import logging

from openwsn.hdlc import Hdlc
from openwsn.hdlc_aggregator import HdlcAggregator
from utils.serial.serial_interface import SerialInterface

# OpenWSN constants.
OPENWSN_IPV6_NETWORK_PREFIX = b"\xbb\xbb\x00\x00\x00\x00\x00\x00"
OPENWSN_NETWORK_KEY_INDEX = b"\x01"
OPENWSN_NETWORK_KEY = b"\xde\xad\xbe\xef\xca\xfe\xde\xad\xbe\xef\xca\xfe\xde\xad\xbe\xef"

# OpenSerial XON/XOFF flow control constants.
OPENSERIAL_XOFF_CHAR = b"\x13"
OPENSERIAL_XON_CHAR = b"\x11"
OPENSERIAL_XONXOFF_ESCAPE_CHAR = b"\x12"
OPENSERIAL_XONXOFF_CHAR_MASK = b"\x10"


class OpenSerialFrameType:
    """OpenSerial frame type enumeration."""
    MOTE2PC_DATA = b"D"
    MOTE2PC_STATUS = b"S"
    MOTE2PC_PACKET = b"P"
    MOTE2PC_PRINTF = b"F"
    PC2MOTE_SET_DAG_ROOT = b"R"
    PC2MOTE_DATA = b"D"
    PC2MOTE_TRIGGER_SERIAL_ECHO = b"S"
    PC2MOTE_COMMAND = b"C"


class OpenSerialFrameAction:
    """OpenSerial frame action enumeration."""
    YES = b"Y"
    NO = b"N"
    TOGGLE = b"T"


class OpenSerialInterface:
    """Interface to an OpenWSN network.

    The OpenWSN's serial interface uses software flow control (XON/XOFF flow
    control).

    Attributes:
        serial: Serial interface.
        received_xonxoff_escape: If true, received an XON/XOFF escape
          character.
    """

    def __init__(self, port: str, baudrate: int) -> None:
        # Open the serial port.
        self.serial = SerialInterface(port, baudrate, xonxoff=True)
        self.received_xonxoff_escape = False

    def set_dag_root(self) -> None:
        """Sets the device as the DAG root."""
        logging.info("Setting %s as the DAG root.", self.serial.port)
        buffer = b"".join([
            OpenSerialFrameType.PC2MOTE_SET_DAG_ROOT,
            OpenSerialFrameAction.YES,
            OPENWSN_IPV6_NETWORK_PREFIX,
            OPENWSN_NETWORK_KEY_INDEX,
            OPENWSN_NETWORK_KEY,
        ])
        self._send_data(buffer)

    def receive_data(self) -> bytearray:
        """Receives the data through an HDLC frame.

        Returns:
            The raw data within the HDLC frame.
        """
        aggregator = HdlcAggregator()
        while not aggregator.has_received_data():
            byte = self.serial.read(num_bytes=1)

            # Escape XON/XOFF characters.
            escaped_byte = self._escape_xonxoff(byte)
            aggregator.receive_byte(escaped_byte)
        return aggregator.buffer

    def _send_data(self, buffer: bytearray) -> None:
        """Sends data to a device via HDLC.

        Args:
            buffer: Raw data to send.
        """
        frame = Hdlc.hdlcify(buffer)
        self.serial.write(frame)

    def _escape_xonxoff(self, byte: bytes) -> bytes:
        """Escapes XON/XOFF characters.

        Args:
            byte: Received byte to escape.

        Returns:
            The escaped byte. If the received byte is the XON/XOFF escape
            character, returns an empty bytes object.
        """
        if byte == OPENSERIAL_XONXOFF_ESCAPE_CHAR:
            self.received_xonxoff_escape = True
            return b""

        if self.received_xonxoff_escape:
            self.received_xonxoff_escape = False
            return bytes(
                a ^ b for a, b in zip(byte, OPENSERIAL_XONXOFF_CHAR_MASK))
        if byte != OPENSERIAL_XON_CHAR and byte != OPENSERIAL_XOFF_CHAR:
            return byte
        return b""
