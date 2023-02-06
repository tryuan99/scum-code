"""The nRF bootloader flashes a binary onto SCuM and boots it using the 3-wire bus."""

import random

from absl import logging

from utils.serial import serial_interface

# Baud rate of the nRF board.
NRF_BAUDRATE = 250000

# Binary size in bytes to be flashed onto SCuM.
SCUM_BINARY_SIZE = 64 * 1024


class NrfBootloader:
    """SCuM bootloader using the nRF board."""

    def __init__(self, port: str):
        # Open the serial port to the nRF board.
        self.serial = serial_interface.SerialInterface(port, NRF_BAUDRATE)

    def bootload(self, binary: str, use_random_padding: bool) -> None:
        """Bootloads SCuM using the nRF board.

        Args:
            binary: Binary image to flash onto SCuM.
            use_random_padding: If true, pad the binary with random bytes.
        """
        with open(binary, "rb") as f:
            data = f.read(SCUM_BINARY_SIZE)

        # Pad the binary up to the binary size.
        if use_random_padding:
            data += random.randbytes(SCUM_BINARY_SIZE - len(data))
        else:
            data += bytes(SCUM_BINARY_SIZE - len(data))

        # Send the binary data over UART.
        logging.info("Sending the SCuM binary to the nRF board.")
        self.serial.write(data)
        # Read the response that the SRAM load is complete.
        logging.info(self.serial.read())
        # Read the response that the 3-wire bus bootload is complete.
        logging.info(self.serial.read())
