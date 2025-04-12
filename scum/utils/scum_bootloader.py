"""The SCuM bootloader flashes a binary onto SCuM and boots it using the 3-wire bus."""

import random

from absl import logging

from utils.serial.serial_interface import SerialInterface

# Serial port baud rate.
SCUM_BOOTLOADER_BAUDRATE = 460800

# Binary size in bytes to be flashed onto SCuM.
SCUM_BINARY_SIZE = 64 * 1024


class ScumBootloader:
    """SCuM bootloader.

    Attributes:
        serial: Serial interface.
    """

    def __init__(
        self,
        port: str,
        baudrate: int = SCUM_BOOTLOADER_BAUDRATE,
    ) -> None:
        # Open the serial port.
        self.serial = SerialInterface(port, baudrate)

    def bootload(
        self,
        binary: str,
        use_random_padding: bool,
        start_serial_monitor: bool,
    ) -> None:
        """Bootloads SCuM.

        Args:
            binary: Binary image to flash onto SCuM.
            use_random_padding: If true, pad the binary with random bytes.
            start_serial_monitor: If true, start a serial monitor.
        """
        with open(binary, "rb") as f:
            data = f.read(SCUM_BINARY_SIZE)

        # Pad the binary up to the binary size.
        if use_random_padding:
            data += random.randbytes(SCUM_BINARY_SIZE - len(data))
        else:
            data += bytes(SCUM_BINARY_SIZE - len(data))

        # Boot SCuM.
        logging.info("Bootloading SCuM.")
        self.serial.write(data)
        # Read the response that the SRAM load is complete.
        logging.info("Firmware load response: %s", self.serial.read())
        # Read the response that the 3-wire bus bootload is complete.
        logging.info("3WB bootload response: %s", self.serial.read())
        # Read the response that the clock calibration is complete.
        logging.info("Clock calibration response: %s", self.serial.read())

        # Start a serial monitor.
        if start_serial_monitor:
            while True:
                read_data = self.serial.read()
                try:
                    logging.info(read_data.decode().strip())
                except:
                    logging.info(read_data)
