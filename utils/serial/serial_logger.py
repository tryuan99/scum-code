"""The serial logger monitors a serial port and logs all data received by the serial port to a file."""

from absl import logging

from utils.serial.serial_interface import SerialInterface


class SerialLogger:
    """Serial logger to log data received by the serial port."""

    def __init__(self, port: str, baudrate: int, output_file: str,
                 log_to_stderr: bool):
        # Open the serial port.
        self.serial = SerialInterface(port, baudrate)
        self.output_file = output_file
        self.log_to_stderr = log_to_stderr

    def run(self) -> None:
        """Logs data received by the serial port."""
        with open(self.output_file, "wb") as f:
            while True:
                read_data = self.serial.read()
                f.write(read_data)
                if self.log_to_stderr:
                    try:
                        logging.info(read_data.decode().strip())
                    except:
                        pass
