"""The serial live plotter is used to plot live data from a serial port."""

from absl import logging
from utils.serial import serial_interface
from utils.plotter.live_plotter import ContinuousLivePlotter
from collections.abc import Callable


class SerialLivePlotter(ContinuousLivePlotter):
    """Serial live plotter to plot data from a serial port."""

    def __init__(self, port: str, baudrate: int, max_duration: float,
                 parse_data: Callable[[str], float], title: str, xlabel: str,
                 ylabel: str, ymin: float, ymax: float):
        super().__init__(max_duration, title, xlabel, ylabel, ymin, ymax)
        self.parse_data = parse_data

        # Open the serial port.
        self.serial = serial_interface.SerialInterface(port, baudrate)

    def next(self) -> float:
        """Returns the next data to be plotted."""
        read_data = bytes()
        while len(read_data) == 0:
            read_data = self.serial.read()
            try:
                read_data = read_data.decode().strip()
            except:
                logging.error("Failed to decode read data.")
                read_data = bytes()
        return self.parse_data(read_data)
