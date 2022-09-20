from absl import app, flags, logging
import time

from utils.serial import serial_interface

FLAGS = flags.FLAGS

def ping_pong(port: str, baudrate: int, num_bytes: int) -> None:
    """Continuously sends data to the Arduino and reads the responses.

    Args:
        port: Serial port of the Arduino.
        baudrate: Baud rate of the Arduino.
        num_bytes: Number of bytes to send every time.
    """
    # Open the serial port to the Arduino.
    arduino_serial = serial_interface.SerialInterface(port, baudrate)

    data = b"A" * (num_bytes - 1) + b"\n"
    while True:
        arduino_serial.write(data)
        logging.info(arduino_serial.read())
        time.sleep(0.5)

def main(argv):
    assert len(argv) == 1
    ping_pong(FLAGS.port, FLAGS.baudrate, FLAGS.num_bytes)

if __name__ == "__main__":
    flags.DEFINE_string("port", "/dev/cu.usbmodem11401", "Serial port of the Arduino.")
    flags.DEFINE_integer("baudrate", 250000, "Baud rate of the serial port.", lower_bound=0)
    flags.DEFINE_integer("num_bytes", 256, "Number of bytes to send to the Arduino.", lower_bound=1)

    app.run(main)
