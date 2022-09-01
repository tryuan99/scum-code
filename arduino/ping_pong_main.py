from absl import app, flags, logging
import serial
import time

FLAGS = flags.FLAGS
PACKET_SIZE = 32

def ping_pong(port: str, num_bytes: int) -> None:
    """Continuously sends data to the Arduino and reads the responses.

    Args:
        port: Serial port of the Arduino.
    """
    # Open the serial port to the Arduino.
    arduino_serial = serial.Serial(
        port=port,
        baudrate=250000,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS)

    # TODO(tyuan): Figure out why this sleep is necessary for the writes to be successful.
    time.sleep(2)

    def write_to_arduino(data: bytes) -> None:
        if len(data) <= 0:
            return
        num_bytes_written = 0
        while num_bytes_written < len(data):
            num_bytes_to_write = min(PACKET_SIZE, len(data) - num_bytes_written)
            num_bytes_sent = arduino_serial.write(data[num_bytes_written:num_bytes_written + num_bytes_to_write])
            logging.info("Wrote %d bytes.", num_bytes_sent)
            num_bytes_written += num_bytes_sent
            time.sleep(0.001)

    data = b"A" * (num_bytes - 1) + b"\n"
    while True:
        write_to_arduino(data)
        logging.info(arduino_serial.read_until())
        time.sleep(0.5)

def main(argv):
    assert len(argv) == 1
    ping_pong(FLAGS.port, FLAGS.num_bytes)

if __name__ == "__main__":
    flags.DEFINE_string("port", "/dev/cu.usbmodem11401", "Serial port of the Arduino.")
    flags.DEFINE_integer("num_bytes", 256, "Number of bytes to send to the Arduino.", lower_bound=0)

    app.run(main)
