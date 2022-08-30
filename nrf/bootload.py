from absl import app, flags, logging
import random
import serial

FLAGS = flags.FLAGS
BINARY_SIZE = 64 * 1024

def bootload_nrf(port: str, binary: str, use_random_padding: bool = False) -> None:
    """Bootloads the nRF board.

    Args:
        port: Serial port of the nRF board.
        binary: Binary image to flash onto the nRF board.
        use_random_padding: If true, pad the binary with random bytes.
    """
    # Open the serial port to the nRF board.
    nrf_serial = serial.Serial(
        port=port,
        baudrate=250000,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS)

    with open(binary, "rb") as f:
        data = f.read(BINARY_SIZE)

    # Pad the binary up to the binary size.
    if use_random_padding:
        data += random.randbytes(BINARY_SIZE - len(data))
    else:
        data += b"\0" * (BINARY_SIZE - len(data))

    # Transfer the data to the nRF board.
    # nrf_serial.write(b"transfersram\n")
    # logging.info(nrf_serial.read_until())

    # Send the binary data over UART.
    nrf_serial.write(data)
    logging.info(nrf_serial.read_until())

    # Execute the 3-wire bus bootloader on the nRF board.
    nrf_serial.write(b"boot3wb\n")
    logging.info(nrf_serial.read_until())

def main(argv):
    assert len(argv) == 1
    bootload_nrf(FLAGS.port, FLAGS.binary, FLAGS.use_random_padding)

if __name__ == "__main__":
    flags.DEFINE_string("port", "/dev/cu.usbmodem0006839038541", "Serial port of the nRF board.")
    flags.DEFINE_string("binary", None, "Binary image to flash onto the nRF board.")
    flags.DEFINE_bool("use_random_padding", False, "If true, pad the binary with random bytes.")
    flags.mark_flag_as_required("binary")

    app.run(main)
