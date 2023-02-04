from absl import app, flags

from utils.serial import serial_monitor

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    monitor = serial_monitor.SerialMonitor(FLAGS.port, FLAGS.baudrate)
    monitor.run()


if __name__ == "__main__":
    flags.DEFINE_string("port", "/dev/cu.usbserial-A10M1IFE",
                        "Serial port to log.")
    flags.DEFINE_integer("baudrate", 19200, "Serial port baud rate.")

    app.run(main)
