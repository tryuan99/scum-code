from absl import app, flags, logging

from openwsn.openserial_interface import (OpenSerialFrameType,
                                          OpenSerialInterface)
from utils.struct import Struct, StructFieldType

FLAGS = flags.FLAGS


class OpenSerialStruct(Struct):
    """OpenSerial struct."""

    @property
    def fields(self) -> dict[str, (StructFieldType, int)]:
        """Returns a dictionary mapping each field name to its size in bytes
        and the array length.
        """
        return {
            "type": (StructFieldType.CHAR, 1),
        }


def main(argv):
    assert len(argv) == 1

    openserial_interface = OpenSerialInterface(FLAGS.port, FLAGS.baudrate)
    openserial_interface.set_dag_root()
    while True:
        data = openserial_interface.receive_data()
        struct = OpenSerialStruct(data)
        if struct.get("type") == OpenSerialFrameType.MOTE2PC_DATA:
            logging.info("Received data: %s", data)


if __name__ == "__main__":
    flags.DEFINE_string("port", "/dev/tty.usbserial-1101",
                        "Serial port to log.")
    flags.DEFINE_integer("baudrate", 115200, "Serial port baud rate.")

    app.run(main)
