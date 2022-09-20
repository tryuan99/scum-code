from absl import app, flags

from scum.utils import nrf_bootloader

FLAGS = flags.FLAGS

def main(argv):
    assert len(argv) == 1

    bootloader = nrf_bootloader.NrfBootloader(FLAGS.port)
    bootloader.bootload(FLAGS.binary, FLAGS.use_random_padding)

if __name__ == "__main__":
    flags.DEFINE_string("port", "/dev/cu.usbmodem0006839038541", "Serial port of the nRF board.")
    flags.DEFINE_string("binary", None, "Binary image to flash onto the nRF board.")
    flags.DEFINE_boolean("use_random_padding", False, "If true, pad the binary with random bytes.")
    flags.mark_flag_as_required("binary")

    app.run(main)
