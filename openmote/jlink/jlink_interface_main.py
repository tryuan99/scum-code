from absl import app, flags

from openmote.jlink.jlink_interface import JLinkInterface

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    jlink_interface = JLinkInterface(FLAGS.verbose)
    jlink_interface.open()


if __name__ == "__main__":
    flags.DEFINE_boolean("verbose", True, "If true, log verbose messages.")

    app.run(main)
