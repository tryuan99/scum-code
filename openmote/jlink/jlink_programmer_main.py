from absl import app, flags

from openmote.jlink.jlink_programmer import JLinkProgrammer

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    jlink_programmer = JLinkProgrammer(FLAGS.verbose)
    jlink_programmer.program(FLAGS.bin_file)


if __name__ == "__main__":
    flags.DEFINE_string("bin_file", None, "Binary file to program.")
    flags.DEFINE_boolean("verbose", True, "If true, log verbose messages.")
    flags.mark_flag_as_required("bin_file")

    app.run(main)
