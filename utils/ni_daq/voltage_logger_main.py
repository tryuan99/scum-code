from absl import app, flags

from utils.ni_daq import voltage_logger

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    logger = voltage_logger.VoltageLogger(
        FLAGS.channel,
        FLAGS.sampling_rate,
        FLAGS.time_to_sample,
        FLAGS.output_file,
        FLAGS.log_to_stderr,
    )
    logger.run()


if __name__ == "__main__":
    flags.DEFINE_string("channel", "ai1", "NI DAQ channel(s) to log.")
    flags.DEFINE_integer("sampling_rate", 250000, "Sampling rate in Hz.", lower_bound=0)
    flags.DEFINE_float("time_to_sample", 2, "Time to sample in seconds.", lower_bound=0)
    flags.DEFINE_string("output_file", None, "Output file to log the data to.")
    flags.DEFINE_boolean(
        "log_to_stderr",
        True,
        "If true, log the received data to standard error in addition to the output file.",
    )
    flags.mark_flag_as_required("output_file")

    app.run(main)
