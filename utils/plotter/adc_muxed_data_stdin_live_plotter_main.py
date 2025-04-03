import re

from absl import app, flags

from utils.plotter import stdin_live_plotter

FLAGS = flags.FLAGS

# Regex pattern.
PATTERN = (
    r"received sensor network packet from (\w+) on channel (\d+), sequence "
    r"number (\d+):")

# Previous sequence number.
previous_sequence_number = None


def _parse_adc_data(read_data: str, source_address: str,
                    num_sensors: int) -> list[float]:
    """Parses the ADC data standard input.

    Args:
        read_data: Read data from standard input.
        source_address: Source address.
        num_sensors: Number of sensors to read.

    Returns:
        The list of ADC outputs in LSBs.
    """
    global previous_sequence_number

    # The OpenVisualizer prints the 16-bit source address, channel, sequence
    # number, and ADC output for each sensor.
    pattern_with_sensor_output = PATTERN + r" (\d+)" * num_sensors
    match = re.search(pattern_with_sensor_output, read_data)
    if match is not None:
        # Filter for the source address.
        address = match.group(1)
        if source_address is not None and address != source_address:
            return 0
        channel = match.group(2)
        # Check the sequence number.
        sequence_number = match.group(3)
        if sequence_number == previous_sequence_number:
            return 0
        previous_sequence_number = sequence_number
        adc_output_floats = [
            float(match.group(i + 4)) for i in range(num_sensors)
        ]
        return adc_output_floats
    return 0


def main(argv):
    assert len(argv) == 1

    plotter = stdin_live_plotter.StdinLivePlotter(
        FLAGS.max_duration,
        lambda read_data: _parse_adc_data(read_data, FLAGS.address, FLAGS.
                                          num_sensors),
        num_traces=FLAGS.num_sensors,
        trace_labels=[
            "Decomposition sensor",
            "Nitrate sensor",
            "Ammonium sensor",
            "Thermistor",
            "Oxygen sensor",
        ],
        title="ADC data",
        xlabel="Time [s]",
        ylabel="ADC output [LSB]",
        ymin=0,
        ymax=600,
    )
    plotter.run()


if __name__ == "__main__":
    flags.DEFINE_string("address", None, "Source address.")
    flags.DEFINE_integer(
        "max_duration",
        1000,
        "Maximum duration to plot in seconds.",
        lower_bound=0,
    )
    flags.DEFINE_integer(
        "num_sensors",
        8,
        "Number of sensors to read.",
        lower_bound=0,
    )

    app.run(main)
