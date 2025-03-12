from absl import app, flags, logging

from utils.plotter import serial_live_plotter

FLAGS = flags.FLAGS


def _parse_adc_data(read_data: str, num_sensors: int) -> list[float]:
    """Parses the ADC data serial output.

    Args:
        read_data: Read data from the serial port.
        num_sensors: Number of sensors to read.

    Returns:
        The list of ADC outputs in LSBs.
    """
    try:
        # The OpenMote prints (sequence number, channel, coarse code, medium
        # code, fine code, ADC output for each sensor, RSSI).
        adc_output = read_data.split(" ")[5:5 + num_sensors]
        adc_output_floats = [float(data) for data in adc_output]
        return adc_output_floats
    except:
        logging.error("Failed to parse ADC data.")
        return [0] * num_sensors


def main(argv):
    assert len(argv) == 1

    plotter = serial_live_plotter.SerialLivePlotter(
        FLAGS.port,
        FLAGS.baudrate,
        FLAGS.max_duration,
        lambda read_data: _parse_adc_data(read_data, FLAGS.num_sensors),
        trace_labels=[
            "Decomposition sensor",
            "Nitrate sensor",
            "Ammonium sensor",
            "Thermistor",
            "Oxygen sensor",
        ],
        num_traces=FLAGS.num_sensors,
        title="ADC data",
        xlabel="Time [s]",
        ylabel="ADC output [LSB]",
        ymin=0,
        ymax=512,
    )
    plotter.run()


if __name__ == "__main__":
    flags.DEFINE_string("port", "/dev/tty.usbserial-101",
                        "Serial port to plot for.")
    flags.DEFINE_integer(
        "baudrate",
        115200,
        "Serial port baud rate.",
        lower_bound=0,
    )
    flags.DEFINE_integer(
        "max_duration",
        30,
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
