import numpy as np
import pandas as pd
from absl import app, flags, logging

FLAGS = flags.FLAGS

# GPIO high threshold in volts.
GPIO_HIGH_THRESHOLD = 0.2  # V

# ADC-to-GPIO offset in seconds. This number was empirically determined.
ADC_TO_GPIO_OFFSET = 29e-6  # seconds


def cleanup_scope_data(data: str, output: str,
                       scope_sampling_rate: int) -> None:
    """Cleans up the scope data and outputs the result to another CSV file.

    Args:
        data: Data filename.
        output: Output filename.
        scope_sampling_rate: Oscilloscope sampling rate.
    """
    # Open the scope data file.
    df = pd.read_csv(data,
                     header=None,
                     names=("Time [s]", "GPIO [V]", "Input voltage [V]"))
    time_column, gpio_column, input_column = df.columns

    gpio_data = df[gpio_column].to_numpy()
    thresholded_gpio_data = (gpio_data > GPIO_HIGH_THRESHOLD).astype(int)
    (positive_edge_indices,) = np.where(np.diff(thresholded_gpio_data) > 0.5)
    positive_edge_data = df.loc[positive_edge_indices -
                                int(scope_sampling_rate * ADC_TO_GPIO_OFFSET)]
    positive_edge_data.to_csv(output, index=False)


def main(argv):
    assert len(argv) == 1
    cleanup_scope_data(FLAGS.data, FLAGS.output, FLAGS.scope_sampling_rate)


if __name__ == "__main__":
    flags.DEFINE_string("data", None, "Data filename.")
    flags.DEFINE_string("output", None, "Output filename.")
    flags.DEFINE_integer("scope_sampling_rate", None,
                         "Oscilloscope sampling rate.")
    flags.mark_flags_as_required(["data", "output", "scope_sampling_rate"])

    app.run(main)
