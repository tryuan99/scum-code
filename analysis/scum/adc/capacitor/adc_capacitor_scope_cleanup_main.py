from absl import app, flags, logging
import numpy as np
import pandas as pd

FLAGS = flags.FLAGS

# GPIO high threshold in volts.
GPIO_HIGH_THRESHOLD = 0.2  # V

# ADC-to-GPIO offset in seconds. This number was empirically determined.
ADC_TO_GPIO_OFFSET = 29e-6  # seconds


def cleanup_scope_data(
    data: str,
    data_without_scum: str,
    output: str,
    output_without_scum: str,
    scope_sampling_rate: int,
) -> None:
    """Cleans up the scope data and outputs the result to another CSV file.

    Args:
        data: Data filename.
        data: Data filename.
        output: Output filename.
        output: Output filename.
        scope_sampling_rate: Oscilloscope sampling rate.
    """
    # Open the scope data file.
    df = pd.read_csv(
        data, header=None, names=("Time [s]", "Capacitor voltage [V]", "GPIO [V]"),
    )
    time_column, capacitor_column, gpio_column = df.columns
    df = df[df[time_column] >= 0].reset_index(drop=True)

    # Open the scope data file without SCuM.
    df_without_scum = pd.read_csv(
        data_without_scum, header=None, names=("Time [s]", "Capacitor voltage [V]"),
    )
    time_column_without_scum, capacitor_column_without_scum = df_without_scum.columns
    df_without_scum = df_without_scum[
        df_without_scum[time_column_without_scum] >= 0
    ].reset_index(drop=True)
    gpio_data = df[gpio_column].to_numpy()
    thresholded_gpio_data = (gpio_data > GPIO_HIGH_THRESHOLD).astype(int)

    num_samples_offset = int(scope_sampling_rate * ADC_TO_GPIO_OFFSET)
    positive_edge_indices = np.where(np.diff(thresholded_gpio_data) > 0.5)[0]
    positive_edge_indices = positive_edge_indices[
        positive_edge_indices > num_samples_offset
    ]
    positive_edge_data = df.loc[positive_edge_indices - num_samples_offset]
    positive_edge_data.to_csv(output, index=False)
    positive_edge_data_without_scum = df_without_scum.loc[
        positive_edge_indices - num_samples_offset
    ]
    positive_edge_data_without_scum.to_csv(output_without_scum, index=False)


def main(argv):
    assert len(argv) == 1
    cleanup_scope_data(
        FLAGS.data,
        FLAGS.data_without_scum,
        FLAGS.output,
        FLAGS.output_without_scum,
        FLAGS.scope_sampling_rate,
    )


if __name__ == "__main__":
    flags.DEFINE_string("data", None, "Data filename.")
    flags.DEFINE_string("data_without_scum", None, "Data filename.")
    flags.DEFINE_string("output", None, "Output filename.")
    flags.DEFINE_string("output_without_scum", None, "Output filename.")
    flags.DEFINE_integer("scope_sampling_rate", None, "Oscilloscope sampling rate.")
    flags.mark_flags_as_required(
        [
            "data",
            "data_without_scum",
            "output",
            "output_without_scum",
            "scope_sampling_rate",
        ]
    )

    app.run(main)
