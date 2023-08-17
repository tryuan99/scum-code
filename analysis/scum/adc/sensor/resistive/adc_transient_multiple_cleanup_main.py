import pandas as pd
from absl import app, flags, logging

FLAGS = flags.FLAGS

# Iteration column name.
ITERATION_COLUMN = "Iteration"

# ADC output column name.
ADC_OUTPUT_COLUMN = "ADC output [LSB]"

# Maximum ADC output in LSBs.
MAX_ADC_OUTPUT = 511  # LSBs

# Number of consecutive maximum ADC samples to mark the next iteration.
NUM_MAX_ADC_SAMPLES_FOR_NEXT_ITERATION = 20

# Decay start threshold in LSBs.
DECAY_START_THRESHOLD = 500  # LSBs


def _find_offset_to_next_iteration(adc_data: pd.Series) -> int:
    """Finds the offset to the next iteration.

    The next iteration when the capacitor is charged, i.e., when the ADC
    samples discontinuously jump to the maximum value.

    Args:
        adc_data: ADC data.

    Returns:
        The offset to the next iteration.
    """
    for i in range(len(adc_data) - NUM_MAX_ADC_SAMPLES_FOR_NEXT_ITERATION + 1):
        if (adc_data.iloc[i:i + NUM_MAX_ADC_SAMPLES_FOR_NEXT_ITERATION] ==
                MAX_ADC_OUTPUT).all():
            return i
    return len(adc_data)


def _find_offset_to_next_decay(adc_data: pd.Series) -> int:
    """Finds the offset to the start of the exponential decay.

    The decay starts at the sample that is less than the maximum value. If this
    value is less than the start threshold, include the previous sample as the
    first sample of the decay.

    Args;
        adc_data: ADC data.

    Returns:
        The offset to the start of the exponential decay.
    """
    for i in range(len(adc_data)):
        if adc_data.iloc[i] < MAX_ADC_OUTPUT:
            if adc_data.iloc[i] < DECAY_START_THRESHOLD:
                return max(i - 1, 0)
            else:
                return i
    return i


def cleanup_multiple_transient_adc_data(data: str, output: str) -> None:
    """Cleans up the ADC data corresponding to multiple decaying exponentials.

    The data file should contain a single column corresponding to the ADC samples.
    The output file will consist of two columns, one for the iteration and one
    for the corresponding ADC data after removing saturated samples.

    Args:
        data: Data filename.
        output: Output filename.
    """
    # Open the ADC data file.
    adc_data = pd.read_csv(data, header=None, names=(ADC_OUTPUT_COLUMN,))
    adc_data.insert(0, ITERATION_COLUMN, 0)

    # Split the ADC data into multiple iterations.
    iteration = 1
    current_index = _find_offset_to_next_iteration(adc_data[ADC_OUTPUT_COLUMN])
    while current_index < len(adc_data):
        offset_to_next_decay_start = _find_offset_to_next_decay(
            adc_data.loc[current_index:, ADC_OUTPUT_COLUMN])
        current_index += offset_to_next_decay_start
        if current_index >= len(adc_data):
            break

        offset_to_next_iteration = _find_offset_to_next_iteration(
            adc_data.loc[current_index:, ADC_OUTPUT_COLUMN])
        adc_data.loc[current_index:current_index + offset_to_next_iteration - 1,
                     ITERATION_COLUMN] = iteration
        current_index += offset_to_next_iteration
        iteration += 1

    # Remove saturated ADC samples.
    cleaned_adc_data = adc_data[adc_data[ITERATION_COLUMN] != 0]
    cleaned_adc_data.to_csv(output, index=False)
    logging.info("Found %d iterations.", iteration - 1)


def main(argv):
    assert len(argv) == 1
    cleanup_multiple_transient_adc_data(FLAGS.data, FLAGS.output)


if __name__ == "__main__":
    flags.DEFINE_string("data", None, "Data filename.")
    flags.DEFINE_string("output", None, "Output filename.")
    flags.mark_flags_as_required(["data", "output"])

    app.run(main)
