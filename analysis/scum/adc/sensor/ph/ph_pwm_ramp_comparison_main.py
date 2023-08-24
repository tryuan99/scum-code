import matplotlib.pyplot as plt
import pandas as pd
from absl import app, flags

from analysis.scum.adc.sensor.ph.ph_pwm_ramp_data import PhPwmRampData

FLAGS = flags.FLAGS


def compare_pwm_ramps(data: list[str], sampling_rate: float) -> None:
    """Compares the pH sensor data when excited by a PWM ramp.

    Args:
        data: List of data filenames.
        sampling_rate: Sampling rate in Hz.
    """
    # Plot the filtered PWM outputs.
    fig, ax = plt.subplots(figsize=(12, 8))
    for file_index, filename in enumerate(data):
        df = pd.read_csv(filename, comment="#")
        pwm_ramp = PhPwmRampData(df, sampling_rate)

        ax.plot(pwm_ramp.pwm_ramp, pwm_ramp.filtered_pwm_output, label=filename)
    ax.set_title("Filtered PWM output")
    ax.set_xlabel("PWM ramp [V]")
    ax.set_ylabel("PWM output [V]")
    plt.legend()
    plt.show()

    # Plot the first difference of the filtered PWM outputs.
    fig, ax = plt.subplots(figsize=(12, 8))
    for file_index, filename in enumerate(data):
        df = pd.read_csv(filename, comment="#")
        pwm_ramp = PhPwmRampData(df, sampling_rate)

        ax.plot(pwm_ramp.pwm_ramp[:-1],
                pwm_ramp.filtered_pwm_output_diff,
                label=filename)
    ax.set_title("Filtered PWM output difference")
    ax.set_xlabel("PWM ramp [V]")
    ax.set_ylabel("PWM output difference [V]")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1
    compare_pwm_ramps(FLAGS.data, FLAGS.sampling_rate)


if __name__ == "__main__":
    flags.DEFINE_list(
        "data",
        [
            "analysis/scum/adc/sensor/ph/data/ph4_3.csv",
            "analysis/scum/adc/sensor/ph/data/ph5_3.csv",
            "analysis/scum/adc/sensor/ph/data/ph7_3.csv",
        ],
        "Data filenames.",
    )
    flags.DEFINE_float("sampling_rate", 1250, "Sampling rate in Hz.")

    app.run(main)
