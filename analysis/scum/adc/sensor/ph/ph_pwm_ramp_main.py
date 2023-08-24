import matplotlib.pyplot as plt
import pandas as pd
from absl import app, flags, logging

from analysis.scum.adc.sensor.ph.ph_pwm_ramp_data import PhPwmRampData

FLAGS = flags.FLAGS


def analyze_pwm_ramp(data: str, sampling_rate: float) -> None:
    """Processes the pH sensor data when excited by a PWM ramp.

    Args:
        data: Data filename.
        sampling_rate: Sampling rate in Hz.
    """
    # Open the data file.
    df = pd.read_csv(data, comment="#")
    logging.info(df.describe())

    # Process the PWM ramp data.
    pwm_ramp = PhPwmRampData(df, sampling_rate)

    # Plot the original and filtered output data.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(pwm_ramp.pwm_time, pwm_ramp.pwm_output, label="Raw output")
    ax.plot(pwm_ramp.pwm_time,
            pwm_ramp.filtered_pwm_output,
            label="Filtered output")
    secax = ax.twinx()
    secax.plot(pwm_ramp.pwm_time,
               pwm_ramp.pwm_ramp,
               color="C2",
               label="PWM ramp")
    ax.set_title("Output filtering")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Output [V]")
    ax.legend()
    secax.set_ylabel("Ramp [V]")
    secax.legend(loc="lower right")
    plt.show()

    # Plot the first and second differences of the filtered output data.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(pwm_ramp.pwm_time[:-1],
            pwm_ramp.filtered_pwm_output_diff,
            label="Filtered output diff")
    secax = ax.twinx()
    secax.plot(pwm_ramp.pwm_time,
               pwm_ramp.pwm_ramp,
               color="C2",
               label="PWM ramp")
    ax.set_title("Filtered output difference")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Output difference [V]")
    ax.legend()
    secax.set_ylabel("Ramp [V]")
    secax.legend(loc="lower right")
    plt.show()


def main(argv):
    assert len(argv) == 1
    analyze_pwm_ramp(FLAGS.data, FLAGS.sampling_rate)


if __name__ == "__main__":
    flags.DEFINE_string(
        "data",
        "analysis/scum/adc/sensor/ph/data/ph5_3.csv",
        "Data filename.",
    )
    flags.DEFINE_float("sampling_rate", 1250, "Sampling rate in Hz.")

    app.run(main)
