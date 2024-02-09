import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from absl import app, flags

FLAGS = flags.FLAGS


def plot_digital_clock_spectrum(measured_data: str) -> None:
    """Plots the digital clock spectrum.

    Args:
        measured_data: Measured data filename.
    """
    df_measured = pd.read_csv(measured_data, comment="#")
    frequency_column, power_column = df_measured.columns

    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the measured data.
    df_measured.plot.line(ax=ax, x=frequency_column, linewidth=2)

    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dBm]")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_digital_clock_spectrum(FLAGS.digital_clock_spectrum_measured_data)


if __name__ == "__main__":
    flags.DEFINE_string(
        "digital_clock_spectrum_measured_data",
        "tapeout/clocks/data/digital_clock_spectrum_measured.csv",
        "Digital clock spectrum measured data.")

    app.run(main)
