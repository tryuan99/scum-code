import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from absl import app, flags

FLAGS = flags.FLAGS

# Number of tuning bits for the analog clock.
ANALOG_CLOCK_NUM_TUNING_BITS = 16


def plot_analog_clock_tuning(measured_data: str) -> None:
    """Plots the analog clock frequency as a function of its tuning code.

    Args:
        measured_data: Measured data filename.
    """
    df_measured = pd.read_csv(measured_data, comment="#")
    tuning_code_column, frequency_column = df_measured.columns

    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the measured data.
    df_measured.plot.line(ax=ax, x=tuning_code_column, linewidth=2, marker="^")

    plt.xlabel(f"Analog tuning code ({ANALOG_CLOCK_NUM_TUNING_BITS} bits)")
    plt.ylabel("Frequency [MHz]")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_analog_clock_tuning(FLAGS.analog_clock_measured_data)


if __name__ == "__main__":
    flags.DEFINE_string("analog_clock_measured_data",
                        "tapeout/clocks/data/analog_clock_measured.csv",
                        "Analog clock measured data.")

    app.run(main)
