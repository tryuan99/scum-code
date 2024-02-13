import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from absl import app, flags

FLAGS = flags.FLAGS

# Number of tuning bits for the analog clock.
ANALOG_CLOCK_NUM_TUNING_BITS = 16


def plot_analog_clock_tuning(measured_data: str, simulated_data: str) -> None:
    """Plots the analog clock frequency as a function of its tuning code.

    Args:
        measured_data: Measured data filename.
        simulated_data: Simulated data filename.
    """
    df_measured = pd.read_csv(measured_data, comment="#")
    df_simulated = pd.read_csv(simulated_data, comment="#")

    plt.style.use(["science", "grid"])
    plt.rcParams.update({
        "font.size": 16,
        "lines.linewidth": 3,
        "lines.markersize": 8,
    })
    fig, ax = plt.subplots(figsize=(12, 3))

    # Plot the measured data.
    df_measured.plot.line(ax=ax,
                          x=df_measured.columns[0],
                          linewidth=2,
                          marker="^")

    # Plot the simulated data.
    # df_simulated.plot.line(ax=ax, x=df_simulated.columns[0], linestyle="--")

    plt.xlabel(f"ADC clock tuning code ({ANALOG_CLOCK_NUM_TUNING_BITS} bits)")
    plt.ylabel("Clock frequency [MHz]")
    plt.legend(["ADC clock frequency"])
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_analog_clock_tuning(FLAGS.analog_clock_measured_data,
                             FLAGS.analog_clock_simulated_data)


if __name__ == "__main__":
    flags.DEFINE_string("analog_clock_measured_data",
                        "tapeout/clocks/data/analog_clock_measured.csv",
                        "Analog clock measured data.")
    flags.DEFINE_string(
        "analog_clock_simulated_data",
        "tapeout/clocks/data/analog_clock_schematic_simulated.csv",
        "Analog clock simulated data.")

    app.run(main)
