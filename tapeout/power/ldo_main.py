import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from absl import app, flags

FLAGS = flags.FLAGS


def plot_ldo(measured_data: str, simulated_data: str) -> None:
    """Plots LDO voltage as a function of its current.

    Args:
        measured_data: Measured data filename.
        simulated_data: Simulated data filename.
    """
    df_measured = pd.read_csv(measured_data, comment="#")
    df_simulated = pd.read_csv(simulated_data, comment="#")
    current_column_measured = df_measured.columns[0]
    current_column_simulated = df_simulated.columns[0]
    max_current_to_plot = min(df_measured[current_column_measured].max(),
                              df_simulated[current_column_simulated].max())

    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the measured data.
    rows_to_plot_measured = df_measured[
        current_column_measured] <= max_current_to_plot
    df_measured[rows_to_plot_measured].plot.line(ax=ax,
                                                 x=current_column_measured,
                                                 linewidth=2)

    # Plot the simulated data.
    rows_to_plot_simulated = df_simulated[
        current_column_simulated] <= max_current_to_plot
    df_simulated[rows_to_plot_simulated].plot.line(ax=ax,
                                                   x=current_column_simulated,
                                                   linestyle="--")

    plt.xlabel("I_LOAD [A]")
    plt.ylabel("LDO_VDD [V]")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_ldo(FLAGS.ldo_measured_data, FLAGS.ldo_simulated_data)


if __name__ == "__main__":
    flags.DEFINE_string("ldo_measured_data",
                        "tapeout/power/data/ldo_analog_measured_chip13.csv",
                        "LDO measured data.")
    flags.DEFINE_string("ldo_simulated_data",
                        "tapeout/power/data/ldo_analog_simulated.csv",
                        "LDO simulated data.")

    app.run(main)
