import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from absl import app, flags

FLAGS = flags.FLAGS


def plot_iref_vs_vref(measured_data: str, simulated_data: str) -> None:
    """Plots IREF as a function of VREF.

    Args:
        measured_data: Measured data filename.
        simulated_data: Simulated data filename.
    """
    df_measured = pd.read_csv(measured_data, comment="#")
    df_measured[df_measured.columns[df_measured.columns !=
                                    df_measured.columns[0]]] *= 1000000
    df_simulated = pd.read_csv(simulated_data, comment="#")
    df_simulated[df_simulated.columns[df_simulated.columns !=
                                      df_simulated.columns[0]]] *= 1000000

    plt.style.use(["science", "grid"])
    plt.rcParams.update({
        "font.size": 16,
        "lines.linewidth": 1,
        "lines.markersize": 8,
    })
    fig, ax = plt.subplots(figsize=(12, 3))

    # Plot the measured data.
    df_measured.plot.line(ax=ax, x=df_measured.columns[0], linewidth=5)

    # Plot the simulated data.
    df_simulated.plot.line(ax=ax, x=df_simulated.columns[0], linestyle="--")

    plt.xlabel("Output voltage at IREF [V]")
    plt.ylabel("Reference current [µA]")
    plt.legend(["Measured reference current [µA]"])
    plt.show()


def plot_iref_ext(measured_data: str, simulated_data: str) -> None:
    """Plots the voltage at IREF_EXT as a function of IREF_EXT.

    Args:
        measured_data: Measured data filename.
        simulated_data: Simulated data filename.
    """
    df_measured = pd.read_csv(measured_data, comment="#")
    df_simulated = pd.read_csv(simulated_data, comment="#")

    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the measured data.
    df_measured.plot.line(ax=ax, x=df_measured.columns[0], linewidth=2)

    # Plot the simulated data.
    df_simulated.plot.line(ax=ax, x=df_simulated.columns[0], linestyle="--")

    plt.xlabel("IREF_EXT [A]")
    plt.ylabel("V(IREF_EXT) [V]")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_iref_vs_vref(FLAGS.iref_vs_vref_measured_data,
                      FLAGS.iref_vs_vref_simulated_data)
    plot_iref_ext(FLAGS.iref_ext_measured_data, FLAGS.iref_ext_simulated_data)


if __name__ == "__main__":
    flags.DEFINE_string("iref_vs_vref_measured_data",
                        "tapeout/power/data/iref_vs_vref_measured.csv",
                        "IREF vs. VREF measured data.")
    flags.DEFINE_string("iref_ext_measured_data",
                        "tapeout/power/data/iref_ext_measured.csv",
                        "IREF_EXT measured data.")
    flags.DEFINE_string("iref_vs_vref_simulated_data",
                        "tapeout/power/data/iref_vs_vref_simulated.csv",
                        "IREF vs. VREF simulated data.")
    flags.DEFINE_string("iref_ext_simulated_data",
                        "tapeout/power/data/iref_ext_simulated.csv",
                        "IREF_EXT simulated data.")

    app.run(main)
