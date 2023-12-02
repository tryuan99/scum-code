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

    plt.style.use(["science", "grid"])
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the measured data.
    df_measured.plot.line(ax=ax, x=df_measured.columns[0], linewidth=2)

    # Plot the simulated data.
    df_simulated.plot.line(ax=ax, x=df_simulated.columns[0], linestyle="--")

    plt.xlabel("I_LOAD [A]")
    plt.ylabel("LDD_VDO [V]")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_ldo(FLAGS.ldo_measured_data, FLAGS.ldo_simulated_data)


if __name__ == "__main__":
    flags.DEFINE_string("ldo_measured_data",
                        "tapeout/power/data/ldo_analog_measured.csv",
                        "LDO measured data.")
    flags.DEFINE_string("ldo_simulated_data",
                        "tapeout/power/data/ldo_analog_simulated.csv",
                        "LDO simulated data.")

    app.run(main)
