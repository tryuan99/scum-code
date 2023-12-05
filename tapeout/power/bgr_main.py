import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from absl import app, flags

FLAGS = flags.FLAGS


def plot_bgr(measured_data: str, simulated_data: str) -> None:
    """Plots bandgap reference voltage as a function of the supply voltage.

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

    plt.xlabel("LDO_VIN [V]")
    plt.ylabel("BG_REF [V]")
    plt.legend()
    plt.show()


def main(argv):
    assert len(argv) == 1

    plot_bgr(FLAGS.bgr_measured_data, FLAGS.bgr_simulated_data)


if __name__ == "__main__":
    flags.DEFINE_string("bgr_measured_data",
                        "tapeout/power/data/bgr_measured.csv",
                        "Bandgap reference measured data.")
    flags.DEFINE_string("bgr_simulated_data",
                        "tapeout/power/data/bgr_simulated.csv",
                        "Bandgap reference simulated data.")

    app.run(main)
