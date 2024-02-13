import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from absl import app


def plot_vbat(ax) -> None:
    """Plots the output voltage as a function of the load current while
    sweeping V_BAT.
    """
    # Data for SW_CAP_OUT from V_BAT.
    vbat_3v3 = pd.DataFrame({
        "load_current_mA": [0, 5, 10],
        "output_voltage_V": [1.068, 0.979, 0.851],
    })
    vbat_3v6 = pd.DataFrame({
        "load_current_mA": [0, 5, 10, 15],
        "output_voltage_V": [1.162, 1.082, 0.986, 0.795],
    })
    vbat_3v9 = pd.DataFrame({
        "load_current_mA": [0, 5, 10, 15, 20],
        "output_voltage_V": [1.265, 1.189, 1.103, 1.000, 0.784],
    })
    vbat_4v2 = pd.DataFrame({
        "load_current_mA": [0, 5, 10, 15, 20, 25],
        "output_voltage_V": [1.371, 1.303, 1.222, 1.130, 1.020, 0.860],
    })

    vbat_3v3.plot.line(ax=ax, x=vbat_3v3.columns[0], linewidth=2, marker="^")
    vbat_3v6.plot.line(ax=ax, x=vbat_3v6.columns[0], linewidth=2, marker="^")
    vbat_3v9.plot.line(ax=ax, x=vbat_3v9.columns[0], linewidth=2, marker="^")
    vbat_4v2.plot.line(ax=ax, x=vbat_4v2.columns[0], linewidth=2, marker="^")

    ax.axhline(1.1, linestyle="--", linewidth=2)
    ax.set_xlabel("Load current [mA]")
    ax.set_ylabel("Switched-capacitor DC-DC\nconverter output voltage [V]")
    ax.legend(
        ["V_BAT = 3.3 V", "V_BAT = 3.6 V", "V_BAT = 3.9 V", "V_BAT = 4.2 V"])
    ax.set_title("3:1 configuration")


def plot_vmid(ax) -> None:
    """Plots the outupt voltage as a function of the load current while
    sweeping V_MID.
    """
    # Data for SW_CAP_OUT from V_MID.
    vmid_1v5 = pd.DataFrame({
        "load_current_mA": [0, 1, 2, 3, 4, 5, 6],
        "output_voltage_V": [0.748, 0.735, 0.720, 0.704, 0.684, 0.664, 0.635],
    })
    vmid_1v8 = pd.DataFrame({
        "load_current_mA": [
            0, 1, 2, 3, 4, 5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 12, 14,
            15, 15.5
        ],
        "output_voltage_V": [
            0.898, 0.889, 0.881, 0.870, 0.862, 0.851, 0.843, 0.838, 0.832,
            0.827, 0.821, 0.815, 0.807, 0.801, 0.795, 0.789, 0.768, 0.732,
            0.703, 0.689
        ],
    })
    vmid_2v2 = pd.DataFrame({
        "load_current_mA":
            list(range(37)),
        "output_voltage_V": [
            1.098, 1.089, 1.081, 1.072, 1.064, 1.055, 1.046, 1.037, 1.028,
            1.019, 1.010, 1.000, 0.991, 0.982, 0.972, 0.963, 0.953, 0.942,
            0.932, 0.921, 0.908, 0.896, 0.883, 0.869, 0.855, 0.841, 0.828,
            0.814, 0.800, 0.784, 0.769, 0.758, 0.755, 0.754, 0.748, 0.740, 0.740
        ],
    })

    vmid_1v5.plot.line(ax=ax, x=vmid_1v5.columns[0], linewidth=2, marker="^")
    vmid_1v8.plot.line(ax=ax, x=vmid_1v8.columns[0], linewidth=2, marker="^")
    vmid_2v2.plot.line(ax=ax, x=vmid_2v2.columns[0], linewidth=2, marker="^")

    ax.set_xlabel("Load current [mA]")
    ax.set_ylabel("Switched-capacitor DC-DC\nconverter output voltage [V]")
    ax.set_title("2:1 configuration")
    ax.legend(["V_MID = 1.5 V", "V_MID = 1.8 V", "V_MID = 2.2 V"])


def main(argv):
    assert len(argv) == 1

    plt.style.use(["science", "grid"])
    plt.rcParams.update({
        "font.size": 16,
        "lines.markersize": 8,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    plot_vbat(ax1)
    plot_vmid(ax2)
    plt.show()


if __name__ == "__main__":
    app.run(main)
