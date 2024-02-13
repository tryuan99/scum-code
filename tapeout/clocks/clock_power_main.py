import matplotlib.pyplot as plt
import pandas as pd
import scienceplots
from absl import app


def main(argv):
    assert len(argv) == 1

    plt.style.use(["science", "grid"])
    plt.rcParams.update({
        "font.size": 16,
        "lines.markersize": 8,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Data for the CPU clock domain variations.
    cpu_data = pd.DataFrame({
        "CPU_clock_kHz": [
            5, 200, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000,
            70000, 80000, 90000, 100000
        ],
        "VDD_D_current_mA": [
            0.053, 0.0818, 0.100, 0.131, 0.3757, 0.684, 1.292, 1.904, 2.509,
            3.115, 3.737, 4.348, 4.948, 5.563, 6.169
        ],
    })

    cpu_data[cpu_data.columns[0]] /= 1000
    cpu_data.plot.line(ax=ax1, x=cpu_data.columns[0], linewidth=2, marker="^")

    ax1.set_xlabel("CPU clock frequency [MHz]")
    ax1.set_ylabel("VDD_D current draw [mA]")
    ax1.legend(["Measured current draw"])

    # Data for the ADC clock domain variations.
    adc_data = pd.DataFrame({
        "ADC_clock_kHz": [
            200, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 32000
        ],
        "VDD_D_current_mA": [
            0.0818, 0.091, 0.118, 0.329, 0.589, 0.842, 1.077, 1.294, 1.496,
            1.573
        ],
    })

    adc_data[adc_data.columns[0]] /= 1000
    adc_data.plot.line(ax=ax2, x=adc_data.columns[0], linewidth=2, marker="^")

    ax2.set_xlabel("ADC clock frequency [MHz]")
    ax2.set_ylabel("VDD_D current draw [mA]")
    ax2.legend(["Measured current draw"])

    plt.show()


if __name__ == "__main__":
    app.run(main)
