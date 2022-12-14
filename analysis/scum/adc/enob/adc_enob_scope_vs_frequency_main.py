from absl import app, flags, logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

FLAGS = flags.FLAGS

# ENOB in bits at various input sinusoid frequencies in Hz.
ENOB_VS_FREQUENCY = {
    1: 4.6268,
    2: 4.6132,
    5: 4.6117,
    10: 4.6211,
    20: 4.6159,
    50: 4.6015,
    100: 4.6147,
    200: 4.5951,
    500: 4.5709,
    1000: 4.4937,
    2000: 4.3725,
}


def plot_adc_enob_vs_frequency(data: Dict[int, float]) -> None:
    """Plots the ADC ENOB as a function of frequency.

    Args:
        data: Dictionary from frequency in Hz to ENOB in bits.
    """
    # Plot the ENOB as a function of frequency.
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.semilogx(data.keys(), data.values())
    ax.set_xlabel("Input sinusoid frequency [Hz]")
    ax.set_ylabel("ENOB [bits]")
    ax.set_title("ENOB vs. input sinusoid frequency")
    plt.show()


def main(argv):
    assert len(argv) == 1
    plot_adc_enob_vs_frequency(ENOB_VS_FREQUENCY)


if __name__ == "__main__":
    app.run(main)
