"""The pH PWM ramp data represents the data collected from the pH sensor when
excited by a PWM ramp.

The Saleae Logic Pro 2 analyzer records the time, the PWM ramp voltage, and the
output voltage.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal

# The PWM ramp starts when it has increased by at least 2%.
PWM_RAMP_START_THRESHOLD = 0.02

# FFT bin at which to zer out the high-frequency compoennts.
FFT_CUTOFF_BIN = 3


class PhPwmRampData:
    """Represents the PWM ramp voltage and the output voltage along with the time."""

    def __init__(self, df: pd.DataFrame, sampling_rate: float):
        self.df = df.copy()
        self.time_column, self.pwm_column, self.output_column = self.df.columns
        self.sampling_rate = sampling_rate
        self._process_pwm_ramp_data()

    @property
    def pwm_time(self) -> np.ndarray:
        """PWM time in seconds."""
        return self._pwm_time

    @property
    def pwm_ramp(self) -> np.ndarray:
        """PWM ramp voltage in V."""
        return self._pwm_ramp

    @property
    def pwm_output(self) -> np.ndarray:
        """Output data in V."""
        return self._pwm_output

    @property
    def filtered_pwm_output(self) -> np.ndarray:
        """Filtered output data in V."""
        return self._filtered_pwm_output

    @property
    def filtered_pwm_output_diff(self) -> np.ndarray:
        """First difference of the filtered output data in V."""
        return np.diff(self.filtered_pwm_output)

    @property
    def filtered_pwm_output_diff_diff(self) -> np.ndarray:
        """Second difference of the filtered output data in V."""
        return np.diff(self.filtered_pwm_output_diffr)

    def _process_pwm_ramp_data(self) -> None:
        """Processes the PWM ramp data."""
        # Process the PWM ramp.
        clean_pwm_ramp = self._cleanup_pwm_ramp()
        pwm_ramp_start_index = self._find_pwm_ramp_start(clean_pwm_ramp)
        pwm_ramp_end_index = np.argmax(clean_pwm_ramp)
        self._pwm_ramp = clean_pwm_ramp[pwm_ramp_start_index:pwm_ramp_end_index]

        # Process the output within the PWM ramp.
        self._pwm_time = self.df[
            self.time_column][pwm_ramp_start_index:pwm_ramp_end_index]
        self._pwm_output = self.df[
            self.output_column][pwm_ramp_start_index:pwm_ramp_end_index]
        self._filter_output()

    def _cleanup_pwm_ramp(self, cutoff_frequency: float = 10) -> np.ndarray:
        """Filters the PWM ramp.

        Args:
            cutoff_frequency: Cutoff frequency of the filter in Hz.

        Returns:
            The filtered PWM ramp data.
        """
        # Generate a Butterworth filter to filter the PWM ramp.
        butter = scipy.signal.butter(3,
                                     cutoff_frequency,
                                     fs=self.sampling_rate,
                                     output="sos")
        return scipy.signal.sosfiltfilt(butter, self.df[self.pwm_column])

    def _filter_output(self) -> np.ndarray:
        """Filters the output data."""
        # Perform the FFT on the output data.
        pwm_output_fft = np.fft.fft(self.pwm_output)
        # self._plot_output_fft(pwm_output_fft)
        # Zero out the high-frequency components.
        pwm_output_fft[FFT_CUTOFF_BIN:len(pwm_output_fft) - FFT_CUTOFF_BIN +
                       1] = 0
        self._filtered_pwm_output = np.real(np.fft.ifft(pwm_output_fft))

    @staticmethod
    def _find_pwm_ramp_start(data: np.ndarray) -> int:
        """Finds the index corresponding to the start of the PWM ramp.

        Args:
            data: PWM ramp data.

        Returns:
            The PWM ramp start index.
        """
        max_pwm_ramp_value = np.max(data)
        min_pwm_ram_value = np.min(data)
        # Find all indices, where the PWM ramp has increased by at least 2%.
        ramp_indices = np.squeeze(
            np.argwhere(data > PWM_RAMP_START_THRESHOLD *
                        (max_pwm_ramp_value - min_pwm_ram_value) +
                        min_pwm_ram_value))
        pwm_ramp_start_index = np.argmax(ramp_indices > np.argmin(data))
        return ramp_indices[pwm_ramp_start_index]

    @staticmethod
    def _plot_output_fft(fft_data: np.ndarray) -> None:
        """Plot the FFT of the output.

        Args:
            fft_data: FFT data.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(np.abs(fft_data), label="Magnitude")
        ax.set_title("FFT magnitude of the output data")
        ax.set_xlabel("FFT bin")
        ax.set_ylabel("FFT magnitude")
        plt.legend()
        plt.show()
