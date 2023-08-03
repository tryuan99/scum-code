"""The ADC data represents the ADC data collected over time."""

import numpy as np
import scipy.signal

from utils.regression.exponential_regression import ExponentialRegression
from utils.regression.linear_regression import (LinearRegression,
                                                WeightedLinearRegression)
from utils.regression.polynomial_regression import PolynomialRegression

# Number of bits in an ADC sample.
NUM_ADC_SAMPLE_BITS = 10

# Maximum difference in LSBs between consecutive ADC samples.
MAX_DIFF_BETWEEN_CONSECUTIVE_ADC_SAMPLES = 64  # LSBs

# Number of ADC samples to average at the end to find the minimum ADC output.
NUM_AVERAGES_FOR_MIN_ADC_OUTPUT = 10

# Empirically determined scaling factor of the exponential measured by SCuM.
EXPONENTIAL_SCALING_FACTOR = 870

# Standard deviation of SCuM's ADC.
SIGMA = 5


class ExponentialAdcData:
    """Represents the ADC samples of a decaying exponential.

    The ADC data is assumed to be a decaying exponential and has decayed
    completely by the end of the samples.
    """

    def __init__(self, samples: np.ndarray, sampling_rate: float):
        assert samples.ndim == 1, "ADC data must be one-dimensional."
        self.samples = np.copy(samples)
        self.num_samples = len(self.samples)
        self.sampling_rate = sampling_rate

    @property
    def t_axis(self) -> np.ndarray:
        """Time axis in seconds."""
        return np.arange(len(self.samples)) / self.sampling_rate

    @property
    def min_adc_output(self) -> float:
        """Minimum ADC output."""
        # Average the last ADC samples to find the minimum ADC output.
        return np.mean(self.samples[-NUM_AVERAGES_FOR_MIN_ADC_OUTPUT:])

    @property
    def max_adc_output(self) -> float:
        """Maximum ADC output."""
        return np.max(self.samples)

    def disambiguate_msb_9(self) -> None:
        """Disambiguates the 9th bit from the ADC data.

        This function fixes any discontinuities in the ADC data caused by the
        stuck MSB.
        """
        # Correct the ADC samples by the value of the MSB.
        correction = 2**(NUM_ADC_SAMPLE_BITS - 1)

        # Fix any discontinuities caused by the MSB bit.
        diffs = np.squeeze(
            np.argwhere(
                np.abs(np.diff(self.samples)) >
                MAX_DIFF_BETWEEN_CONSECUTIVE_ADC_SAMPLES))
        self.samples[:np.min(diffs)] += correction

        # Debounce the ADC data at the discontinuities.
        for i in range(np.min(diffs), np.max(diffs) + 1):
            if self.samples[i] - self.samples[
                    i - 1] < -MAX_DIFF_BETWEEN_CONSECUTIVE_ADC_SAMPLES:
                self.samples[i] += correction

    def filter_samples(self, cutoff_frequency: float = 10) -> None:
        """Filters the noise from the ADC data.

        Args:
            cutoff_frequency: Cutoff frequency in Hz.
        """
        # Use a Butterworth filter.
        butter = scipy.signal.butter(3,
                                     cutoff_frequency,
                                     fs=self.sampling_rate,
                                     output="sos")
        self.samples = scipy.signal.sosfiltfilt(butter, data)

    def perform_exponential_regression(self) -> ExponentialRegression:
        """Performs an exponential regression on the ADC data.

        Returns:
            The exponential regression.
        """
        t = self.t_axis
        three_tau_index = self._estimate_three_tau_index()
        return ExponentialRegression(t[:three_tau_index],
                                     self.samples[:three_tau_index])

    def perform_linear_regression(self) -> LinearRegression:
        """Performs a linear regression on the ADC data in log space.

        Returns:
            The linear regression in log space.
        """
        t = self.t_axis
        three_tau_index = self._estimate_three_tau_index()
        return LinearRegression(
            t[:three_tau_index],
            np.log(self.samples[:three_tau_index] - self.min_adc_output))

    def perform_weighted_linear_regression(self) -> WeightedLinearRegression:
        """Performs a weighted linear regression on the ADC data in log space.

        The variance of the ADC samples increases exponentially, and the weights
        are the reciprocals of the corresponding variances.

        Returns:
            The weighted linear regression in log space.
        """
        t = self.t_axis
        three_tau_index = self._estimate_three_tau_index()
        variances = (
            SIGMA *
            np.exp(np.arange(len(self.samples)) /
                   (three_tau_index / 3)) / EXPONENTIAL_SCALING_FACTOR)**2
        weights = 1 / variances
        return WeightedLinearRegression(
            t[:three_tau_index],
            np.log(self.samples[:three_tau_index] - self.min_adc_output),
            weights[:three_tau_index])

    def perform_polynomial_regression(self,
                                      degree: int = 5) -> PolynomialRegression:
        """Performs a polynomial regression on the ADC data.

        Args:
            degree: Polynomial degree.

        Returns:
            The polynomial regression.
        """
        t = self.t_axis
        three_tau_index = self._estimate_three_tau_index()
        return PolynomialRegression(
            t[:three_tau_index],
            self.samples[:three_tau_index] - self.min_adc_output, degree)

    def estimate_tau(self) -> float:
        """Estimates the time constant based on the three tau index.

        Returns:
            The estimated time constant.
        """
        three_tau_index = self._estimate_three_tau_index()
        return three_tau_index / 3 / self.sampling_rate

    def _estimate_three_tau_index(self) -> int:
        """Estimates the index at 3tau.

        The sample at 3tau corresponds to where the exponential has decayed by 95%.
        Note that this function only estimates 3tau due to the presence of noise.

        Returns:
            The index at 3tau.
        """
        # Average the last ADC samples to find the minimum ADC output.
        three_tau_index = np.argmax(self.samples < 0.95 * self.min_adc_output +
                                    0.05 * self.max_adc_output)
        return three_tau_index
