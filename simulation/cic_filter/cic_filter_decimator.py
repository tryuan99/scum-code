"""The CIC filter decimator class simulates a CIC filter with a decimator.
The filter integrates the signal, decimates it, and applies a comb filter.
"""

import numpy as np


class CicFilterDecimator:
    """CIC filter decimator."""

    def __init__(self, R: int, N: int, comb_filter: np.ndarray = None) -> None:
        self.R = R
        self.N = N

        if comb_filter is None:
            comb_filter = np.ones(1)
        self.comb_filter = comb_filter
        self.num_comb_filter_taps = len(comb_filter)
        self.comb_filter_diff = self._create_comb_filter_diff(comb_filter)

    def filter(self,
               signal: np.ndarray,
               downsampling: bool = True) -> np.ndarray:
        """Filters and decimates the given signal.

        Args:
            signal: Signal.
            downsampling: If true, downsample the signal after filtering.
        """
        # Integrate the signal N times.
        integrated = signal
        for _ in range(self.N):
            integrated = np.cumsum(integrated)

        if downsampling:
            downsampled = integrated[::self.R // self.num_comb_filter_taps]
            combed = downsampled
        else:
            combed = integrated

        # Apply a comb filter to the signal N times.
        for _ in range(self.N):
            combed = self._convolve(combed, self.comb_filter_diff)

        if downsampling:
            return combed[::self.num_comb_filter_taps]
        return combed

    @staticmethod
    def calculate_spectrum_magnitude(
            signal: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Calculates the magnitude of the spectrum of the signal.

        Args:
            signal: Signal.
            length: FFT length.

        Returns:
            A 2-tuple consisting of the omega axis and the magnitude vector.
        """
        signal_fft = np.fft.fft(signal, length)
        signal_fft_abs = np.abs(signal_fft)
        omega = np.linspace(0, 2 * np.pi, length, endpoint=False)
        return omega, signal_fft_abs

    @staticmethod
    def _convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Convolves two signals but only includes boundary effects at the
        beginning.

        Args:
            a: First input signal.
            b: Second input signal.

        Returns:
            The convolution of a and b with boundary effects only at the
            beginning.
        """
        max_length = max(len(a), len(b))
        return np.convolve(a, b)[:max_length]

    @staticmethod
    def _create_comb_filter_diff(coefficients: np.ndarray) -> np.ndarray:
        """Creates the difference comb filter coefficients.

        Args:
            coefficients: The desired filter coefficients without zero padding.

        Returns:
            The difference coefficients of the comb filter.
        """
        coefficients_with_zero_padding = np.concatenate(
            (np.array([0]), coefficients, np.array([0])))
        diff = np.diff(coefficients_with_zero_padding).astype(np.float64)
        return diff
