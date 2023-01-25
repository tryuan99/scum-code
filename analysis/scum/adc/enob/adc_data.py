"""The ADC data represents the ADC data collected over time."""

import numpy as np


class AdcData:
    """Represents a sequence of ADC samples."""

    def __init__(self, adc_data: np.ndarray, sampling_rate: float):
        assert adc_data.ndim == 1, "ADC data must be one-dimensional."
        self.samples = np.copy(adc_data)
        self.num_samples = len(self.samples)
        self.sampling_rate = sampling_rate

    @property
    def sample_axis(self) -> np.ndarray:
        """Returns the sample axis."""
        return np.arange(len(self.samples))

    @property
    def time_axis(self) -> np.ndarray:
        """Returns the time axis."""
        return np.arange(len(self.samples)) / self.sampling_rate

    def _get_sample_msb_9(self, index: int, msb: int) -> int:
        """Returns the ADC sample at the given index with the 9th bit set to the given MSB.

        Args:
            index: Index of the ADC sample.
            msb: Value of the 9th bit.
        """
        return msb * 2**8 + self.samples[index]

    @staticmethod
    def _get_disambiguation_cost_msb_9(previous_sample: int,
                                       current_sample: int,
                                       msb_switch: bool) -> int:
        """Returns the cost of disambiguating the 9th bit of the new sample
        given the old sample and whether the MSB switched.

        Args:
            previous_sample: Previous disambiguated ADC sample.
            current_sample: Current disambiguated ADC sample.
            msb_switch: If true, the MSB switched betwen the previous and the
                        current ADC samples.
        """
        MSB_SWITCH_PENALTY = 20
        return (np.abs(current_sample - previous_sample) +
                MSB_SWITCH_PENALTY * msb_switch)

    def disambiguate_msb_9(self) -> None:
        """Disambiguates the 9th bit from the ADC data.

        Some ADC samples are 8-bit values because the 9th bit was omitted due
        to limited memory.
        """
        if self.num_samples <= 0:
            return

        # Use dynamic programming.
        # The (i, j)th entry in the running costs table is a length-2 tuple
        # (minimum cost of MSB disambiguation up to the ith ADC sample if the
        # MSB bit of the ith ADC sample is j, MSB bit of the (i - 1)th ADC
        # sample).
        # Note that we allow the MSB bit to be 2 only because of possible
        # overflow over 511.
        running_costs = np.zeros((self.num_samples, 3, 2),
                                 dtype=self.samples.dtype)
        for i in range(1, self.num_samples):
            for j in range(3):
                new_costs = [
                    self._get_disambiguation_cost_msb_9(
                        self._get_sample_msb_9(i - 1, msb),
                        self._get_sample_msb_9(i, j),
                        j != msb,
                    ) for msb in range(3)
                ]
                running_costs[i, j] = (
                    running_costs[i - 1, np.argmin(new_costs), 0] +
                    np.min(new_costs),
                    np.argmin(new_costs),
                )

        current_msb = np.argmin(running_costs[-1, :, 0])
        for i in range(self.num_samples - 1, -1, -1):
            self.samples[i] += current_msb * 2**8
            current_msb = running_costs[i, current_msb, 1]
