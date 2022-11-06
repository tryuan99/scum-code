"""The ADC config stores SCuM board-specific ADC configs."""


class AdcConfig:
    """ADC config."""

    def __init__(
        self,
        ldo_output: float,
        m: float,
        b: float,
        actual_min_lsbs: int,
        actual_max_lsbs: int,
        actual_min_volts: float,
        actual_max_volts: float,
    ):
        self.ldo_output = ldo_output
        self.m = m
        self.b = b

        self.num_bits = 10
        self.min_value = 0
        self.max_value = 1023

        self.actual_min_lsbs = actual_min_lsbs
        self.actual_max_lsbs = actual_max_lsbs
        self.actual_min_volts = actual_min_volts
        self.actual_max_volts = actual_max_volts

        self.max_sampling_rate = 41.14e3

    @property
    def amplitude_lsbs(self) -> float:
        """Returns the sinusoidal amplitude in LSBs."""
        return (self.actual_max_lsbs - self.actual_min_lsbs) / 2

    @property
    def offset_lsbs(self) -> float:
        """Returns the sinusoidal offset in LSBs."""
        return (self.actual_min_lsbs + self.actual_max_lsbs) / 2

    @property
    def amplitude_volts(self) -> float:
        """Returns the sinusoidal amplitude in volts."""
        return (self.actual_max_volts - self.actual_min_volts) / 2

    @property
    def offset_volts(self) -> float:
        """Returns the sinusoidal offset in volts."""
        return (self.actual_min_volts + self.actual_max_volts) / 2

    def lsb2volt(self, lsb: int) -> float:
        """Converts the ADC output from LSBs to volts.

        Args:
            lsb: ADC output in LSBs.
        """
        return (lsb - self.b) / self.m

    def volt2lsb(self, volt: float) -> float:
        """Converts the ADC output from volts to LSBs.

        Args:
            volt: ADC output in volts.
        """
        return self.m * volt + self.b

    def lsb2volt_stddev(self, lsb: int) -> float:
        """Converts the standard deviation from LSBs to volts.

        Args:
            lsb: Standard deviation in LSBs.
        """
        return lsb / self.m

    def volt2lsb_stddev(self, volt: float) -> float:
        """Converts the standard deviation from volts to LSBs.

        Args:
            volt: Standard deviation in volts.
        """
        return self.m * volt


ADC_CONFIGS = {
    "l35": AdcConfig(1.393, 744.2, 46.34, 40, 511, 0, 0.62),
}
