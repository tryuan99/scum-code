"""The voltage logger logs the voltage data from analog input channels on the NI DAQ.

NI 9205 is a voltage input module.
"""

from absl import logging
import nidaqmx


class VoltageLogger:
    """Voltage logger to log the voltage data from analog input channels on the NI DAQ."""

    def __init__(
        self,
        channel: str,
        sampling_rate: int,
        time_to_sample: float,
        output_file: str,
        log_to_stderr: bool,
    ):
        # Initialize the NI DAQ task.
        self.task = nidaqmx.Task()
        self.task.ai_channels.add_ai_voltage_chan(
            f"cDAQ1Mod4/{channel}",
            terminal_config=nidaqmx.constants.TerminalConfiguration.RSE,
        )
        self.task.timing.cfg_samp_clk_timing(rate=sampling_rate,
                                             samps_per_chan=int(time_to_sample *
                                                                sampling_rate))
        self.output_file = output_file
        self.log_to_stderr = log_to_stderr

    def run(self) -> None:
        """Logs the voltage data from the analog input channels."""
        with open(self.output_file, "w") as f:
            data = self.task.read(number_of_samples_per_channel=nidaqmx.
                                  constants.READ_ALL_AVAILABLE)
            if len(data) == 0:
                return
            if isinstance(data[0], list):
                write_data = [",".join(map(str, z)) for z in zip(*data)]
            else:
                write_data = map(str, data)
            write_data = "\n".join(write_data)
            f.write(f"{write_data}\n")
            if self.log_to_stderr:
                logging.info(write_data)
        self.task.close()
