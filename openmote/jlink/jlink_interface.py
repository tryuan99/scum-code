"""The JLink interface is used to interface with a J-Link device."""

import subprocess

from absl import logging

# The JLink command to interface with a J-Link device.
JLINK_COMMAND = "JLinkExe -Device cc2538sf53 -Speed 4000 -If JTAG -JTAGConf \"-1,-1\" -AutoConnect 1 -ExitOnError 1"


class JLinkInterface:
    """Interface to a J-Link device."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def open(self) -> None:
        """Opens a connection to the J-Link device."""
        self._execute(JLINK_COMMAND)

    def run(self, command_file: str) -> None:
        """Runs the commands in the given file.

        Args:
            command_file: Path to the commands file.
        """
        command = f"{JLINK_COMMAND} -CommandFile {command_file}"
        self._execute(command)

    def _execute(self, command: str) -> None:
        """Execute the given J-Link command.

        Args:
            command: J-Link command.
        """
        if self.verbose:
            logging.info("Running J-Link command: %s", command)
        subprocess.run(command, shell=True, stderr=subprocess.STDOUT)
