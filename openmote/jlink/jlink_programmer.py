"""The JLink programmer is used to program an OpenMote over J-Link."""

import tempfile

from absl import logging

from openmote.jlink.jlink_interface import JLinkInterface


class JLinkProgrammer:
    """J-Link programmer."""

    def __init__(self, verbose: bool = False):
        self.jlink_interface = JLinkInterface(verbose)
        self.verbose = verbose

    def program(self, bin_file: str) -> None:
        """Program the OpenMote with the given binary file.

        Args:
            bin_file: Path to the binary file.
        """
        if self.verbose:
            logging.info("Programming binary file: %s", bin_file)
        with tempfile.NamedTemporaryFile("w+") as f:
            f.write("h\n")
            f.write(f"loadbin {bin_file}, 0x00200000\n")
            f.write(f"verifybin {bin_file}, 0x00200000\n")
            f.write("r\n")
            f.write("go\n")
            f.write("q\n")
            f.flush()
            self.jlink_interface.run(f.name)
