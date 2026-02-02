import pandas as pd
from absl import app, flags

from openwsn.analysis.multi_event_plotter import MultiEventPlotter

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    # Read the CSV files.
    dfs = [pd.read_csv(log, comment="#") for log in FLAGS.logs]
    event_plotter = MultiEventPlotter(dfs)
    event_plotter.plot_timeline()


if __name__ == "__main__":
    flags.DEFINE_multi_string("logs", [
        "openwsn/analysis/logs/single_channel_calibration_multiple_1a.csv",
        "openwsn/analysis/logs/single_channel_calibration_multiple_1b.csv",
    ], "CSV logs.")

    app.run(main)
