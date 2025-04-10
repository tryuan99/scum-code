import pandas as pd
from absl import app, flags

from openwsn.analysis.event_plotter import EventPlotter

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    # Read the CSV file.
    df = pd.read_csv(FLAGS.log, comment="#")
    event_plotter = EventPlotter(df)
    event_plotter.plot_timeline()


if __name__ == "__main__":
    flags.DEFINE_string("log", None, "CSV log.")
    flags.mark_flag_as_required("log")

    app.run(main)
