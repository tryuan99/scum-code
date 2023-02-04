from absl import app, flags

from utils.plotter import dummy_live_plotter

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    plotter = dummy_live_plotter.DummyLivePlotter(FLAGS.max_num_points,
                                                  FLAGS.max_counter)
    plotter.run()


if __name__ == "__main__":
    flags.DEFINE_integer("max_num_points", 100,
                         "Maximum number of points to plot at a time.")
    flags.DEFINE_integer("max_counter", 10, "Maximum counter value.")

    app.run(main)
