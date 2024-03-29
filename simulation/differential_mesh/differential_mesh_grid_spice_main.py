from absl import app, flags

from simulation.differential_mesh.differential_mesh_graph_factory import \
    DifferentialMeshGraphFactory

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    # Create the grid graph from the number of rows and columns.
    grid = DifferentialMeshGraphFactory.create_zero_2d_graph(
        FLAGS.num_rows, FLAGS.num_cols)

    # Output the SPICE netlist file.
    grid.output_spice_netlist(FLAGS.netlist)


if __name__ == "__main__":
    flags.DEFINE_integer("num_rows", 5, "Number of rows.")
    flags.DEFINE_integer("num_cols", 5, "Number of columns.")
    flags.DEFINE_string("netlist", None, "Output netlist file.")
    flags.mark_flag_as_required("netlist")

    app.run(main)
