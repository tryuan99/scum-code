from absl import app, flags

FLAGS = flags.FLAGS

from simulation.differential_mesh.differential_mesh_graph_factory import \
    DifferentialMeshGraphFactory
from simulation.differential_mesh.differential_mesh_grid import \
    DifferentialMeshGrid


def main(argv):
    assert len(argv) == 1

    # Create the graph from the number of rows and columns.
    graph = DifferentialMeshGraphFactory.create_zero_2d_graph(
        FLAGS.num_rows, FLAGS.num_cols)
    grid = DifferentialMeshGrid(graph)
    grid.add_edge_measurement_noise(FLAGS.noise)
    grid.draw()

    # Create the graph from the edge list.
    graph = DifferentialMeshGraphFactory.create_from_edge_list(FLAGS.edgelist)
    grid = DifferentialMeshGrid(graph)
    grid.add_edge_measurement_noise(FLAGS.noise)
    grid.draw()


if __name__ == "__main__":
    flags.DEFINE_integer("num_rows", 3, "Number of rows.")
    flags.DEFINE_integer("num_cols", 4, "Number of columns.")
    flags.DEFINE_string(
        "edgelist", "simulation/differential_mesh/data/example_2x2.edgelist",
        "Input edge list.")
    flags.DEFINE_float("noise", 0, "Standard deviation of the added noise.")

    app.run(main)
