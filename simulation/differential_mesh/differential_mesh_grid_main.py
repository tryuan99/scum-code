from absl import app, flags

FLAGS = flags.FLAGS

from simulation.differential_mesh.differential_mesh_graph_factory import \
    DifferentialMeshGraphFactory
from simulation.differential_mesh.differential_mesh_grid import \
    DifferentialMeshGrid


def main(argv):
    assert len(argv) == 1

    graph = DifferentialMeshGraphFactory.create_from_edge_list(
        FLAGS.edgelist, noise=FLAGS.noise)
    grid = DifferentialMeshGrid(graph)
    grid.draw()


if __name__ == "__main__":
    flags.DEFINE_string(
        "edgelist",
        "simulation/differential_mesh/data/example_3x3_zero.edgelist",
        "Input edge list.")
    flags.DEFINE_float("noise", 0, "Standard deviation of the added noise.")

    app.run(main)
