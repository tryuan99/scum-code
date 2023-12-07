from absl import app, flags, logging

from simulation.differential_mesh.differential_mesh_graph_factory import \
    DifferentialMeshGraphFactory
from simulation.differential_mesh.differential_mesh_grid import \
    DifferentialMeshGrid
from simulation.differential_mesh.differential_mesh_solver import (
    MatrixDifferentialMeshSolver, StochasticDifferentialMeshSolver)

FLAGS = flags.FLAGS


def _log_node_potentials(potentials: list[int, float]) -> None:
    """Logs the node potentials.

    Args:
        potentials: List of node potentials.
    """
    logging.info("Node potentials:")
    for node, potential in potentials:
        logging.info("%d %f", node, potential)


def _log_edge_measurements(measurements: list[tuple[int, int], float]) -> None:
    """Logs the edge differential measurements.

    Args:
        measurements: List of edge measurements.
    """
    logging.info("Edge differential measurements:")
    for (u, v), measurement in measurements:
        logging.info("%d %d %f", u, v, measurement)


def main(argv):
    assert len(argv) == 1

    graph = DifferentialMeshGraphFactory.create_from_edge_list(FLAGS.edgelist)
    grid = DifferentialMeshGrid(graph)
    grid.add_edge_measurement_noise(FLAGS.noise)
    solver = StochasticDifferentialMeshSolver(grid, verbose=FLAGS.verbose)
    solver.solve()
    grid.draw()
    _log_edge_measurements(solver.get_edge_measurements())
    _log_node_potentials(solver.get_node_potentials())
    logging.info("MSE = %f", solver.calculate_mean_squared_error())


if __name__ == "__main__":
    flags.DEFINE_string(
        "edgelist", "simulation/differential_mesh/data/example_2x2.edgelist",
        "Input edge list.")
    flags.DEFINE_float("noise", 0, "Standard deviation of the added noise.")
    flags.DEFINE_boolean("verbose", True, "If true, log verbose messages.")

    app.run(main)
