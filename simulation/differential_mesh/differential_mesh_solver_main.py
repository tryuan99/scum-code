from absl import app, flags, logging

from simulation.differential_mesh.differential_mesh_graph_factory import \
    DifferentialMeshGraphFactory
from simulation.differential_mesh.differential_mesh_grid import \
    DifferentialMeshGrid
from simulation.differential_mesh.differential_mesh_solver import (
    MatrixDifferentialMeshSolver, StochasticDifferentialMeshSolver)

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    graph = DifferentialMeshGraphFactory.create_from_edge_list(FLAGS.edgelist)
    grid = DifferentialMeshGrid(graph)
    grid.add_edge_measurement_noise(FLAGS.noise)
    solver = StochasticDifferentialMeshSolver(grid, verbose=FLAGS.verbose)
    solver.solve()
    grid.draw()
    solver.log_node_potentials()
    logging.info("MSE = %f", solver.calculate_mean_squared_error())


if __name__ == "__main__":
    flags.DEFINE_string(
        "edgelist", "simulation/differential_mesh/data/example_2x2.edgelist",
        "Input edge list.")
    flags.DEFINE_float("noise", 0, "Standard deviation of the added noise.")
    flags.DEFINE_boolean("verbose", True, "If true, log verbose messages.")

    app.run(main)
