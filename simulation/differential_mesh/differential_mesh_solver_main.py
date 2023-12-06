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
    solver = StochasticDifferentialMeshSolver(grid, verbose=True)
    solver.solve()
    grid.draw()
    logging.info("MSE = %f", solver.calculate_mean_squared_error())


if __name__ == "__main__":
    flags.DEFINE_string(
        "edgelist", "simulation/differential_mesh/data/example_2x2.edgelist",
        "Input edge list.")

    app.run(main)
