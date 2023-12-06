from absl import app, flags, logging

from simulation.differential_mesh.differential_mesh_grid import \
    DifferentialMeshGrid
from simulation.differential_mesh.differential_mesh_solver import \
    StochasticDifferentialMeshSolver

FLAGS = flags.FLAGS


def main(argv):
    assert len(argv) == 1

    grid = DifferentialMeshGrid.read_edge_list(FLAGS.edgelist)
    solver = StochasticDifferentialMeshSolver(grid, verbose=True)
    solver.solve()
    grid.draw()
    logging.info("MSE = %f", solver.calculate_mean_squared_error())


if __name__ == "__main__":
    flags.DEFINE_string(
        "edgelist", "simulation/differential_mesh/data/example_2x2.edgelist",
        "Input edge list.")

    app.run(main)
