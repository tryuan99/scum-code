"""The differential mesh simulator solves the differential mesh grid and
simulates various metrics of the solution.
"""

import networkx as nx
import numpy as np

from simulation.differential_mesh.differential_mesh_grid import \
    DifferentialMeshGrid
from simulation.differential_mesh.differential_mesh_solver import \
    DifferentialMeshSolver


class DifferentialMeshSimulator:
    """Differential mesh simulator.

    Attributes:
        graph: Differential mesh graph.

    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph

    def simulate_node_standard_errors(self, solver_cls: DifferentialMeshSolver,
                                      noise: float, num_iterations: int,
                                      verbose: bool) -> list[tuple[int, float]]:
        """Simulate the standard error at each node.

        Args:
            solver_cls: Differential mesh solver class.
            noise: Standard deviation of the added noise.
            num_iterations: Number of iterations to simulate.
            verbose: If verbose, log verbose messages.

        Returns:
            A list of 2-tuples, each consisting of the node label and the
            corresponding standard error.
        """
        # Initialize the solved node potentials as a 2D matrix.
        potentials = np.zeros((self.graph.number_of_nodes(), num_iterations))

        # Simulate the solver.
        for iteration in range(num_iterations):
            # Solve for the node potentials.
            grid = DifferentialMeshGrid(self.graph)
            grid.add_edge_measurement_noise(noise)
            solver = solver_cls(grid, verbose=verbose)
            solver.solve()

            # Record the solved node potentials.
            for node, potential in solver.get_node_potentials():
                potentials[node - 1, iteration] = potential

        # Calculate the standard eror of each node.
        stderrs = np.std(potentials, axis=1)

        return [(node_index + 1, stderr)
                for node_index, stderr in enumerate(stderrs)]
