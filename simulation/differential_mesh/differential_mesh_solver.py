"""The differential mesh solver accepts a differential mesh grid and solves for
the node potentials given the edge differential measurements.

The solution ensures that each node takes on the average potential of what its
neighbors think the node is at. Equivalently, the sum of the errors of the
incident differential measurements is 0 for every node.

The node potentials are written to the graph as node attributes.
"""

import itertools
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator

import networkx as nx
import numpy as np
from absl import logging

from simulation.differential_mesh.differential_mesh_grid import (
    DIFFERENTIAL_MESH_GRID_EDGE_MEASUREMENT_ATTRIBUTE,
    DIFFERENTIAL_MESH_GRID_NODE_POTENTIAL_ATTRIBUTE,
    DIFFERENTIAL_MESH_GRID_ROOT_NODE, DifferentialMeshGrid)


class DifferentialMeshSolver(ABC):
    """Interface for a differential mesh solver.

    Attributes:
        grid: Differential mesh grid.
        verbose: If verbose, log verbose messages.
    """

    def __init__(self, grid: DifferentialMeshGrid, verbose: bool = False):
        self.grid = grid
        self.verbose = verbose
        self._reset_node_potentials()

    @property
    def graph(self) -> nx.DiGraph:
        """Graph object corresponding to the differential mesh grid."""
        return self.grid.graph

    @abstractmethod
    def solve(self) -> None:
        """Solves for the node potentials."""

    def get_node_potentials(self) -> list[tuple[int, float]]:
        """Returns the solved node potentials.

        This function returns a list of 2-tuples, each consisting of the node
        label and its potential.
        """
        return [
            (node, self._get_node_potential(node)) for node in self.graph.nodes
        ]

    def get_edge_measurements(self) -> list[tuple[tuple[int, int], float]]:
        """Returns the edge differential measurements.

        This function returns a list of 2-tuples, each consisting of a 2-tuple
        denoting the adjacent nodes of the edge and the edge differential
        measurement.
        """
        return [((u, v), self._get_edge_measurement(u, v))
                for u, v in self.graph.edges]

    def calculate_mean_squared_error(self) -> float:
        """Calculates the mean squared error along all edges.

        The error along an edge is defined as the difference between the node
        potential difference and the corresponding edge differential
        measurement.

        Returns:
            Mean squared error of the solved node potentials.
        """
        mean_squared_error = np.mean([
            self._calculate_edge_error(u, v)**2 for u, v in self.graph.edges()
        ])
        return mean_squared_error

    def _get_neighbors(self, node: int) -> Iterator[int]:
        """Returns an iterator over the neighbors of the given node.

        Args:
            node: Node for which to return the neighbors.
        """
        return itertools.chain(self.graph.predecessors(node),
                               self.graph.successors(node))

    def _get_node_potential(self, node: int) -> float:
        """Returns the node potential.

        Args:
            node: Node in the differential mesh grid.
        """
        return self.graph.nodes[node].get(
            DIFFERENTIAL_MESH_GRID_NODE_POTENTIAL_ATTRIBUTE, 0)

    def _set_node_potential(self, node: int, potential: float) -> None:
        """Sets the node potential.

        Args:
            node: Node in the differential mesh grid.
            potential: Potential to set the node to.
        """
        self.graph.nodes[node][
            DIFFERENTIAL_MESH_GRID_NODE_POTENTIAL_ATTRIBUTE] = potential

    def _reset_node_potentials(self) -> None:
        """Sets potential of all nodes to zero."""
        for node in self.graph.nodes:
            self._set_node_potential(node, 0)

    def _calculate_node_error(self, node: int) -> float:
        """Calculates the node error of the given node.

        The node error is defined as the sum of incident differential
        measurement errors. We define it as the sum of the outgoing edge errors
        minus the sum of the incoming edge errors.

        Args:
            node: Node for which to calculate the error.

        Returns:
            The error at the given node.
        """
        outgoing_edge_errors = np.sum([
            self._calculate_edge_error(u, v)
            for u, v in self.graph.out_edges(node)
        ])
        incoming_edge_errors = np.sum([
            self._calculate_edge_error(u, v)
            for u, v in self.graph.in_edges(node)
        ])
        return outgoing_edge_errors - incoming_edge_errors

    def _get_incident_edges(self, node: int) -> Iterator[int]:
        """Returns an iterator over the incident edges of the given node.

        Args:
            node: Node for which to return the neighbors.
        """
        return itertools.chain(self.graph.in_edges(node, data=True),
                               self.graph.out_edges(node, data=True))

    def _get_edge_measurement(self, u: int, v: int) -> float:
        """Gets the edge differential measurement along the given edge.

        Args:
            u: Outgoing node.
            v: Incoming node.

        Raises:
            KeyError: If the edge does not exist.
        """
        if not self.graph.has_edge(u, v):
            raise KeyError(f"({u}, {v})")
        return self.graph.edges[
            u, v][DIFFERENTIAL_MESH_GRID_EDGE_MEASUREMENT_ATTRIBUTE]

    def _calculate_edge_error(self, u: int, v: int) -> float:
        """Calculates the edge error between node u and node v.

        The edge error is defined as the difference between the adjacent node
        potentials and the edge differential measurement.

        Args:
            u: Outgoing node.
            v: Incoming node.

        Returns:
            The error at the given edge.

        Raises:
            KeyError: If the edge does not exist.
        """
        potential_difference = self._get_node_potential(
            u) - self._get_node_potential(v)
        edge_measurement = self._get_edge_measurement(u, v)
        return potential_difference - edge_measurement


class StochasticDifferentialMeshSolver(DifferentialMeshSolver):
    """Stochastic differential mesh solver.

    Attributes:
        max_error: Maximum error threshold for the error at each node to declare
            convergence.
    """

    def __init__(self,
                 grid: DifferentialMeshGrid,
                 verbose: bool = False,
                 max_error: float = 0.001):
        super().__init__(grid, verbose)
        self.max_error = max_error

    def solve(self) -> None:
        """Solves for the node potentials.

        The stochastic differential mesh solver randomly chooses nodes and sets
        the potential equal to its neighbors' estimates of the node's potential
        accordingly until the node potentials converge.

        For optimality, the node error of every node should be zero.
        """
        iteration = 0
        while not self._has_converged():
            iteration += 1

            # Choose a random node that is not the root node.
            node = random.choice([
                node for node in self.graph.nodes
                if node != DIFFERENTIAL_MESH_GRID_ROOT_NODE
            ])

            # Set the node potential to the average of its neighbors' estimates.
            potential = self._calculate_average_neighbor_potential(node)
            self._set_node_potential(node, potential)

            # If verbose, log the overall error.
            if self.verbose:
                logging.info("Iteration %d: MSE=%f", iteration,
                             self.calculate_mean_squared_error())

    def _calculate_average_neighbor_potential(self, node: int) -> float:
        """Calculates the node potential as the average of the neighbors'
        estimates.

        Args;
            node: Node for which to calculate the node potential.
        """
        # The average of the neighbors' estimates of the given node's potential
        # is equivalent to the sum of the neighboring node potentials and the
        # outgoing edge measurements minus the incoming edge measurements, all
        # divided by the node's degree.
        neighbor_potentials = np.sum([
            self._get_node_potential(neighbor)
            for neighbor in self._get_neighbors(node)
        ])
        outgoing_edge_measurements = np.sum([
            self._get_edge_measurement(u, v)
            for u, v in self.graph.out_edges(node)
        ])
        incoming_edge_measurements = np.sum([
            self._get_edge_measurement(u, v)
            for u, v in self.graph.in_edges(node)
        ])
        mean_estimate = (neighbor_potentials + outgoing_edge_measurements -
                         incoming_edge_measurements) / self.graph.degree(node)
        return mean_estimate

    def _has_converged(self) -> bool:
        """Returns whether the solver has converged on a solution.

        Convergence occurs once the error at each node is less than or equal to
        the maximum error.
        """
        for node in self.graph.nodes:
            if self._calculate_node_error(node) > self.max_error:
                return False
        return True


class MatrixDifferentialMeshSolver(DifferentialMeshSolver):
    """Matrix differential mesh solver.

    The order of the nodes is determined by the order of the nodes determined
    by the graph.
    """

    def __init__(self, grid: DifferentialMeshGrid, verbose: bool = False):
        super().__init__(grid, verbose)

    def solve(self) -> None:
        """Solves for the node potentials.

        The matrix differential mesh solver inverts a matrix to solve for the
        node potentials.
        """
        A = self._create_laplacian_matrix()
        b = self._create_edge_measurements_vector()
        node_potentials = np.linalg.solve(A, b)
        for node_index, node in enumerate(self.graph.nodes):
            self._set_node_potential(node, node_potentials[node_index])

    def _get_node_to_index_map(self) -> dict[int, int]:
        """Returns a map from the node to its index."""
        return {
            node: node_index for node_index, node in enumerate(self.graph.nodes)
        }

    def _create_laplacian_matrix(self) -> np.ndarray:
        """Creates the modified Laplacian matrix.

        The Laplacian matrix is modified by setting the row corresponding to
        the root node to an elementary basis vector because the root node has
        a potential of zero.

        Returns:
            The modified Laplacian matrix.
        """
        node_to_index_map = self._get_node_to_index_map()
        root_index = node_to_index_map[DIFFERENTIAL_MESH_GRID_ROOT_NODE]

        # Create the Laplacian matrix.
        laplacian_matrix = np.zeros(
            (self.graph.number_of_nodes(), self.graph.number_of_nodes()))
        for u, v in self.graph.edges():
            u_index = node_to_index_map[u]
            v_index = node_to_index_map[v]
            laplacian_matrix[u_index, v_index] -= 1
            laplacian_matrix[v_index, u_index] -= 1
            laplacian_matrix[u_index, u_index] += 1
            laplacian_matrix[v_index, v_index] += 1

        # Modify the Laplacian matrix.
        laplacian_matrix[root_index, :] = 0
        laplacian_matrix[root_index, root_index] = 1
        return laplacian_matrix

    def _create_edge_measurements_vector(self) -> np.ndarray:
        """Creates a vector consisting of the sum of edge differential
        measurements at each node.

        Returns:
            The vector of sums of edge differential measurements at each node.
        """
        node_to_index_map = self._get_node_to_index_map()
        root_index = node_to_index_map[DIFFERENTIAL_MESH_GRID_ROOT_NODE]

        edge_measurements_vector = np.zeros(self.graph.number_of_nodes())
        for node_index, node in enumerate(self.graph.nodes):
            outgoing_edge_measurements = np.sum([
                self._get_edge_measurement(u, v)
                for u, v in self.graph.out_edges(node)
            ])
            incoming_edge_measurements = np.sum([
                self._get_edge_measurement(u, v)
                for u, v in self.graph.in_edges(node)
            ])
            edge_measurements = outgoing_edge_measurements - incoming_edge_measurements
            edge_measurements_vector[node_index] = edge_measurements
        edge_measurements_vector[root_index] = 0
        return edge_measurements_vector


DIFFERENTIAL_MESH_SOLVERS = {
    "stochastic": StochasticDifferentialMeshSolver,
    "matrix": MatrixDifferentialMeshSolver,
}
