"""The differential mesh solver accepts a differential mesh graph and solves for
the node potentials given the differential edge measurements.

The solution ensures that each node takes on the average potential of what its
neighbors think the node is at. Equivalently, the sum of the incident edge
errors is 0 at every node, where the edge error is defined as the difference
between the incident nodes' potential difference and the differential
measurement.

The node potentials are written to the graph as node attributes.
"""

import random
from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
from absl import logging

from simulation.differential_mesh.differential_mesh_graph import (
    DIFFERENTIAL_MESH_GRAPH_ROOT_NODE, DifferentialMeshGraph)
from utils.priority_queue import PriorityQueue


class DifferentialMeshSolver(ABC):
    """Interface for a differential mesh solver.

    Attributes:
        graph: Differential mesh graph.
        verbose: If verbose, log verbose messages.
    """

    def __init__(self,
                 graph: DifferentialMeshGraph,
                 verbose: bool = False) -> None:
        self.graph = graph
        self.verbose = verbose
        self.graph.reset_node_potentials()

    @abstractmethod
    def solve(self) -> None:
        """Solves for the node potentials."""

    def calculate_mean_squared_error(self) -> float:
        """Returns the mean squared edge error."""
        mean_squared_error = np.mean([
            self._calculate_edge_error(u, v)**2
            for u, v in self.graph.graph.edges()
        ])
        return mean_squared_error

    def _calculate_node_error(self, node: int) -> float:
        """Calculates the node error of the given node.

        The node error is defined as the directed sum of the incident edge
        errors, i.e., the sum of the outgoing edge errors minus the sum of the
        incoming edge errors.

        Args:
            node: Node for which to calculate the error.

        Returns:
            The error at the given node.
        """
        outgoing_edge_errors = np.sum([
            self._calculate_edge_error(u, v)
            for u, v in self.graph.graph.out_edges(node)
        ])
        incoming_edge_errors = np.sum([
            self._calculate_edge_error(u, v)
            for u, v in self.graph.graph.in_edges(node)
        ])
        return outgoing_edge_errors - incoming_edge_errors

    def _calculate_edge_error(self, u: int, v: int) -> float:
        """Calculates the edge error between node u and node v.

        The edge error is defined as the difference between the incident nodes'
        potential difference and the differential edge measurement.

        Args:
            u: Outgoing node.
            v: Incoming node.

        Returns:
            The error at the given edge.

        Raises:
            KeyError: If the edge does not exist.
        """
        potential_difference = self.graph.get_node_potential(
            u) - self.graph.get_node_potential(v)
        edge_measurement = self.graph.get_edge_measurement(u, v)
        return potential_difference - edge_measurement


class MatrixDifferentialMeshSolver(DifferentialMeshSolver):
    """Matrix differential mesh solver.

    The matrix differential mesh solver solves Lx = b, where L is the graph's
    Laplacian matrix and b is the graph's edge measurements vector.
    The order of the nodes is determined by the order of the nodes determined
    by the graph.
    """

    def __init__(self,
                 graph: DifferentialMeshGraph,
                 verbose: bool = False) -> None:
        super().__init__(graph, verbose)

    def solve(self) -> None:
        """Solves for the node potentials."""
        node_to_index_map = self.graph.get_node_to_index_map()
        L = self.graph.create_laplacian_matrix()
        b = self.graph.create_edge_measurements_vector()

        # Modify the Laplacian matrix to set the reference node potential to 0.
        root_index = node_to_index_map[DIFFERENTIAL_MESH_GRAPH_ROOT_NODE]
        L[root_index, :] = 0
        L[root_index, root_index] = 1

        node_potentials = np.linalg.solve(L, b)
        for node, node_index in node_to_index_map.items():
            self.graph.set_node_potential(node, node_potentials[node_index])


class IterativeDifferentialMeshSolver(DifferentialMeshSolver, ABC):
    """Interface for a iterative differential mesh solver.

    Attributes:
        max_error: Maximum node error to declare convergence.
        max_num_iterations: Maximum number of iterations.
    """

    def __init__(self,
                 graph: DifferentialMeshGraph,
                 verbose: bool = False,
                 max_error: float = 0.001,
                 max_num_iterations: int = None) -> None:
        super().__init__(graph, verbose)
        self.max_error = max_error
        self.max_num_iterations = max_num_iterations or np.iinfo(int).max

    def solve(self) -> None:
        """Solves for the node potentials.

        The iterative differential mesh solver chooses a node and sets its
        potential equal to the average of its neighbors' estimates of the
        node's potential. This process repeats until the node potentials
        converge.

        For optimality, the node error should be zero at every node.
        """
        iteration = 0
        while not self._has_converged() and iteration < self.max_num_iterations:
            iteration += 1

            # Class-specific logic for choosing a node.
            node = self._choose_node()

            # Set the node potential to the average of its neighbors' estimates.
            potential = self._calculate_average_neighbor_potential(node)
            self.graph.set_node_potential(node, potential)

            # Class-specific logic after updating the node potential.
            self._post_update_node(node)

            # If verbose, log the overall mean squared edge error.
            if self.verbose:
                logging.info("Iteration %d: MSE=%f", iteration,
                             self.calculate_mean_squared_error())

        if not self._has_converged():
            logging.warning(
                "Iterative differential mesh solver did not converge.")

        # Subtract the root node's potential from all node potentials. This is
        # necessary because the root node may be selected and have its
        # potential updated.
        root_potential = self.graph.get_node_potential(
            DIFFERENTIAL_MESH_GRAPH_ROOT_NODE)
        for node in self.graph.graph.nodes:
            potential = self.graph.get_node_potential(node)
            self.graph.set_node_potential(node, potential - root_potential)

    @abstractmethod
    def _choose_node(self) -> int:
        """Returns a node to update for the current iteration."""

    def _post_update_node(self, node: int) -> None:
        """Callback function after updating the given node's potential.

        Args:
            node: Node that had its potential updated.
        """
        return

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
            self.graph.get_node_potential(neighbor)
            for neighbor in self.graph.get_neighbors(node)
        ])
        outgoing_edge_measurements = np.sum([
            self.graph.get_edge_measurement(u, v)
            for u, v in self.graph.graph.out_edges(node)
        ])
        incoming_edge_measurements = np.sum([
            self.graph.get_edge_measurement(u, v)
            for u, v in self.graph.graph.in_edges(node)
        ])
        mean_estimate = (
            neighbor_potentials + outgoing_edge_measurements -
            incoming_edge_measurements) / self.graph.graph.degree(node)
        return mean_estimate

    def _has_converged(self) -> bool:
        """Returns whether the solver has converged on a solution.

        Convergence occurs once the node error is less than or equal to the
        maximum error at every node.
        """
        for node in self.graph.graph.nodes:
            if np.abs(self._calculate_node_error(node)) > self.max_error:
                return False
        return True


class PriorityDifferentialMeshSolver(IterativeDifferentialMeshSolver):
    """Priority differential mesh solver."""

    def __init__(self,
                 graph: DifferentialMeshGraph,
                 verbose: bool = False,
                 max_error: float = 0.001,
                 max_num_iterations: int = None) -> None:
        super().__init__(graph, verbose, max_error, max_num_iterations)

        # Initialize the node priority queue.
        self.node_queue = PriorityQueue(self.graph.graph.number_of_nodes())
        for node in self.graph.graph.nodes:
            error = self._calculate_node_error(node)
            self.node_queue.add(node, -np.abs(error))

    def _choose_node(self) -> int:
        """Returns a node to update for the current iteration.

        The priority differential mesh solver chooses the node with the highest
        error.
        """
        node, _ = self.node_queue.peek()
        return node

    def _post_update_node(self, node: int) -> None:
        """Callback function after updating the given node's potential.

        Args:
            node: Node that had its potential updated.
        """
        # Update the node's priority as well as those of its neighbors in the
        # priority queue.
        self.node_queue.update(node, 0)
        for neighbor in self.graph.get_neighbors(node):
            error = self._calculate_node_error(neighbor)
            self.node_queue.update(neighbor, -np.abs(error))

    def _has_converged(self) -> bool:
        """Returns whether the solver has converged on a solution.

        Convergence occurs once the node error is less than or equal to the
        maximum error at every node.
        """
        _, max_node_error = self.node_queue.peek()
        return np.abs(max_node_error) <= self.max_error


class StochasticDifferentialMeshSolver(IterativeDifferentialMeshSolver):
    """Stochastic differential mesh solver."""

    def __init__(self,
                 graph: DifferentialMeshGraph,
                 verbose: bool = False,
                 max_error: float = 0.001,
                 max_num_iterations: int = None) -> None:
        super().__init__(graph, verbose, max_error, max_num_iterations)

    def _choose_node(self) -> int:
        """Returns a node to update for the current iteration.

        The stochastic differential mesh solver randomly chooses the node.
        """
        nodes = list(self.graph.graph.nodes)
        while True:
            node = random.choice(nodes)
            if np.abs(self._calculate_node_error(node)) > self.max_error:
                return node


DIFFERENTIAL_MESH_SOLVERS = {
    "matrix": MatrixDifferentialMeshSolver,
    "priority": PriorityDifferentialMeshSolver,
    "stochastic": StochasticDifferentialMeshSolver,
}
