"""The differential mesh graph factory creates a 2-dimensional directed grid
graph for a differential mesh grid.
"""

import networkx as nx
import numpy as np

from simulation.differential_mesh.differential_mesh_grid import (
    DIFFERENTIAL_MESH_GRID_EDGE_MEASUREMENT_ATTRIBUTE,
    DIFFERENTIAL_MESH_GRID_ROOT_NODE)


class DifferentialMeshGraphFactory:
    """2-dimensional directed grid graph factory."""

    @classmethod
    def create_zero_2d_graph(cls,
                             num_rows: int,
                             num_cols: int,
                             noise: float = 0) -> nx.DiGraph:
        """Creates a 2-dimensional directed grid graph with zero edge weight.

        Args:
            num_rows: Number of rows.
            num_cols: Number of columns.
            noise: Standard deviation of the added noise.

        Returns:
            The zero 2-dimensional directed grid graph with the given dimensions.

        Raises:
            ValueError: If the number of rows or number of columns is less than
              1.
        """
        if num_rows < 1:
            raise ValueError("Number of rows cannot be less than 1.")
        if num_cols < 1:
            raise ValueError("Number of columns cannot be less than 1.")

        graph = nx.DiGraph()

        # Add the horizontal edges.
        node_index = DIFFERENTIAL_MESH_GRID_ROOT_NODE
        for _ in range(num_rows):
            for _ in range(num_cols - 1):
                graph.add_edge(
                    node_index, node_index + 1,
                    **{DIFFERENTIAL_MESH_GRID_EDGE_MEASUREMENT_ATTRIBUTE: 0})
                node_index += 1
            node_index += 1

        # Add the vertical edges.
        node_index = DIFFERENTIAL_MESH_GRID_ROOT_NODE
        for _ in range(num_rows - 1):
            for _ in range(num_cols):
                graph.add_edge(
                    node_index, node_index + num_cols,
                    **{DIFFERENTIAL_MESH_GRID_EDGE_MEASUREMENT_ATTRIBUTE: 0})
                node_index += 1

        if noise > 0:
            DifferentialMeshGraphFactory._add_edge_measurement_noise(
                graph, noise)
        return graph

    @classmethod
    def create_from_edge_list(cls,
                              edge_list: str,
                              noise: float = 0) -> nx.DiGraph:
        """Creates a 2-dimensional directed grid graph from an edge list file.

        Args:
            path: Edge list filename.
            noise: Standard deviation of the added noise.

        Returns:
            The 2-dimensional directed grid graph corresponding to the edge list.
        """
        with open(edge_list, "r") as f:
            graph = nx.read_weighted_edgelist(f,
                                              create_using=nx.DiGraph,
                                              nodetype=int)

        if noise > 0:
            DifferentialMeshGraphFactory._add_edge_measurement_noise(
                graph, noise)
        return graph

    @classmethod
    def _add_edge_measurement_noise(cls, graph: nx.DiGraph,
                                    stddev: float) -> None:
        """Adds noise to the edge measurements.

        Args:
            stddev: Standard deviation of the added noise.
        """
        for _, _, data in graph.edges(data=True):
            data[
                DIFFERENTIAL_MESH_GRID_EDGE_MEASUREMENT_ATTRIBUTE] += np.random.normal(
                    scale=stddev)
