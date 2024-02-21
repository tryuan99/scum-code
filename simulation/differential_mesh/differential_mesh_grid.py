"""The differential mesh grid is a 2-dimensional directed grid graph, where the
directed edge weights between incident nodes represent the differential
measurement.
"""

import networkx as nx
import numpy as np

from simulation.differential_mesh.differential_mesh_graph import (
    DIFFERENTIAL_MESH_GRAPH_NODE_POSITION_ATTRIBUTE, DifferentialMeshGraph)


class DifferentialMeshGrid(DifferentialMeshGraph):
    """2-dimensional directed grid graph.

    The graph should contain integer nodes from 1 to the total number of nodes,
    and all edges run along the horizontal or vertical grid lines.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        super().__init__(graph)
        self._validate_grid()
        self._place_nodes_in_grid()

    def _validate_grid(self) -> None:
        """Validates the grid.

        All edges of the rgid should run along horizontal or vertical grid
        lines.

        Raises:
            ValueError: If the grid is invalid.
        """
        # Validate that the edges run along horizontal or vertical grid lines.
        num_cols = self._get_number_of_columns()

        # Validate that the graph is a 2-dimensional grid.
        if self.graph.number_of_nodes() % num_cols != 0:
            raise ValueError("Graph is not a 2-dimensional grid.")

    def _get_number_of_rows(self) -> int:
        """Returns the number of rows in the graph."""
        num_cols = self._get_number_of_columns()
        return self.graph.number_of_nodes() // num_cols

    def _get_number_of_columns(self) -> int:
        """Returns the number of columns in the graph.

        Raises:
            ValueError: If the graph is not a grid.
        """
        vertical_edge_differences = np.array(
            [np.abs(u - v) for u, v in self.graph.edges if np.abs(u - v) != 1])
        if len(vertical_edge_differences) == 0:
            return self.graph.number_of_nodes()
        if not np.all(
                vertical_edge_differences == vertical_edge_differences[0]):
            raise ValueError(
                "Edges should run along horizontal or vertical grid lines.")
        return vertical_edge_differences[0]

    def _place_nodes_in_grid(self) -> None:
        """Place the nodes onto a grid for drawing.

        The position of each node is stored as a node attribute.
        """
        node_to_index_map = self.get_node_to_index_map()
        num_cols = self._get_number_of_columns()
        for node, data in self.graph.nodes(data=True):
            node_index = node_to_index_map[node]
            row = node_index // num_cols
            col = node_index % num_cols
            data[DIFFERENTIAL_MESH_GRAPH_NODE_POSITION_ATTRIBUTE] = (col, row)
