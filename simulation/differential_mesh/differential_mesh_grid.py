"""The differential mesh grid is a 2-dimensional directed grid graph, where the
directed weights between adjacent nodes represent the differential measurement.

The weight of the edge (i, j) represents the differential measurement V(i) - V(j).
The node potential is stored as a node attribute.
"""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Root node.
DIFFERENTIAL_MESH_GRID_ROOT_NODE = 1

# Node attribute for the node potential.
DIFFERENTIAL_MESH_GRID_NODE_POTENTIAL_ATTRIBUTE = "value"

# Edge attribute for the differential measurement.
DIFFERENTIAL_MESH_GRID_EDGE_MEASUREMENT_ATTRIBUTE = "weight"


class DifferentialMeshGrid:
    """2-dimensional directed grid graph.

    Attributes:
        graph: Graph object corresponding to the differential mesh grid.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def draw(self) -> None:
        """Draws the differential mesh grid."""
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        nx.draw_networkx(self.graph,
                         pos=pos,
                         ax=ax,
                         labels=nx.get_node_attributes(
                             self.graph,
                             DIFFERENTIAL_MESH_GRID_NODE_POTENTIAL_ATTRIBUTE))
        # nx.draw_networkx_labels(self.graph, pos=pos, ax=ax)
        nx.draw_networkx_edge_labels(self.graph, pos=pos, ax=ax)
        plt.show()

    @classmethod
    def read_edge_list(cls,
                       edge_list: str,
                       stddev: float = 0.0) -> "DifferentialMeshGrid":
        """Create a differential mesh grid from a file with the edge list.

        Args:
            path: Edge list filename.
            stddev: Standard deviation of the added noise.

        Returns:
            The differential mesh grid corresponding to the edge list.
        """
        with open(edge_list, "r") as f:
            graph = nx.read_weighted_edgelist(f,
                                              create_using=nx.DiGraph,
                                              nodetype=int)

        # Add noise to the differential measurements.
        if stddev > 0.0:
            for _, _, data in graph.edges(data=True):
                data["weight"] += np.random.normal(scale=stddev)
        return cls(graph)
