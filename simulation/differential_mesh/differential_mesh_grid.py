"""The differential mesh grid is a 2-dimensional directed grid graph, where the
directed weights between adjacent nodes represent the differential measurement.

The weight of the edge (i, j) represents the differential measurement V(i) - V(j).
The node potential is stored as a node attribute.
"""

import matplotlib.pyplot as plt
import networkx as nx

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
        nx.draw_networkx_edge_labels(self.graph, pos=pos, ax=ax)
        plt.show()
