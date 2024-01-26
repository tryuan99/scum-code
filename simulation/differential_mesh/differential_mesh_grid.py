"""The differential mesh grid is a 2-dimensional directed grid graph, where the
directed edge weights between incident nodes represent the differential
measurement.

The weight of the edge (i, j) represents the differential measurement V(i) - V(j).
The node potential is stored as a node attribute.
"""

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Root node.
DIFFERENTIAL_MESH_GRID_ROOT_NODE = 1

# Node attribute for the node potential.
DIFFERENTIAL_MESH_GRID_NODE_POTENTIAL_ATTRIBUTE = "value"

# Node attribute for the node position.
DIFFERENTIAL_MESH_GRID_NODE_POSITION_ATTRIBUTE = "pos"

# Edge attribute for the differential measurement.
DIFFERENTIAL_MESH_GRID_EDGE_MEASUREMENT_ATTRIBUTE = "weight"

# Default edge drawing color.
DIFFERENTIAL_MESH_GRID_EDGE_COLOR = "#1f78b4"


class DifferentialMeshGrid:
    """2-dimensional directed grid graph.

    The graph should contain integer nodes from 1 to the total number of nodes,
    and all edges run along the horizontal or vertical grid lines.

    Attributes:
        graph: Graph object corresponding to the differential mesh grid.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph.copy()
        self._validate_graph()
        self._place_nodes_in_grid()

    def add_edge_measurement_noise(self, stddev: float) -> None:
        """Adds noise to the edge measurements.

        Args:
            stddev: Standard deviation of the added noise.
        """
        for _, _, data in self.graph.edges(data=True):
            data[
                DIFFERENTIAL_MESH_GRID_EDGE_MEASUREMENT_ATTRIBUTE] += np.random.normal(
                    scale=stddev)

    def draw(self) -> None:
        """Draws the differential mesh grid."""
        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.get_node_attributes(
            self.graph, DIFFERENTIAL_MESH_GRID_NODE_POSITION_ATTRIBUTE)

        # Round the node potentials for drawing.
        potentials = nx.get_node_attributes(
            self.graph, DIFFERENTIAL_MESH_GRID_NODE_POTENTIAL_ATTRIBUTE)
        if potentials:
            self._round_map_values(potentials)
            node_labels = potentials
        else:
            node_labels = {node: node for node in self.graph.nodes}

        # Round the edge measurements for drawing.
        measurements = nx.get_edge_attributes(
            self.graph, DIFFERENTIAL_MESH_GRID_EDGE_MEASUREMENT_ATTRIBUTE)
        self._round_map_values(measurements)

        # Draw the graph with node and edge labels.
        nx.draw_networkx_nodes(self.graph,
                               pos=pos,
                               ax=ax,
                               node_color="none",
                               node_size=1000)
        nx.draw_networkx_labels(self.graph, pos=pos, ax=ax, labels=node_labels)
        nx.draw_networkx_edges(self.graph,
                               pos=pos,
                               ax=ax,
                               edge_color=DIFFERENTIAL_MESH_GRID_EDGE_COLOR,
                               node_size=1000)
        nx.draw_networkx_edge_labels(
            self.graph,
            pos=pos,
            ax=ax,
            edge_labels=measurements,
            font_color=DIFFERENTIAL_MESH_GRID_EDGE_COLOR)
        plt.show()

    def output_spice_netlist(self, netlist: str, target_node: int = -1) -> None:
        """Outputs the SPICE netlist corresponding to the differential mesh
        grid.

        Each edge is represented by a 1 Ohm resistor, and the root node is
        grounded. A 1 A current source is connected to the target node, and the
        resulting node potential is equal to the standard error of the node
        potential.

        Args:
            netlist: Netlist file to output.
            node: Target node.
        """
        # By default, the target node is the furthest node from the reference node.
        if target_node not in self.graph.nodes:
            target_node = self.graph.number_of_nodes()

        with open(netlist, "w") as f:
            # SPICE title card.
            num_rows = self._get_number_of_rows()
            num_cols = self._get_number_of_columns()
            f.write(f"* SPICE netlist for {num_rows}x{num_cols} grid\n")

            # Add a 1 Ohm resistor for each edge.
            resistor_index = 1
            for u, v in self.graph.edges:
                f.write(f"R{resistor_index} {u} {v} {{R}}\n")
                resistor_index += 1

            # Add the reference node.
            f.write(f"V 0 {DIFFERENTIAL_MESH_GRID_ROOT_NODE} 0\n")

            # Add a 1 A current source to the target node.
            f.write(f"I 0 {target_node} 1\n")

            # Add a DC analysis, define the resistance, and print the node
            # potential.
            f.write(".dc I 1 1 1\n")
            f.write(".param R=1\n")
            f.write(f".print dc v({target_node})\n")
            f.write(".end\n")

    def _validate_graph(self) -> None:
        """Validates the graph.

        The graph should contain integer nodes from 1 to the total number of
        nodes, and all edges should run along horizontal or vertical grid
        lines.

        Raises:
            ValueError: If the graph is invalid.
        """
        # Validate the nodes are labeled from 1 to the total number of nodes.
        if min(self.graph.nodes) != DIFFERENTIAL_MESH_GRID_ROOT_NODE:
            raise ValueError(f"Minimum node should be labeled "
                             f"{DIFFERENTIAL_MESH_GRID_ROOT_NODE}.")
        if max(self.graph.nodes) != self.graph.number_of_nodes():
            raise ValueError(f"Node should be labeled consecutively from "
                             f"{DIFFERENTIAL_MESH_GRID_ROOT_NODE}.")

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
        num_cols = self._get_number_of_columns()
        for node, data in self.graph.nodes(data=True):
            node_index = node - 1
            row = node_index // num_cols
            col = node_index % num_cols
            data[DIFFERENTIAL_MESH_GRID_NODE_POSITION_ATTRIBUTE] = (col, row)

    @staticmethod
    def _round_map_values(input_map: dict[Any, float],
                          decimals: int = 6) -> None:
        """Rounds the map values to the given number of decimals.

        Args:
            a: Input dictionary.
            decimals: Number of decimals to round to.
        """
        for key in input_map:
            input_map[key] = np.round(input_map[key], decimals)
