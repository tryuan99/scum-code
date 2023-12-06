from absl import app, flags

FLAGS = flags.FLAGS

from simulation.differential_mesh.differential_mesh_grid import \
    DifferentialMeshGrid


def main(argv):
    assert len(argv) == 1

    grid = DifferentialMeshGrid.read_edge_list(FLAGS.edgelist)
    grid.draw()


if __name__ == "__main__":
    flags.DEFINE_string(
        "edgelist", "simulation/differential_mesh/data/example_2x2.edgelist",
        "Input edge list.")

    app.run(main)
