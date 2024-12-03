import numpy as np
import pyvista as pv


class VisualMesh:
    def __init__(self, mesh):
        """
        Initialize the plotter with the mesh.


        Args:
            mesh: The mesh object.
        """
        self.mesh = mesh[0]
        self.mesh_type = mesh[0].type

    def plot_mesh(self, point_label: bool, colors) -> None:
        """
        Plots the mesh


        Args:
            Color: List containing the colors of the mesh and edges, respectively.


        Returns:
            None
        """

        if self.mesh_type == "triangle":
            number_of_nodes_per_element = 3
        else:
            number_of_nodes_per_element = 4

        # Declaring coordinates of nodes
        points = self.mesh.coordinates_of_nodes
        points = points.reshape(len(points) // number_of_nodes_per_element, 3)

        # Defining strips for the mesh
        new_tags = np.array([i - 1 for i in self.mesh.nodes_of_elements])
        cells = np.concatenate(([len(new_tags)], new_tags), dtype=int)

        # Defining the mesh object for PyVista
        mesh = pv.PolyData(points, strips=cells)

        # Plotting
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, show_edges=True, color="lightblue")
        if point_label:
            plotter.add_point_labels(mesh.points, range(mesh.n_points))
        plotter.camera_position = "yx"
        plotter.show()

    def movie(self, resolution, fps, colors):
        pass

    def gif(self, resolution):
        pass

    def save(self, file_name):
        pass
