import numpy as np
import pyvista as pv
import fastfem.mesh as msh


class Plotter:
    def __init__(self, mesh):
        """
        Initialize the plotter with the mesh.

        Args:
            mesh: The mesh object.
        """
        self.mesh = mesh
        self.plotter = pv.Plotter()

    def plot_mesh(self):
        """
        Plots the mesh

        Args:
            None

        Returns:
            None
        """

        points = []
        for coordiante in self.mesh.coordinates_of_nodes:
            points.append(coordiante)

        strips = []
        for element in self.mesh.nodes_of_elements:
            strips.append(element)

        mesh = pv.PolyData(points, strips)
        pl = pv.Plotter()
        pl.add_mesh(mesh, color="lightblue", show_edges=True, edge_color="black")
        pl.add_point_labels(mesh.points, range(mesh.n_points))
        pl.camera_position = "yx"
        pl.camera.zoom(1.2)
        pl.show()
