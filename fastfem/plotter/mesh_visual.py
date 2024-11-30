import numpy as np
import pyvista as pv
from pyvista import CellType

class VisualMesh:
    def __init__(self, mesh):
        """
        Initialize the plotter with the mesh.


        Args:
            mesh: The mesh object.
        """
        self.mesh = mesh


    def plot_mesh(self, point_label: bool, colors: str) -> None:
        """
        Plots the mesh


        Args:
            Color: List containing the colors of the mesh and edges, respectively.
            point_label: Boolean value to determine whether the points are labeled or not.

        Returns:
            None
        """
        # Defining the points and strips
        points = np.array(list(self.mesh["surface"].mesh[0].nodes.values()))
        strips = np.array(list(self.mesh["surface"].mesh[0].elements.values()))

        # 0-indexing the strips
        strips_flat = strips.ravel() - 1
        cells = np.insert(strips_flat, np.arange(0, len(strips_flat), 3), 3)
        cell_arr = np.array(cells, dtype=np.int32)
        cell_types = np.full(len(strips), CellType.TRIANGLE, dtype=np.uint8) 

        # Create the unstructured grid
        grid = pv.UnstructuredGrid(cell_arr, cell_types, points)
        plotter = pv.Plotter()
        plotter.add_mesh(grid, show_edges=True, color=colors)

        points = grid.points
        mask = points[:, 2] == 0 # Labeling points on xy plane
        plotter.add_point_labels(points[mask], points[mask].tolist(), point_size=20, font_size=10)
        plotter.camera_position = 'xy'
        plotter.show()


    def movie(self, resolution, fps, colors):
        pass




    def gif(self, resolution):
        pass




    def save(self, file_name):
        pass
