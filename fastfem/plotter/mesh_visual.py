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


    def define_plotter(self) -> tuple[pv.UnstructuredGrid, pv.Plotter]:
        """
        Plots the mesh


        Args:
            Color: List containing the colors of the mesh and edges, respectively.
            point_label: Boolean value to determine whether the points are labeled or not.

        Returns:
            None
        """

        # Recovering element type
        if self.mesh["surface"].mesh[0].type == "triangle":
            cell_type = CellType.TRIANGLE
            side_number = 3
        else:
            cell_type = CellType.QUAD
            side_number = 4

        # Defining the points and strips
        points = np.array(list(self.mesh["surface"].mesh[0].nodes.values()))
        strips = np.array(list(self.mesh["surface"].mesh[0].elements.values()))

        # 0-indexing the strips
        strips_flat = strips.ravel() - 1
        cells = np.insert(strips_flat, np.arange(0, len(strips_flat), side_number), side_number)
        cell_arr = np.array(cells, dtype=np.int32)
        cell_types = np.full(len(strips), cell_type, dtype=np.uint8) 

        # Create the unstructured grid
        grid = pv.UnstructuredGrid(cell_arr, cell_types, points)
        plotter = pv.Plotter()

        return grid, plotter


    def plot_mesh(self, point_label=None, mesh_color=None, edge_color=None, edge_thickness=None) -> None:
        """
        Plots the mesh


        Args:
            mesh_color: Color of the mesh.
            edge_color: Color of the edges.
            edge_thickness: Thickness of the edges.
            point_label: Boolean value to determine whether the points are labeled or not.

        Returns:
            None
        """

        # Check aesthetics parameters
        if point_label is None:
            point_label = False
        if mesh_color is None:
            mesh_color = "white"
        if edge_color is None:
            edge_color = "black"
        if edge_thickness is None:
            edge_thickness = 1
        
        # Define the grid and plotter objects
        grid = self.define_plotter()[0]
        plotter = self.define_plotter()[1]
        plotter.add_mesh(grid, show_edges=True, color=mesh_color, edge_color=edge_color, line_width=edge_thickness) 

        # Add point labels, if specified
        points = grid.points
        if point_label:
            mask = points[:, 2] == 0 # Labeling points on xy plane
            plotter.add_point_labels(points[mask], points[mask].tolist(), point_size=20, font_size=10)
        plotter.camera_position = 'xy'
        plotter.show()


    def plot_data(self, data: np.ndarray) -> None:
        """
        Plots the mesh with temperature data, for a single time step.

        Args:
            data: The temperature data for each node.

        Returns:
            None

        """

        # Define the grid and plotter objects
        grid = self.define_plotter()[0]
        plotter = self.define_plotter()[1]

        # Assign temperature data to the grid
        grid.point_data['temperature'] = data.flatten('F')
        plotter.add_mesh(grid, scalars=grid.points, cmap='coolwarm')
        plotter.camera_position = 'xy'
        plotter.show()
    

    def animate_mesh(self, data: tuple[float, ...], file_path: str) -> None:
        pass


    def make_gif(self):
        pass


    def save(self, file_name):
        pass
