import numpy as np
import pyvista as pv
from pyvista import CellType
import time


class VisualMesh:
    def __init__(self, mesh):
        """
        Initialize the plotter with the mesh.


        Args:
            mesh: The mesh object.

        Returns:
            None
        """

        self.mesh = mesh

    def define_plotter(self) -> pv.UnstructuredGrid:
        """
        Given the Mesh object, redefines it for PyVista plotting.

        Returns:
            grid: PyVista grid object.
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
        cells = np.insert(
            strips_flat, np.arange(0, len(strips_flat), side_number), side_number
        )
        cell_arr = np.array(cells, dtype=np.int32)
        cell_types = np.full(len(strips), cell_type, dtype=np.uint8)

        # Create the unstructured grid
        grid = pv.UnstructuredGrid(cell_arr, cell_types, points)

        return grid

    def plot_mesh(
        self,
        point_label: bool = False,
        mesh_color: str = "white",
        edge_color: str = "black",
        edge_thickness: int = 1,
    ) -> None:
        """
        Plots the mesh


        Args:
            mesh_color: Color of the mesh.
            edge_color: Color of the edges.
            edge_thickness: Thickness of the edges.
            point_label: Boolean value to determine whether the points are labeled or not.
        """
        # Define the grid and plotter objects
        grid = self.define_plotter()
        plotter = pv.Plotter()
        plotter.add_mesh(
            grid,
            show_edges=True,
            color=mesh_color,
            edge_color=edge_color,
            line_width=edge_thickness,
        )

        # Add point labels, if specified
        points = grid.points

        if point_label:
            mask = points[:, 2] == 0  # Labeling points on xy plane
            plotter.add_point_labels(
                points[mask], points[mask].tolist(), point_size=20, font_size=10
            )
        plotter.camera_position = "xy"
        plotter.show()

    def plot_data(self, data: np.ndarray, show=True) -> None:
        """
        Plots the mesh with temperature data, for a single time step.

        Args:
            data: The temperature data for each node, contained in a 1D array.
        """
        # Define the grid and plotter objects
        grid = self.define_plotter()
        plotter = pv.Plotter()

        # Assign temperature data to the grid
        grid.point_data["Temperature"] = data.flatten("F")
        plotter.add_mesh(grid, scalars="Temperature", cmap="coolwarm")
        plotter.camera_position = "xy"

        if show:
            plotter.show()

    def animate_mesh(
        self, fps: float, frames: int, data: np.ndarray, cmap: str = "viridis"
    ) -> None:
        """
        Plots the mesh with temperature data, for successive time steps.

        Args:
            fps: Frames per second for the animation.
            frames: Number of time steps (frames).
            data: The temperature data for each node, contained in a 2D array.
            cmap: Colormap for the data.
        """
        # Define the grid and plotter objects
        grid = self.define_plotter()
        plotter = pv.Plotter()

        # Assign temperature data to the grid
        grid.point_data["Temperature"] = data[0].flatten("F")
        plotter.add_mesh(grid, scalars="Temperature", cmap=cmap)

        # Animate
        plotter.camera_position = "xy"
        plotter.show(auto_close=False, interactive_update=True)
        plotter.update()

        for i in range(1, frames):
            grid.point_data["Temperature"] = data[i].flatten("F")
            plotter.update_scalars(data[i].flatten("F"), mesh=grid, render=True)
            plotter.update()
            time.sleep(1 / fps)

        plotter.close()

    def make_gif(self):
        pass


    def save(self, file_name):
        pass
