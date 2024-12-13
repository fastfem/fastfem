import time

import numpy as np
import pyvista as pv
from pyvista import CellType

import fastfem.mesh as m


class VisualMesh:
    def __init__(self, mesh: m.Mesh) -> None:
        """
        Initialize the plotter with the mesh.


        Args:
            mesh: The mesh object.
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
        return pv.UnstructuredGrid(cell_arr, cell_types, points)

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
            point_label: Boolean value to determine whether the points are labeled.
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
            formatted_labels = [f"({p[0]:.2f}, {p[1]:.2f})" for p in points[mask]]
            plotter.add_point_labels(
                points[mask], formatted_labels, point_size=20, font_size=10
            )
        plotter.camera_position = "xy"
        plotter.show()

    def plot_data(self, data: np.ndarray, cmap: str = "viridis") -> None:
        """
        Plots the mesh with temperature data, for a single time step.

        Args:
            data: The temperature data for each node, contained in a 2D array.
            cmap: Colormap for the data.
        """
        # Define the grid and plotter objects
        grid = self.define_plotter()
        plotter = pv.Plotter()

        # Assign temperature data to the grid
        grid.point_data["Data"] = data.flatten()
        plotter.add_mesh(
            mesh=grid,
            scalars="Data",
            cmap=cmap,
            scalar_bar_args={
                "vertical": False,
                "title": "Data",
                "title_font_size": 25,
                "label_font_size": 10,
            },
        )
        plotter.camera_position = "xy"
        plotter.show()

    def animate_data(
        self,
        fps: float,
        total_time: float,
        data: np.ndarray,
        force: bool = False,
        cmap: str = "viridis",
    ) -> None:
        """
        Plots the mesh with temperature data, for successive time steps.

        Args:
            fps: Frames per second for the animation.
            total_time: Total time for the animation.
            data: The temperature data for each node, contained in a 3D array.
            cmap: Colormap for the data.
        """
        if fps > 25 and not force:
            message = "The maximum value for fps is 25. Please decrease your fps value."
            raise ValueError(message)

        # Define the grid and plotter objects
        grid = self.define_plotter()
        plotter = pv.Plotter()

        # Defining the number of time steps
        frames = int(total_time * fps)

        # Assign temperature data to the grid
        grid.point_data["Data"] = data[0].flatten()
        plotter.add_mesh(
            mesh=grid,
            color="red",
            render_points_as_spheres=True,
            point_size=10,
            scalars="Data",
            cmap=cmap,
            scalar_bar_args={
                "title_font_size": 20,
                "label_font_size": 16,
                "n_labels": 3,
                "italic": True,
                "fmt": "%.2f",
                "font_family": "arial",
            },
        )

        # Setting plotter settings
        plotter.camera_position = "xy"
        plotter.show(auto_close=False, interactive_update=True)
        plotter.render()
        text_actor = None

        # Animate
        for i in range(1, frames):
            # Starting a timer, accounting for the time taken to render the frame
            start_time = time.time()

            # Update the data
            grid.point_data["Data"] = data[i].flatten()
            plotter.render()

            # Update time text
            if text_actor is not None:
                plotter.remove_actor(text_actor)  # type:ignore
            text_actor = plotter.add_text(
                f"t = {i / fps:.2f}", position="upper_left", font_size=26, color="black"
            )

            # Ending the timer, waiting accordingly
            end_time = time.time()
            remaining_time = 1 / fps - (end_time - start_time)
            if remaining_time > 0:
                time.sleep(remaining_time)

    def make_movie(
        self,
        filename: str,
        fps: float,
        total_time: float,
        data: np.ndarray,
        cmap: str = "viridis",
    ) -> None:
        """
        Creates a movie with the given mesh and temperature data.

        Args:
            filename: The name of the file to be created.
            fps: Frames per second for the movie.
            total_time: Total time for the movie.
            data: The temperature data for each node, contained in a 2D array.
            cmap: Colormap for the data.
        """
        # Define the grid and plotter objects
        grid = self.define_plotter()
        plotter = pv.Plotter(off_screen=True)

        # Defining the number of time steps
        frames = int(total_time * fps)

        # Assign initial temperature data
        grid.point_data["Data"] = data[0].flatten("F")
        plotter.add_mesh(grid, scalars="Data", cmap=cmap)
        plotter.camera_position = "xy"

        # Creating file
        plotter.open_movie(filename)
        # plotter.show()
        plotter.show(auto_close=False)
        plotter.write_frame()
        text = None

        # Animate and record frames
        for i in range(1, frames):
            # Update the data
            grid.point_data["Data"] = data[i].flatten("F")
            plotter.render()

            # Update time text
            if text is not None:
                plotter.remove_actor(text)  # type:ignore
            text = plotter.add_text(
                f"t = {i / fps:.2f}", position="upper_left", font_size=20
            )
            plotter.write_frame()

        plotter.close()

    def make_gif(
        self,
        filename: str,
        fps: float,
        total_time: float,
        data: np.ndarray,
        cmap: str = "viridis",
    ) -> None:
        """
        Creates a GIF with the given mesh and temperature data.

        Args:
            filename: The name of the file to be created.
            fps: Frames per second for the movie.
            total_time: Total time for the movie.
            data: The temperature data for each node, contained in a 2D array.
            cmap: Colormap for the data.
        """
        # Define the grid and plotter objects
        grid = self.define_plotter()
        plotter = pv.Plotter(off_screen=True)

        # Defining the number of time steps
        frames = int(total_time * fps)

        # Assign initial temperature data
        grid.point_data["Data"] = data[0].flatten("F")
        plotter.add_mesh(grid, scalars="Data", cmap=cmap)
        plotter.camera_position = "xy"

        # Creating file
        plotter.open_gif(filename)
        # plotter.show(auto_close=False)
        # plotter.write_frame()
        text = None

        for i in range(1, frames):
            # Update the data
            grid.point_data["Data"] = data[i].flatten("F")
            plotter.render()

            # Update time text
            if text is not None:
                plotter.remove_actor(text)  # type:ignore
            text = plotter.add_text(
                f"t = {i / fps:.2f}", position="upper_left", font_size=20
            )
            plotter.write_frame()

        plotter.close()
