import numpy as np
import pyvista as pv
from unittest.mock import MagicMock
import pytest

import fastfem.mesh as m
import fastfem.plotter as p

# Relevant constants for mesh building/videos
nh = 10
nv = 10
total_time = 2
fps = 25
time_steps = int(total_time * fps)


@pytest.fixture(params=["triangle", "quadrangle"])
def mesh(request: pytest.FixtureRequest) -> m.Mesh:
    """
    Fixture to create a mesh with different element types.

    Args:
        request: Pytest fixture.

    Returns:
        m.Mesh: The mesh object.
    """
    return m.create_a_rectangle_mesh(
        horizontal_length=1,
        vertical_length=1,
        nodes_in_horizontal_direction=nh,
        nodes_in_vertical_direction=nv,
        element_type=request.param,
    )


@pytest.fixture(params=[1, 2, 3])
def dummy_data() -> np.ndarray:
    """
    Fixture to create dummy data for testing.

    Returns:
        np.ndarray: Data.
    """
    left_temp = np.random.randint(0, 10)
    right_temp = np.random.randint(0, 10)
    top_temp = np.random.randint(0, 10)
    bottom_temp = np.random.randint(0, 10)
    min_temp = np.random.randint(10, 20)
    max_temp = np.random.randint(20, 30)
    data = np.random.uniform(low=min_temp, high=max_temp, size=(time_steps, nv, nh))
    for i in range(time_steps):
        data[i, 0, :] = bottom_temp
        data[i, -1, :] = top_temp
        data[i, :, 0] = left_temp
        data[i, :, -1] = right_temp
    return data


def test_define_plotter(mesh: m.Mesh) -> None:
    """
    Tests if the VisualMesh class defines a mesh properly.

    Args:
        mesh: The mesh object.
    """
    visualizer = p.VisualMesh(mesh)
    grid = visualizer.define_plotter()
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells > 0
    assert grid.n_points == mesh.number_of_nodes


@pytest.mark.parametrize("point_label", [True, False])
@pytest.mark.parametrize(
    "color", ["white", "black", "red", "green", "blue", "yellow", "cyan", "magenta"]
)
@pytest.mark.parametrize("edge_thickness", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_plot_mesh(
    monkeypatch: pytest.MonkeyPatch,
    mesh: m.Mesh,
    point_label: bool,
    color: str,
    edge_thickness: int,
) -> None:
    """
    Tests if the mesh is plotted properly, without data.

    Args:
        monkeypatch: Pytest fixture.
        mesh: The mesh object.
        point_label: Boolean value to determine whether the points are labeled.
        color: The color of the mesh/edges.
        edge_thickness: Thickness of the edges.
    """
    visualizer = p.VisualMesh(mesh)
    monkeypatch.setattr(pv.Plotter, "show", MagicMock())
    visualizer.plot_mesh(
        point_label=point_label,
        mesh_color=color,
        edge_color=color,
        edge_thickness=edge_thickness,
    )


# @pytest.mark.parametrize("cmap", ["viridis", "plasma", "inferno", "magma", "cividis"])
def test_plot_data(
    monkeypatch: pytest.MonkeyPatch, mesh: m.Mesh, dummy_data: np.ndarray
) -> None:
    """
    Tests if the mesh is plotted properly, for a single frame.

    Args:
        monkeypatch: Pytest fixture.
        mesh: The mesh object.
        dummy_data: The temperature data for each node, contained in a 2D array.
    """
    visualizer = p.VisualMesh(mesh)
    monkeypatch.setattr(pv.Plotter, "show", MagicMock())
    visualizer.plot_data(
        data=dummy_data[0],
    )


def test_animate_data(
    monkeypatch: pytest.MonkeyPatch,
    mesh: m.Mesh,
    dummy_data: np.ndarray,
) -> None:
    """
    Tests if the mesh is animated properly/

    Args:
        monkeypatch: Pytest fixture.
        mesh: The mesh object.
        dummy_data: The temperature data for each node, contained in a 3D array.
    """
    visualizer = p.VisualMesh(mesh)
    monkeypatch.setattr(pv.Plotter, "show", MagicMock())
    visualizer.animate_data(
        fps,
        total_time,
        dummy_data,
    )
    with pytest.raises(ValueError):
        visualizer.animate_data(
            fps=np.random.uniform(25.1, 100),
            total_time=total_time,
            data=dummy_data,
            cmap="invalid_cmap",
        )


# def test_make_movie(
#     monkeypatch: pytest.MonkeyPatch, mesh: m.Mesh, dummy_data: np.ndarray
# ) -> None:
#     """
#     Tests if the movie is created.

#     Args:
#         monkeypatch: Pytest fixture.
#         mesh: The mesh object.
#         dummy_data: The temperature data for each node, contained in a 3D array.
#     """
#     visualizer = p.VisualMesh(mesh)
#     monkeypatch.setattr(
#         pv.Plotter, "write_frame", MagicMock()
#     )  # Added since pv.Plotter.open_movie() does not store frames in memory.
#     monkeypatch.setattr(pv.Plotter, "close", MagicMock())
#     visualizer.make_movie(
#         filename="test.mp4",
#         fps=fps,
#         total_time=total_time,
#         data=dummy_data,
#     )


# def test_make_gif(
#     monkeypatch: pytest.MonkeyPatch, mesh: m.Mesh, dummy_data: np.ndarray
# ) -> None:
#     """
#     Tests if the gif is created.

#     Args:
#         monkeypatch: Pytest fixture.
#         mesh: The mesh object.
#         dummy_data: The temperature data for each node, contained in a 3D array
#     """
#     visualizer = p.VisualMesh(mesh)
#     monkeypatch.setattr(pv.Plotter, "close", MagicMock())
#     visualizer.make_gif(
#         filename="test.gif",
#         fps=fps,
#         total_time=total_time,
#         data=dummy_data,
#     )
