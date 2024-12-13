import pathlib
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import pyvista as pv

import fastfem.mesh as m
import fastfem.plotter as p

# Relevant constants for mesh building/videos
NODES_IN_HORIZONTAL_DIRECTION = 10
NODES_IN_VERTICAL_DIRECTION = 10
TOTAL_TIME = 1
FPS = 10
TIME_STEPS = int(TOTAL_TIME * FPS)


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
        nodes_in_horizontal_direction=NODES_IN_HORIZONTAL_DIRECTION,
        nodes_in_vertical_direction=NODES_IN_VERTICAL_DIRECTION,
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
    data = np.random.uniform(
        low=min_temp,
        high=max_temp,
        size=(TIME_STEPS, NODES_IN_VERTICAL_DIRECTION, NODES_IN_HORIZONTAL_DIRECTION),
    )
    for i in range(TIME_STEPS):
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
def test_plot_mesh(
    monkeypatch: pytest.MonkeyPatch,
    mesh: m.Mesh,
    point_label: bool,
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
    )


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
        fps=FPS,
        total_time=TOTAL_TIME,
        data=dummy_data,
    )
    with pytest.raises(ValueError):
        visualizer.animate_data(
            fps=np.random.uniform(25.1, 100),
            total_time=TOTAL_TIME,
            data=dummy_data,
        )


@pytest.mark.skip(reason="Not currently supported")
def test_make_movie(
    tmp_path: pathlib.Path,
    mesh: m.Mesh,
    dummy_data: np.ndarray,
) -> None:
    """
    Tests if the movie is created.

    Args:
        monkeypatch: Pytest fixture.
        mesh: The mesh object.
        dummy_data: The temperature data for each node, contained in a 3D array.
    """
    visualizer = p.VisualMesh(mesh)
    filename = tmp_path / "test.mp4"
    visualizer.make_movie(
        filename=str(filename),
        fps=FPS,
        total_time=TOTAL_TIME,
        data=dummy_data,
    )
    assert filename.exists()


@pytest.mark.skipif(
    sys.platform in ["win32", "linux"],
    reason="Currently not supported on Windows/Linux",
)
def test_make_gif(
    tmp_path: pathlib.Path,
    mesh: m.Mesh,
    dummy_data: np.ndarray,
) -> None:
    """
    Tests if the gif is created.

    Args:
        monkeypatch: Pytest fixture.
        mesh: The mesh object.
        dummy_data: The temperature data for each node, contained in a 3D array
    """
    visualizer = p.VisualMesh(mesh)
    filename = tmp_path / "test.gif"
    visualizer.make_gif(
        filename=str(filename),
        fps=FPS,
        total_time=TOTAL_TIME,
        data=dummy_data,
    )
    assert filename.exists()
