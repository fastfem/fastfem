import numpy as np
import pyvista as pv
from unittest.mock import MagicMock

import fastfem.mesh as m
import fastfem.plotter as p

colors = ["white", "black", "red", "green", "blue", "yellow", "cyan", "magenta"]
point_label = [True, False]
cmaps = ["viridis", "plasma", "inferno", "magma", "cividis", "twilight"]

nh = np.random.randint(1, 30)
nv = np.random.randint(1, 30)
hl = np.random.randint(1, 3)
vl = np.random.randint(1, 3)

mesh_triangle = m.create_a_rectangle_mesh(
    horizontal_length=hl,
    vertical_length=vl,
    nodes_in_horizontal_direction=nh,
    nodes_in_vertical_direction=nv,
    element_type="triangle",
    file_name=None,
)

mesh_quadrangle = m.create_a_rectangle_mesh(
    horizontal_length=hl,
    vertical_length=vl,
    nodes_in_horizontal_direction=nh,
    nodes_in_vertical_direction=nv,
    element_type="quadrangle",
    file_name=None,
)

# Time
total_time = 10
fps = 25
time_steps = int(total_time * fps)
temperatures = np.zeros((time_steps, nv, nh))

# Define boundary temperatures
left_temp = 0
right_temp = 0
top_temp = 0
bottom_temp = 0
min_temp = np.random.randint(10, 20)
max_temp = np.random.randint(20, 30)

# Generate random temperature data
temperatures = np.random.uniform(low=min_temp, high=max_temp, size=(time_steps, nv, nh))

for i in range(time_steps):
    temperatures[i, 0, :] = bottom_temp
    temperatures[i, -1, :] = top_temp
    temperatures[i, :, 0] = left_temp
    temperatures[i, :, -1] = right_temp


visualizer_triangle = p.VisualMesh(mesh_triangle)
visualizer_quadrangle = p.VisualMesh(mesh_quadrangle)


def test_triangle_define_plotter():
    grid = visualizer_triangle.define_plotter()
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells > 0
    assert grid.n_points > 0


# @pytest.mark.xfail(reason="This might fail with quadrangle elements")
def test_quadrangle_define_plotter():
    grid = visualizer_quadrangle.define_plotter()
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells > 0
    assert grid.n_points > 0


def test_triangle_plot_mesh(monkeypatch):
    monkeypatch.setattr(pv.Plotter, "show", MagicMock())
    visualizer_triangle.plot_mesh(
        point_label=point_label[np.random.randint(0, 2)],
        mesh_color=colors[np.random.randint(0, len(colors))],
        edge_color=colors[np.random.randint(0, len(colors))],
        edge_thickness=np.random.randint(1, 10),
    )


def test_quadrangle_plot_mesh(monkeypatch):
    monkeypatch.setattr(pv.Plotter, "show", MagicMock())
    visualizer_quadrangle.plot_mesh(
        point_label=point_label[np.random.randint(0, 2)],
        mesh_color=colors[np.random.randint(0, len(colors))],
        edge_color=colors[np.random.randint(0, len(colors))],
        edge_thickness=np.random.randint(1, 10),
    )


def test_triangle_plot_data(monkeypatch):
    monkeypatch.setattr(pv.Plotter, "show", MagicMock())
    visualizer_triangle.plot_data(
        data=temperatures[np.random.randint(0, len(temperatures))],
        cmap=cmaps[np.random.randint(0, len(cmaps))],
    )


def test_quadrangle_plot_data(monkeypatch):
    monkeypatch.setattr(pv.Plotter, "show", MagicMock())
    visualizer_quadrangle.plot_data(
        data=temperatures[np.random.randint(0, len(temperatures))],
        cmap=cmaps[np.random.randint(0, len(cmaps))],
    )


# def test_make_movie():


# def test_make_gif():
