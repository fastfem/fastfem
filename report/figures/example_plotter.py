import numpy as np

import fastfem.mesh as m
import fastfem.plotter as p

mesh = m.create_a_rectangle_mesh(
    horizontal_length=2,
    vertical_length=1,
    nodes_in_horizontal_direction=20,
    nodes_in_vertical_direction=20,
    element_type="triangle",
    file_name=None,
)

# Data generation
TOTAL_TIME = 3
FPS = 25
TIME_STEPS = int(TOTAL_TIME * FPS)
DATA = np.random.uniform(
    0,
    50,
    size=(TIME_STEPS, 20, 20),
)

# Visualize
visualizer = p.VisualMesh(mesh)
visualizer.plot_mesh()
visualizer.animate_data(FPS, TOTAL_TIME, DATA)
visualizer.make_movie("filename_movie.mp4", FPS, TOTAL_TIME, DATA)
visualizer.make_gif("filename_gif.gif", FPS, TOTAL_TIME, DATA)