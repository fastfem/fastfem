import numpy as np

import fastfem.mesh as m
import fastfem.plotter as p

NODES_IN_HORIZONTAL_DRECTION = 20
NODES_IN_VERTICAL_DIRECTION = 20
HORIZONTAL_LENGTH = 2
VERTICAL_LENGTH = 1
mesh = m.create_a_rectangle_mesh(
    horizontal_length=HORIZONTAL_LENGTH,
    vertical_length=VERTICAL_LENGTH,
    nodes_in_horizontal_direction=NODES_IN_HORIZONTAL_DRECTION,
    nodes_in_vertical_direction=NODES_IN_VERTICAL_DIRECTION,
    element_type="triangle",
    file_name=None,
)

visualizer = p.VisualMesh(mesh)
visualizer.plot_mesh()

# Time
TOTAL_TIME = 3
FPS = 25
TIME_STEPS = int(TOTAL_TIME * FPS)
TEMPERATURES = np.zeros(
    (TIME_STEPS, NODES_IN_VERTICAL_DIRECTION, NODES_IN_HORIZONTAL_DRECTION)
)

# Dummy Data
LEFT_TEMP = 0
RIGHT_TEMP = 0
TOP_TEMP = 0
BOTTOM_TEMP = 0
MIN_TEMP = 0
MAX_TEMP = 50

TEMPERATURES = np.random.uniform(
    low=MIN_TEMP,
    high=MAX_TEMP,
    size=(TIME_STEPS, NODES_IN_VERTICAL_DIRECTION, NODES_IN_HORIZONTAL_DRECTION),
)

for i in range(TIME_STEPS):
    TEMPERATURES[i, 0, :] = BOTTOM_TEMP
    TEMPERATURES[i, -1, :] = TOP_TEMP
    TEMPERATURES[i, :, 0] = LEFT_TEMP
    TEMPERATURES[i, :, -1] = RIGHT_TEMP


# Visualize
visualizer.animate_data(FPS, TOTAL_TIME, TEMPERATURES)

filename_movie = "filename_movie.mp4"
visualizer.make_movie(filename_movie, FPS, TOTAL_TIME, TEMPERATURES)

filename_gif = "filename_gif.gif"
visualizer.make_gif(filename_gif, FPS, TOTAL_TIME, TEMPERATURES)
