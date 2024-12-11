import numpy as np

import fastfem.mesh as m
import fastfem.plotter as p

nh = 20
nv = 20
hl = 2
vl = 1
mesh = m.create_a_rectangle_mesh(
    horizontal_length=hl,
    vertical_length=vl,
    nodes_in_horizontal_direction=nh,
    nodes_in_vertical_direction=nv,
    element_type="triangle",
    file_name=None,
)

visualizer = p.VisualMesh(mesh)
visualizer.plot_mesh()

# Time
total_time = 3
fps = 25
time_steps = int(total_time * fps)
temperatures = np.zeros((time_steps, nv, nh))

# Dummy Data
left_temp = 0
right_temp = 0
top_temp = 0
bottom_temp = 0
min_temp = 0
max_temp = 50

temperatures = np.random.uniform(low=min_temp, high=max_temp, size=(time_steps, nv, nh))

for i in range(time_steps):
    temperatures[i, 0, :] = bottom_temp
    temperatures[i, -1, :] = top_temp
    temperatures[i, :, 0] = left_temp
    temperatures[i, :, -1] = right_temp


# Visualize
visualizer.animate_data(fps, total_time, temperatures)

filename_movie = "filename_movie.mp4"
visualizer.make_movie(filename_movie, fps, total_time, temperatures)

filename_gif = "filename_gif.gif"
visualizer.make_gif(filename_gif, fps, total_time, temperatures)
