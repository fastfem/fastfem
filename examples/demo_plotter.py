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


# Domains:
bottom_boundary = mesh["bottom_boundary"]
right_boundary = mesh["right_boundary"]
top_boundary = mesh["top_boundary"]
left_boundary = mesh["left_boundary"]
rectangle = mesh["surface"]


visualizer = p.VisualMesh(mesh)
visualizer.plot_mesh()

# Time
total_time = 3
fps = 25
time_steps = int(total_time * fps)
temperatures = np.zeros((time_steps, nv, nh))

# Dummy data
left_right_temp = 15
top_temp = 5
bottom_temp = 25
for t in range(time_steps):
    for i in range(nv):
        for j in range(nh):
            x = j / (nh - 1) * hl
            y = i / (nv - 1) * vl
            if j == 0 or j == nh - 1:
                temperatures[t, i, j] = left_right_temp
            elif i == 0:
                temperatures[t, i, j] = bottom_temp
            elif i == nv - 1:
                temperatures[t, i, j] = top_temp
            else:
                temperatures[t, i, j] = 20 + 20 * np.sin(2 * np.pi * x / hl) * np.cos(
                    2 * np.pi * y / vl
                ) * np.sin(np.pi * t / time_steps)


visualizer.animate_mesh(fps, total_time, temperatures)

filename_movie = "filename_movie.mp4"
visualizer.make_movie(filename_movie, fps, total_time, temperatures)

filename_gif = "filename_gif.gif"
visualizer.make_gif(filename_gif, fps, total_time, temperatures)
