import fastfem.mesh as ffm
import fastfem.elements as elems
from fastfem.fields import numpy_similes as fnp, ShapeComponent
import numpy as np


import matplotlib.pyplot as plt
import matplotlib.tri as mpltri


# ========[ create a mesh element ]========
m = ffm.create_a_rectangle_mesh(
    horizontal_length=1,
    vertical_length=1,
    nodes_in_horizontal_direction=40,
    nodes_in_vertical_direction=40,
    element_type="triangle",
    file_name=None,
)
mesh = elems.LinearSimplexMesh2D(m)

# ========[   Initial Condition    ]========


def initial_condition(x, y):
    sigma = 0.03
    return np.exp(((x - 0.3) ** 2 + (y - 0.2) ** 2) / (-2 * sigma**2))


def initial_condition_dot(x, y):
    return np.zeros(x.shape)


X = mesh.position_field.point[0].coefficients
Y = mesh.position_field.point[1].coefficients

# wave field
U = mesh.Field(initial_condition(X, Y), is_const=False)
# wave field first derivative
Udot = mesh.Field(initial_condition_dot(X, Y), is_const=False)

# ========[     Run Parameters     ]========

c = 0.2  # wave speed
t = 0  # initial time

dt = 0.01  # time step size
tmax = 3  # sim end time

# ========[    Initialize Plot     ]========

tri = mpltri.Triangulation(X, Y)
plt.ion()
plt.figure(0)
plt.show(block=False)


# ========[       Time Loop       ]========

c2 = mesh.Field(c**2, is_const=True)
inv_mass_matrix = fnp.linalg.inv(
    fnp.moveaxis(
        mesh.mass_matrix(), (ShapeComponent.STACK, 0), (ShapeComponent.BASIS, 1)
    ),
    ShapeComponent.BASIS,
)


def solve_accel(wavefield):
    return -fnp.einsum(
        ShapeComponent.BASIS,
        "ij,j->i",
        inv_mass_matrix,
        mesh.integrate_grad_basis_dot_grad_field(wavefield, jacobian_scale=c2),
    )


Uddot = solve_accel(U)
while t < tmax:
    # use Newmark-beta
    U += dt * Udot + (0.5 * dt**2) * Uddot
    Udot += 0.5 * dt * Uddot
    Uddot = solve_accel(U)
    Udot += 0.5 * dt * Uddot
    t += dt

    # plot
    plt.cla()
    plt.title(f"t = {t:.4f}")
    plt.tricontourf(tri, U.coefficients, 100, vmin=-0.25, vmax=0.25)
    plt.draw()
    plt.pause(0.001)
