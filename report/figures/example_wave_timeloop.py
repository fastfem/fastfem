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
