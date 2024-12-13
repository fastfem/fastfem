import numpy as np
import pytest

import fastfem.mesh as ffm
from fastfem.elements.linear_simplex2d import LinearSimplex2D
from fastfem.elements.linear_simplex_mesh2d import LinearSimplexMesh2D
from fastfem.fields.field import Field

meshes = {
    "rect1": ffm.create_a_rectangle_mesh(
        horizontal_length=1,
        vertical_length=1,
        nodes_in_horizontal_direction=10,
        nodes_in_vertical_direction=10,
        element_type="triangle",
        file_name=None,
    ),
    "rect2": ffm.create_a_rectangle_mesh(
        horizontal_length=1,
        vertical_length=2,
        nodes_in_horizontal_direction=None,
        nodes_in_vertical_direction=None,
        element_type="triangle",
        file_name=None,
    ),
}

LS = LinearSimplex2D()


@pytest.fixture(scope="session", params=meshes.keys())
def mesh(request):
    mesh = meshes[request.param]
    return mesh


@pytest.fixture(scope="session")
def element(mesh):
    return LinearSimplexMesh2D(mesh)


def test_atomstack(element):
    stackfield = element.to_atomstack(
        element.Field(np.arange(element.num_nodes), False)
    )
    assert stackfield.shape == ((element.num_elements,), (3,), tuple())
    # maybe want to validate that nodes are assembled correctly


def test_integrate_field(element):
    total = np.zeros((element.num_nodes,))
    for tri in element.element_node_indices:
        total[tri] += LS.integrate_field(
            Field((3,), (2,), element.node_coords[tri, :]),
            Field((3,), tuple(), np.eye(3)),
        ).coefficients
    np.testing.assert_allclose(
        element.integrate_field(element.basis_fields()).coefficients, total
    )


def test_integrate_basis_times_field(element):
    total = np.zeros((element.num_nodes, element.num_nodes))
    for tri in element.element_node_indices:
        total[tri[:, np.newaxis], tri] += LS.integrate_basis_times_field(
            Field((3,), (2,), element.node_coords[tri, :]),
            Field((3,), tuple(), np.eye(3)),
        ).coefficients
    np.testing.assert_allclose(
        element.integrate_basis_times_field(element.basis_fields()).coefficients, total
    )


def test_integrate_grad_basis_dot_field(element):
    total = np.zeros((element.num_nodes, element.num_nodes, 2))
    testfield = Field(
        (element.num_nodes,),
        (2, 2),
        element.basis_fields().coefficients[:, :, None, None] * np.eye(2),
    )
    for tri in element.element_node_indices:
        total[tri[:, np.newaxis], tri, :] += LS.integrate_grad_basis_dot_field(
            Field((3,), (2,), element.node_coords[tri, :]),
            Field((3,), (2, 2), np.eye(3)[:, :, None, None] * np.eye(2)),
        ).coefficients
    np.testing.assert_allclose(
        element.integrate_grad_basis_dot_field(testfield).coefficients,
        total,
    )


def test_integrate_grad_basis_dot_grad_field(element):
    total = np.zeros((element.num_nodes, element.num_nodes))
    for tri in element.element_node_indices:
        total[tri[:, np.newaxis], tri] += LS.integrate_grad_basis_dot_grad_field(
            Field((3,), (2,), element.node_coords[tri, :]),
            Field((3,), tuple(), np.eye(3)),
        ).coefficients
    np.testing.assert_allclose(
        element.integrate_grad_basis_dot_grad_field(
            element.basis_fields()
        ).coefficients,
        total,
    )
