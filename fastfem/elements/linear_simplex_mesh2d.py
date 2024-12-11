import collections.abc as colltypes
import typing

import numpy as np
from numpy.typing import NDArray

import fastfem.fields.numpy_similes as fnp
from fastfem.elements.element2d import StaticElement2D
from fastfem.elements.linear_simplex2d import LinearSimplex2D
from fastfem.fields.field import Field, FieldAxisIndex, ShapeComponent
from fastfem.mesh import Mesh, create_a_rectangle_mesh

atom = LinearSimplex2D()


class LinearSimplexMesh2D(StaticElement2D):
    def __init__(self, mesh: Mesh):
        nodes = dict()
        elems = list()
        # gather all nodes and elements in the above structures
        for component in mesh:
            if component.dimension == 2:
                for submesh in component.mesh:
                    if submesh.number_of_nodes_per_element != 3:
                        raise ValueError(
                            "LinearSimplexMesh2D operates on triangles!"
                            f" '{component.name}' has"
                            f" {submesh.number_of_nodes_per_element} nodes per element."
                        )
                    nodes.update(submesh.nodes)
                    elems.extend(submesh.elements.values())
        self.mesh = mesh
        self.num_nodes = len(nodes)
        self.num_elements = len(elems)
        node_index_to_basis_index = dict()
        self.node_coords = np.empty((self.num_nodes, 2), dtype=float)
        self.element_node_indices = np.empty((self.num_elements, 3), dtype=int)
        # populate node coords, and obtain the node-index -> basis-index mapping
        for i, v in enumerate(nodes.items()):
            if abs(v[1][2]) > 1e-15:
                raise ValueError(
                    f"Node {i} must have a z-component of zero. Found"
                    f" {v[1][2]} instead."
                )
            node_index_to_basis_index[v[0]] = i
            self.node_coords[i, :] = v[1][:2]

        # element [i,:] should be the basis indices of element i.
        for i in range(self.num_elements):
            for j in range(3):
                self.element_node_indices[i, j] = node_index_to_basis_index[elems[i][j]]

        self.position_field = Field((self.num_nodes,), (2,), self.node_coords)
        self.position_field_atomstack = self.to_atomstack(self.position_field)

    def to_atomstack(self, field: Field) -> Field:
        """Takes a field from the Mesh basis into the atom (LinearSimplex2D) field,
        where the field on each element `i` of the mesh corresponds to
        `atomstack.stack[i,...]`. That is, the basis shape becomes (3,), while an
        axis of size `num_elements` is prepended to the stack shape.

        Args:
            field (Field): The field to convert.

        Returns:
            Field: The field as represented by a stack of LinearSimplex2D fields.
        """
        self._verify_field_compatibilities(field)
        field = field.broadcast_to_shape(
            field.stack_shape, self.basis_shape(), field.point_shape
        )
        return fnp.moveaxis(
            field.basis[self.element_node_indices],
            (ShapeComponent.BASIS, 0),
            (ShapeComponent.STACK, 0),
        )

    @typing.override
    def basis_shape(self) -> tuple[int, ...]:
        return (self.num_nodes,)

    @typing.override
    def _interpolate_field(self, field: Field, X: NDArray, Y: NDArray) -> Field:
        raise NotImplementedError

    @typing.override
    def _integrate_field(self, field: Field, jacobian_scale: Field) -> Field:
        return fnp.sum(
            atom._integrate_field(
                self.position_field_atomstack,
                self.to_atomstack(field),
                self.to_atomstack(jacobian_scale),
            ),
            FieldAxisIndex(ShapeComponent.STACK, 0),
        )

    @typing.override
    def _integrate_basis_times_field(
        self, field: Field, indices: colltypes.Sequence | None, jacobian_scale: Field
    ) -> Field:
        return fnp.sum(
            atom._integrate_basis_times_field(
                self.position_field_atomstack,
                self.to_atomstack(field),
                indices,
                self.to_atomstack(jacobian_scale),
            ),
            FieldAxisIndex(ShapeComponent.STACK, 0),
        )

    @typing.override
    def _integrate_grad_basis_dot_field(
        self, field: Field, indices: colltypes.Sequence | None, jacobian_scale: Field
    ) -> Field:
        return fnp.sum(
            atom._integrate_grad_basis_dot_field(
                self.position_field_atomstack,
                self.to_atomstack(field),
                indices,
                self.to_atomstack(jacobian_scale),
            ),
            FieldAxisIndex(ShapeComponent.STACK, 0),
        )

    @typing.override
    def _integrate_grad_basis_dot_grad_field(
        self, field: Field, indices: colltypes.Sequence | None, jacobian_scale: Field
    ) -> Field:
        return fnp.sum(
            atom._integrate_grad_basis_dot_grad_field(
                self.position_field_atomstack,
                self.to_atomstack(field),
                indices,
                self.to_atomstack(jacobian_scale),
            ),
            FieldAxisIndex(ShapeComponent.STACK, 0),
        )


if __name__ == "__main__":
    elem = LinearSimplexMesh2D(
        create_a_rectangle_mesh(
            horizontal_length=1,
            vertical_length=1,
            nodes_in_horizontal_direction=10,
            nodes_in_vertical_direction=10,
            element_type="triangle",
            file_name=None,
        )
    )
    print(elem.num_nodes)
    print(elem.num_elements)
    print(elem.basis_shape())
    print(elem.integrate_field(Field(tuple(), tuple(), 1)))
