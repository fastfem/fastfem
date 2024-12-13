import numpy as np

import fastfem.fields.numpy_similes as fnp
from fastfem.elements.element2d import Element2D
from fastfem.fields.field import Field, ShapeComponent


class LinearSimplex2D(Element2D):
    """A triangular isparametric element with a 3-dimensional shape function space -
    one for each vertex.
    """

    def basis_shape(self):
        return (3,)

    def reference_element_position_field(self):
        return Field((3,), (2,), np.array([[0, 0], [1, 0], [0, 1]]))

    def _interpolate_field(self, field, X, Y):
        X = Field(X.shape, (), X)
        Y = Field(Y.shape, (), Y)
        return (
            field.basis[0, ...] * (1.0 - X - Y)
            + field.basis[1, ...] * X
            + field.basis[2, ...] * Y
        )

    def _compute_field_gradient(self, field, pos_field=None):
        if (
            field.basis_shape == ()
            or len(field.coefficients.shape) == 0
            or field.coefficients.shape[0] == 1
        ):  # we have a constant function.
            return Field(
                (),
                (
                    *field.point_shape,
                    2,
                ),
                np.zeros(field.stack_shape + field.point_shape + (1,)),
            )

        grad_coefs = fnp.stack(
            [
                field.basis[1, ...] - field.basis[0, ...],
                field.basis[2, ...] - field.basis[0, ...],
            ],
            axis=(ShapeComponent.POINT, len(field.point_shape)),
        )
        if pos_field is not None:
            def_grad = self._compute_field_gradient(pos_field)
            grad_coefs = fnp.einsum(
                ShapeComponent.POINT,
                "...ij,...i->...j",
                fnp.linalg.inv(def_grad, ShapeComponent.POINT),
                grad_coefs,
            )

        return grad_coefs

    def _integrate_field(self, pos_field, field, jacobian_scale=...):
        coefs = (np.ones((3, 3)) + np.eye(3)) / 24
        return fnp.einsum(
            ShapeComponent.BASIS,
            ",ij,i,j->",
            fnp.abs(fnp.linalg.det(self._compute_field_gradient(pos_field))),
            coefs,
            field,
            jacobian_scale,
        )

    def _integrate_basis_times_field(
        self, pos_field, field, indices=None, jacobian_scale=...
    ):
        coefs = (
            np.array(
                [
                    [[6, 2, 2], [2, 2, 1], [2, 1, 2]],
                    [[2, 2, 1], [2, 6, 2], [1, 2, 2]],
                    [[2, 1, 2], [1, 2, 2], [2, 2, 6]],
                ]
            )
            / 120
        )
        res = fnp.einsum(
            ShapeComponent.BASIS,
            ",kij,i,j->k",
            fnp.abs(fnp.linalg.det(self._compute_field_gradient(pos_field))),
            coefs,
            field,
            jacobian_scale,
        )
        return res if indices is None else res.basis[*indices]

    def _integrate_grad_basis_dot_field(
        self, pos_field, field, indices=None, jacobian_scale=...
    ):
        # this is rather unoptimized. TODO make better
        basis_diff_coefs = np.array([[-1, -1], [1, 0], [0, 1]])
        defgrad = self._compute_field_gradient(pos_field)
        dginv = fnp.linalg.inv(defgrad)

        # pad to field-shape, excluding last axis, which is dotted (contracted)
        # fieldpad = (np.newaxis,) * (len(field.point_shape) - 1)
        basis_times_field = fnp.einsum(
            ShapeComponent.POINT,
            ",kl,lg,...g->k...",
            # exclude jacobian, since we are delegating to integrate_field subroutine.
            # np.abs(np.linalg.det(defgrad.coefficients)[..., *fieldpad]),
            1,
            basis_diff_coefs,
            dginv,
            field,
        )
        integ = fnp.moveaxis(
            self.integrate_field(
                pos_field,
                basis_times_field,
                jacobian_scale,
            ),
            (ShapeComponent.POINT, 0),
            (ShapeComponent.BASIS, 0),
        )
        if indices is None:
            return integ
        return integ.basis[*indices]
