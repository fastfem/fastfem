from fastfem.elements.element import Element2D
from fastfem.fields.field import Field

import numpy as np


class LinearSimplex2D(Element2D):
    def basis_shape(self):
        return (3,)

    def reference_element_position_field(self):
        return Field((3,), (2,), np.array([[0, 0], [1, 0], [0, 1]]))

    def _interpolate_field(self, field, X, Y):
        field_pad = (np.newaxis,) * len(field.field_shape)
        X = X[..., *field_pad]
        Y = Y[..., *field_pad]
        return (
            field.coefficients[0, ...] * (1 - X - Y)
            + field.coefficients[1, ...] * X
            + field.coefficients[2, ...] * Y
        )

    def _compute_field_gradient(self, field, pos_field=None):

        if (
            field.basis_shape == tuple()
            or len(field.coefficients.shape) == 0
            or field.coefficients.shape[0] == 1
        ):  # we have a constant function.
            return Field(
                tuple(),
                field.field_shape + (2,),
                np.zeros(field.stack_shape + field.field_shape + (1,)),
            )

        grad_coefs = np.stack(
            [
                field.coefficients[1, ...] - field.coefficients[0, ...],
                field.coefficients[2, ...] - field.coefficients[0, ...],
            ],
            axis=-1,
        )
        if pos_field is not None:
            def_grad = self._compute_field_gradient(pos_field)
            grad_coefs = np.einsum(
                "...ij,...i->...j", np.linalg.inv(def_grad.coefficients), grad_coefs
            )

        return Field(tuple(), field.field_shape + (2,), grad_coefs)

    def _integrate_field(self, pos_field, field, jacobian_scale=...):
        coefs = (np.ones((3, 3)) + np.eye(3)) / 24
        fieldpad = (np.newaxis,) * len(field.field_shape)
        return np.einsum(
            "...,ij,i...,j...->...",
            np.abs(
                np.linalg.det(self._compute_field_gradient(pos_field).coefficients)[
                    ..., *fieldpad
                ]
            ),
            coefs,
            field.coefficients,
            jacobian_scale.coefficients[..., *fieldpad],
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
        fieldpad = (np.newaxis,) * len(field.field_shape)
        res = np.einsum(
            "...,kij,i...,j...->k...",
            np.abs(
                np.linalg.det(self._compute_field_gradient(pos_field).coefficients)[
                    ..., *fieldpad
                ]
            ),
            coefs,
            field.coefficients,
            jacobian_scale.coefficients[..., *fieldpad],
        )
        return res if indices is None else res[*indices]

    def _integrate_grad_basis_dot_field(
        self, pos_field, field, indices=None, jacobian_scale=...
    ):
        # this is rather unoptimized. TODO make better
        basis_diff_coefs = np.array([[-1, -1], [1, 0], [0, 1]])
        defgrad = self._compute_field_gradient(pos_field)
        dginv = np.linalg.inv(defgrad.coefficients)

        # pad to field-shape, excluding last axis, which is dotted (contracted)
        # fieldpad = (np.newaxis,) * (len(field.field_shape) - 1)
        basis_times_field = np.einsum(
            "...,kl,...lg,j...g->jk...",
            # exclude jacobian, since we are delegating to integrate_field subroutine.
            # np.abs(np.linalg.det(defgrad.coefficients)[..., *fieldpad]),
            1,
            basis_diff_coefs,
            dginv,
            field.coefficients,
        )
        return self.integrate_field(
            pos_field,
            Field((3,), field.field_shape[:-1], basis_times_field),
            jacobian_scale,
        )
