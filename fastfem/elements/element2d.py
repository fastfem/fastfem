import abc
import collections.abc as colltypes

import numpy as np
from numpy.typing import ArrayLike, NDArray

import fastfem.fields.numpy_similes as fnp
from fastfem.elements.element import ElementBase, IsoparametricElement
from fastfem.fields.field import Field as FieldType
from fastfem.fields.field import ShapeComponent
from fastfem.fields.field import _is_compatible as shapes_compatible


class Element2D(ElementBase, IsoparametricElement):
    """
    Handles the management of element operations. Element classes define operations
    to compute integrals on them. The elements themselves are handled in a data-driven
    way, where positions and fields are handled as arrays of coefficients of the shape
    functions defined by the element class. As isoparametric elements, positions
    and fields use the same set of shape functions. This class also assumes derivatives
    map into the same space, so integrate_field makes sense.
    """

    @abc.abstractmethod
    def reference_element_position_field(self) -> FieldType:
        """
        The position field of the reference (un-transformed) element. This is a vector
        field.

        Returns:
            Field: A field of shape `(basis_shape(),(),(2,))` corresponding to the
            position field.
        """
        raise NotImplementedError

    def interpolate_field(
        self,
        field: FieldType,
        X: ArrayLike,
        Y: ArrayLike,
    ) -> FieldType:
        """Evaluates field at (X,Y) in reference coordinates.
        The result is an array of values `field(X,Y)`.

        X and Y must have a compatible shape with `field.stack_shape`.

        Args:
            field (Field): the field to interpolate
                representing the field to be interpolated.
            X (ArrayLike): x values (in reference coordinates).
            Y (ArrayLike): y values (in reference coordinates).

        Returns:
            NDArray: The interpolated values `field(X,Y)`
        """
        self._verify_field_compatibilities(field=field)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        if not shapes_compatible(np.shape(X), np.shape(Y)):
            message = f"shapes of X {np.shape(X)} and Y {np.shape(Y)} are incompatible."
            raise ValueError(message)
        field, _ = FieldType.broadcast_field_compatibility(
            field, FieldType(self.basis_shape(), (), np.array(0))
        )
        return self._interpolate_field(field, X, Y)

    @abc.abstractmethod
    def _interpolate_field(
        self,
        field: FieldType,
        X: NDArray,
        Y: NDArray,
    ) -> FieldType:
        """Evaluates field at (X,Y) in reference coordinates.
        The result is an array of values `field(X,Y)`.

        X and Y must have a compatible shape with `field.stack_shape`.

        Args:
            field (Field): the field to interpolate
                representing the field to be interpolated.
            X (NDArray): x values (in reference coordinates).
            Y (NDArray): y values (in reference coordinates).

        Returns:
            NDArray: The interpolated values `field(X,Y)`
        """
        raise NotImplementedError

    def compute_field_gradient(
        self,
        field: FieldType,
        pos_field: FieldType | None = None,
    ) -> FieldType:
        """
        Calculates the gradient of a field f, returning a new field.
        The result is an array of shape `(*basis_shape,...,*fieldshape,2)`,
        where field is of shape `(*basis_shape,...,*fieldshape)`.

        The last index is the coordinate of the derivative.

        This gradient can be computed in either reference space (with respect
        to the coordinates of the reference element), or in global space.
        If a global cartesian gradient should be calculated, then pos_field
        must be set to the coordinate matrix of the element. Otherwise,
        pos_field can be kept None.

        Args:
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            pos_field (Field | None, optional): If set, `pos_field` specifies
                the position fields of the element, and the gradient will be computed in
                Cartesian coordinates. This method supports element-stacking.
                Defaults to None.

        Returns:
            Field: The field of the gradient.
        """
        if pos_field is None:
            self._verify_field_compatibilities(field=field)
            field, _ = FieldType.broadcast_field_compatibility(
                field, FieldType(self.basis_shape(), (), np.array(0))
            )
        else:
            self._verify_field_compatibilities(field=field, pos_field=pos_field)
            if pos_field.point_shape != (2,):
                message = (
                    "pos_field must have point_shape (2,). Found"
                    f" {pos_field.point_shape} instead."
                )
                raise ValueError(message)
            pos_field, field, _ = FieldType.broadcast_field_compatibility(
                pos_field, field, FieldType(self.basis_shape(), (), np.array(0))
            )

        return self._compute_field_gradient(field, pos_field)

    @abc.abstractmethod
    def _compute_field_gradient(
        self,
        field: FieldType,
        pos_field: FieldType | None,
    ) -> FieldType:
        """
        Calculates the gradient of a field f, returning a new field.
        The result is an array of shape `(*basis_shape,...,*fieldshape,2)`,
        where field is of shape `(*basis_shape,...,*fieldshape)`.

        The last index is the coordinate of the derivative.

        This gradient can be computed in either reference space (with respect
        to the coordinates of the reference element), or in global space.
        If a global cartesian gradient should be calculated, then pos_field
        must be set to the coordinate matrix of the element. Otherwise,
        pos_field can be kept None.

        Args:
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            pos_field (Field | None, optional): If set, `pos_field` specifies
                the position fields of the element, and the gradient will be computed in
                Cartesian coordinates. This method supports element-stacking.
                Defaults to None.

        Returns:
            Field: The field of the gradient.
        """
        raise NotImplementedError

    def interpolate_field_gradient(
        self,
        field: FieldType,
        X: ArrayLike,
        Y: ArrayLike,
        pos_field: FieldType | None = None,
    ) -> FieldType:
        """
        Calculates the gradient of a field f at the reference coordinates (X,Y).
        The result is an array of shape `(...,*fieldshape,2)`,
        where field is of shape `(*basis_shape,...,*fieldshape)`.

        X and Y must have compatible shape.

        The last index is the coordinate of the derivative.

        This gradient can be computed in either reference space (with respect
        to the coordinates of the reference element), or in global space.
        If a global cartesian gradient should be calculated, then pos_field
        must be set to the coordinate matrix of the element. Otherwise,
        pos_field can be kept None.

        Args:
            field (Field): an array of shape (*basis_shape,*fieldshape)
                representing the field to be interpolated.
            X (ArrayLike): x values (in reference coordinates).
            Y (ArrayLike): y values (in reference coordinates).
            pos_field (Field | None, optional): If set, `pos_field` specifies
                the position fields of the element, and the gradient will be computed in
                Cartesian coordinates. Defaults to None.

        Returns:
            NDArray: an array representing the gradient of the field evaluated
                at each point.
        """

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        if pos_field is None:
            self._verify_field_compatibilities(field=field)
            if not shapes_compatible(np.shape(X), np.shape(Y)):
                message = (
                    f" shapes of X {np.shape(X)} and Y {np.shape(Y)} incompatible."
                )
                raise ValueError(message)
            field, _ = FieldType.broadcast_field_compatibility(
                field, FieldType(self.basis_shape(), (), np.array(0))
            )
        else:
            self._verify_field_compatibilities(field=field, pos_field=pos_field)
            if not shapes_compatible(np.shape(X), np.shape(Y)):
                message = f"shapes of X {np.shape(X)} and Y {np.shape(Y)} incompatible."
                raise ValueError(message)
            if pos_field.point_shape != (2,):
                message = (
                    "pos_field must have point_shape (2,). Found"
                    f" {pos_field.point_shape} instead."
                )
                raise ValueError(message)
            pos_field, field, _ = FieldType.broadcast_field_compatibility(
                pos_field, field, FieldType(self.basis_shape(), (), np.array(0))
            )

        return self._interpolate_field_gradient(field, X, Y, pos_field)

    def _interpolate_field_gradient(
        self,
        field: FieldType,
        X: NDArray,
        Y: NDArray,
        pos_field: FieldType | None,
    ) -> FieldType:
        """
        Calculates the gradient of a field f at the reference coordinates (X,Y).
        The result is an array of shape `(...,*fieldshape,2)`,
        where field is of shape `(*basis_shape,...,*fieldshape)`.

        X and Y must have compatible shape.

        The last index is the coordinate of the derivative.

        This gradient can be computed in either reference space (with respect
        to the coordinates of the reference element), or in global space.
        If a global cartesian gradient should be calculated, then pos_field
        must be set to the coordinate matrix of the element. Otherwise,
        pos_field can be kept None.

        Args:
            field (Field): an array of shape (*basis_shape,*fieldshape)
                representing the field to be interpolated.
            X (NDArray): x values (in reference coordinates).
            Y (NDArray): y values (in reference coordinates).
            pos_field (Field | None, optional): If set, `pos_field` specifies
                the position fields of the element, and the gradient will be computed in
                Cartesian coordinates. Defaults to None.

        Returns:
            NDArray: an array representing the gradient of the field evaluated
                at each point.
        """
        message = (
            "Element._interpolate_field_gradient() called, which delegates to"
            " compute_field_gradient() and interpolate_field()"
        )
        self._dev_warn(message)
        return self.interpolate_field(
            self.compute_field_gradient(field, pos_field),
            X,
            Y,
        )

    def integrate_field(
        self,
        pos_field: FieldType,
        field: FieldType,
        jacobian_scale: FieldType = FieldType((), (), 1),  # NOQA: B008
    ) -> FieldType:
        """
        Integrates `field` $f$ over the element. The result is the value of
        $\\int \\alpha f ~dV$ over the element, as an array of shape
        `(...,*point_shape)`.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape `(...,*fieldshape)`.
        """
        self._verify_field_compatibilities(
            pos_field=pos_field, field=field, jacobian_scale=jacobian_scale
        )
        if pos_field.point_shape != (2,):
            message = (
                "pos_field must have point_shape (2,). Found"
                f" {pos_field.point_shape} instead."
            )
            raise ValueError(message)
        pos_field, field, jacobian_scale, _ = FieldType.broadcast_field_compatibility(
            pos_field,
            field,
            jacobian_scale,
            FieldType(self.basis_shape(), (), np.array(0)),
        )
        return self._integrate_field(pos_field, field, jacobian_scale)

    @abc.abstractmethod
    def _integrate_field(
        self,
        pos_field: FieldType,
        field: FieldType,
        jacobian_scale: FieldType,
    ) -> FieldType:
        """
        Integrates `field` $f$ over the element. The result is the value of
        $\\int \\alpha f ~dV$ over the element, as an array of shape
        `(...,*point_shape)`.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape `(...,*fieldshape)`.
        """
        raise NotImplementedError

    def integrate_basis_times_field(
        self,
        pos_field: FieldType,
        field: FieldType,
        indices: colltypes.Sequence | None = None,
        jacobian_scale: FieldType = FieldType((), (), 1),  # NOQA: B008
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\phi_i f ~ dV$
        for fields $f$ and $\\alpha$, on this element, given by `field` and
        `jacobian_scale`, respectively. When None, `jacobian_scale` is interpreted
        as the identity.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape `(*basis_shape,...,*fieldshape)`
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the
                    basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        self._verify_field_compatibilities(
            pos_field=pos_field, field=field, jacobian_scale=jacobian_scale
        )
        if pos_field.point_shape != (2,):
            message = (
                "pos_field must have point_shape (2,). Found"
                f" {pos_field.point_shape} instead."
            )
            raise ValueError(message)
        pos_field, field, jacobian_scale, _ = FieldType.broadcast_field_compatibility(
            pos_field,
            field,
            jacobian_scale,
            FieldType(self.basis_shape(), (), np.array(0)),
        )
        return self._integrate_basis_times_field(
            pos_field, field, indices, jacobian_scale
        )

    @abc.abstractmethod
    def _integrate_basis_times_field(
        self,
        pos_field: FieldType,
        field: FieldType,
        indices: colltypes.Sequence | None,
        jacobian_scale: FieldType,
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\phi_i f ~ dV$
        for fields $f$ and $\\alpha$, on this element, given by `field` and
        `jacobian_scale`, respectively. When None, `jacobian_scale` is interpreted
        as the identity.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape `(*basis_shape,...,*fieldshape)`
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the
                    basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        raise NotImplementedError

    def mass_matrix(
        self,
        pos_field: FieldType,
        indices: colltypes.Sequence | None = None,
        jacobian_scale: FieldType = FieldType((), (), 1),  # NOQA: B008
    ) -> FieldType:
        """
        Recovers the mass matrix entries $\\int \\alpha \\phi_i \\phi_j ~ dV$
        for this element. `pos_field` has shape `(basis_shape,...,2)`, where the
        ellipses represent element stacking.

        The full mass matrix is of shape
        `(*basis_shape, *basis_shape,...)`, where the last indices match the element
        stack, but slices of the mass matrix can be
        obtained by passing into the `indices` argument. The implementation of
        `basis_mass_matrix` should assume that `indices` argument is smaller than the
        whole matrix, so if `M[*indices]` is larger than `M`, one should use
        `basis_mass_matrix(pos_field)[*indices,...]` instead of
        `basis_mass_matrix(pos_field,indices)`.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the mass
                    matrix to access, or None if the whole matrix should be returned.
                    Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: an array representing the mass matrix, or portions thereof.
        """
        self._verify_field_compatibilities(
            pos_field=pos_field, jacobian_scale=jacobian_scale
        )
        if pos_field.point_shape != (2,):
            message = (
                "pos_field must have point_shape (2,). Found"
                f" {pos_field.point_shape} instead."
            )
            raise ValueError(message)
        pos_field, jacobian_scale, _ = FieldType.broadcast_field_compatibility(
            pos_field,
            jacobian_scale,
            FieldType(self.basis_shape(), (), np.array(0)),
        )
        return self._mass_matrix(pos_field, indices, jacobian_scale)

    def _mass_matrix(
        self,
        pos_field: FieldType,
        indices: colltypes.Sequence | None,
        jacobian_scale: FieldType,
    ) -> FieldType:
        """
        Recovers the mass matrix entries $\\int \\alpha \\phi_i \\phi_j ~ dV$
        for this element. `pos_field` has shape `(basis_shape,...,2)`, where the
        ellipses represent element stacking.

        The full mass matrix is of shape
        `(*basis_shape, *basis_shape,...)`, where the last indices match the element
        stack, but slices of the mass matrix can be
        obtained by passing into the `indices` argument. The implementation of
        `basis_mass_matrix` should assume that `indices` argument is smaller than the
        whole matrix, so if `M[*indices]` is larger than `M`, one should use
        `basis_mass_matrix(pos_field)[*indices,...]` instead of
        `basis_mass_matrix(pos_field,indices)`.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the mass
                    matrix to access, or None if the whole matrix should be returned.
                    Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: an array representing the mass matrix, or portions thereof.
        """
        self._dev_warn(
            "Element._mass_matrix() called, which delegates "
            "to integrate_basis_times_field()"
        )
        mat = fnp.moveaxis(
            self.integrate_basis_times_field(
                pos_field,
                fnp.moveaxis(
                    self.basis_fields(),
                    (ShapeComponent.STACK, 0),
                    (ShapeComponent.POINT, 0),
                ),
                None if indices is None else indices[: len(self.basis_shape())],
                jacobian_scale,
            ),
            (ShapeComponent.POINT, 0),
            (ShapeComponent.BASIS, 0),
        )
        return (
            mat if indices is None else mat.basis[*indices[len(self.basis_shape()) :]]
        )

    def integrate_grad_basis_dot_field(
        self,
        pos_field: FieldType,
        field: FieldType,
        indices: colltypes.Sequence | None = None,
        jacobian_scale: FieldType = FieldType((), (), 1),  # NOQA: B008
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot f ~ dV$
        for a field $f$, on this element. The dot product takes the last axis of
        `field`, which must have size equal to the domain dimension.
        The basis for the vector field is assumed to be in global coordinates,
        which is assumed to be over the kronecker dot product.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        self._verify_field_compatibilities(
            pos_field=pos_field, field=field, jacobian_scale=jacobian_scale
        )
        if pos_field.point_shape != (2,):
            message = (
                "pos_field must have point_shape (2,). Found"
                f" {pos_field.point_shape} instead."
            )
            raise ValueError(message)
        pos_field, field, jacobian_scale, _ = FieldType.broadcast_field_compatibility(
            pos_field,
            field,
            jacobian_scale,
            FieldType(self.basis_shape(), (), np.array(0)),
        )
        return self._integrate_grad_basis_dot_field(
            pos_field, field, indices, jacobian_scale
        )

    @abc.abstractmethod
    def _integrate_grad_basis_dot_field(
        self,
        pos_field: FieldType,
        field: FieldType,
        indices: colltypes.Sequence | None,
        jacobian_scale: FieldType,
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot f ~ dV$
        for a field $f$, on this element. The dot product takes the last axis of
        `field`, which must have size equal to the domain dimension.
        The basis for the vector field is assumed to be in global coordinates,
        which is assumed to be over the kronecker dot product.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        raise NotImplementedError

    def integrate_grad_basis_dot_grad_field(
        self,
        pos_field: FieldType,
        field: FieldType,
        indices: colltypes.Sequence | None = None,
        jacobian_scale: FieldType = FieldType((), (), 1),  # NOQA: B008
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot \\nabla f ~ dV$
        for a field $f$, on this element. The dot product is over the gradients.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        self._verify_field_compatibilities(
            pos_field=pos_field, field=field, jacobian_scale=jacobian_scale
        )
        if pos_field.point_shape != (2,):
            message = (
                "pos_field must have point_shape (2,). Found"
                f" {pos_field.point_shape} instead."
            )
            raise ValueError(message)
        pos_field, field, jacobian_scale, _ = FieldType.broadcast_field_compatibility(
            pos_field,
            field,
            jacobian_scale,
            FieldType(self.basis_shape(), (), np.array(0)),
        )
        return self._integrate_grad_basis_dot_grad_field(
            pos_field, field, indices, jacobian_scale
        )

    def _integrate_grad_basis_dot_grad_field(
        self,
        pos_field: FieldType,
        field: FieldType,
        indices: colltypes.Sequence | None,
        jacobian_scale: FieldType,
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot \\nabla f ~ dV$
        for a field $f$, on this element. The dot product is over the gradients.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        self._dev_warn(
            "Element._integrate_grad_basis_dot_grad_field() called, "
            "which delegates to integrate_grad_basis_dot_field()"
        )
        return self.integrate_grad_basis_dot_field(
            pos_field,
            self.compute_field_gradient(field, pos_field),
            indices,
            jacobian_scale,
        )

    # TODO boundary integrals,


class StaticElement2D(ElementBase, IsoparametricElement):
    """
    Handles the management of element operations. Element classes define operations
    to compute integrals on them. The elements themselves are handled in a data-driven
    way, where positions and fields are handled as arrays of coefficients of the shape
    functions defined by the element class. As isoparametric elements, positions
    and fields use the same set of shape functions, but it is not assumed that
    derivatives do as well.

    Static elements have a defined position field, which must be set at initialization.
    """

    position_field: FieldType

    def interpolate_field(
        self,
        field: FieldType,
        X: ArrayLike,
        Y: ArrayLike,
    ) -> FieldType:
        """Evaluates field at (X,Y) in reference coordinates.
        The result is an array of values `field(X,Y)`.

        X and Y must have a compatible shape with `field.stack_shape`.

        Args:
            field (Field): the field to interpolate
                representing the field to be interpolated.
            X (ArrayLike): x values (in reference coordinates).
            Y (ArrayLike): y values (in reference coordinates).

        Returns:
            FieldType: The interpolated values `field(X,Y)`
        """
        self._verify_field_compatibilities(field=field)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(Y, np.ndarray):
            Y = np.array(Y)
        if not shapes_compatible(field.stack_shape, np.shape(X), np.shape(Y)):
            message = (
                f"field stack_shape {field.stack_shape} must be compatible with the"
                f" shapes of X {np.shape(X)} and Y {np.shape(Y)}."
            )
            raise ValueError(message)
        field, _ = FieldType.broadcast_field_compatibility(
            field, FieldType(self.basis_shape(), (), np.array(0))
        )
        return self._interpolate_field(field, X, Y)

    @abc.abstractmethod
    def _interpolate_field(
        self,
        field: FieldType,
        X: NDArray,
        Y: NDArray,
    ) -> FieldType:
        """Evaluates field at (X,Y) in reference coordinates.
        The result is an array of values `field(X,Y)`.

        X and Y must have a compatible shape with `field.stack_shape`.

        Args:
            field (Field): the field to interpolate
                representing the field to be interpolated.
            X (NDArray): x values (in reference coordinates).
            Y (NDArray): y values (in reference coordinates).

        Returns:
            FieldType: The interpolated values `field(X,Y)`
        """
        raise NotImplementedError

    def integrate_field(
        self,
        field: FieldType,
        jacobian_scale: FieldType = FieldType((), (), 1),  # NOQA: B008
    ) -> FieldType:
        """
        Integrates `field` $f$ over the element. The result is the value of
        $\\int \\alpha f ~dV$ over the element, as an array of shape
        `(...,*point_shape)`.

        Args:
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape `(...,*fieldshape)`.
        """
        self._verify_field_compatibilities(field=field, jacobian_scale=jacobian_scale)
        field, jacobian_scale, _ = FieldType.broadcast_field_compatibility(
            field,
            jacobian_scale,
            FieldType(self.basis_shape(), (), np.array(0)),
        )
        return self._integrate_field(field, jacobian_scale)

    @abc.abstractmethod
    def _integrate_field(
        self,
        field: FieldType,
        jacobian_scale: FieldType,
    ) -> FieldType:
        """
        Integrates `field` $f$ over the element. The result is the value of
        $\\int \\alpha f ~dV$ over the element, as an array of shape
        `(...,*point_shape)`.

        Args:
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape `(...,*fieldshape)`.
        """
        raise NotImplementedError

    def integrate_basis_times_field(
        self,
        field: FieldType,
        indices: colltypes.Sequence | None = None,
        jacobian_scale: FieldType = FieldType((), (), 1),  # NOQA: B008
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\phi_i f ~ dV$
        for fields $f$ and $\\alpha$, on this element, given by `field` and
        `jacobian_scale`, respectively. When None, `jacobian_scale` is interpreted
        as the identity.

        Args:
            field (Field): an array of shape `(*basis_shape,...,*fieldshape)`
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the
                    basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        self._verify_field_compatibilities(field=field, jacobian_scale=jacobian_scale)
        field, jacobian_scale, _ = FieldType.broadcast_field_compatibility(
            field,
            jacobian_scale,
            FieldType(self.basis_shape(), (), np.array(0)),
        )
        return self._integrate_basis_times_field(field, indices, jacobian_scale)

    @abc.abstractmethod
    def _integrate_basis_times_field(
        self,
        field: FieldType,
        indices: colltypes.Sequence | None,
        jacobian_scale: FieldType,
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\phi_i f ~ dV$
        for fields $f$ and $\\alpha$, on this element, given by `field` and
        `jacobian_scale`, respectively. When None, `jacobian_scale` is interpreted
        as the identity.

        Args:
            field (Field): an array of shape `(*basis_shape,...,*fieldshape)`
                representing the field to be interpolated.
            fieldshape (tuple[int,...], optional): the shape of `field` pointwise.
                Defaults to tuple(), representing a scalar field.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the
                    basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        raise NotImplementedError

    def mass_matrix(
        self,
        indices: colltypes.Sequence | None = None,
        jacobian_scale: FieldType = FieldType((), (), 1),  # NOQA: B008
    ) -> FieldType:
        """
        Recovers the mass matrix entries $\\int \\alpha \\phi_i \\phi_j ~ dV$
        for this element. `pos_field` has shape `(basis_shape,...,2)`, where the
        ellipses represent element stacking.

        The full mass matrix is of shape
        `(*basis_shape, *basis_shape,...)`, where the last indices match the element
        stack, but slices of the mass matrix can be
        obtained by passing into the `indices` argument. The implementation of
        `basis_mass_matrix` should assume that `indices` argument is smaller than the
        whole matrix, so if `M[*indices]` is larger than `M`, one should use
        `basis_mass_matrix(pos_field)[*indices,...]` instead of
        `basis_mass_matrix(pos_field,indices)`.

        Args:
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the mass
                    matrix to access, or None if the whole matrix should be returned.
                    Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: an array representing the mass matrix, or portions thereof.
        """
        self._verify_field_compatibilities(jacobian_scale=jacobian_scale)
        jacobian_scale, _ = FieldType.broadcast_field_compatibility(
            jacobian_scale,
            FieldType(self.basis_shape(), (), np.array(0)),
        )
        return self._mass_matrix(indices, jacobian_scale)

    def _mass_matrix(
        self,
        indices: colltypes.Sequence | None,
        jacobian_scale: FieldType,
    ) -> FieldType:
        """
        Recovers the mass matrix entries $\\int \\alpha \\phi_i \\phi_j ~ dV$
        for this element. `pos_field` has shape `(basis_shape,...,2)`, where the
        ellipses represent element stacking.

        The full mass matrix is of shape
        `(*basis_shape, *basis_shape,...)`, where the last indices match the element
        stack, but slices of the mass matrix can be
        obtained by passing into the `indices` argument. The implementation of
        `basis_mass_matrix` should assume that `indices` argument is smaller than the
        whole matrix, so if `M[*indices]` is larger than `M`, one should use
        `basis_mass_matrix(pos_field)[*indices,...]` instead of
        `basis_mass_matrix(pos_field,indices)`.

        Args:
            pos_field (Field): an array representing the positions of the
                    element nodes. This is of the shape `(basis_shape,...,2)`
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the mass
                    matrix to access, or None if the whole matrix should be returned.
                    Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: an array representing the mass matrix, or portions thereof.
        """

        self._dev_warn(
            "Element._mass_matrix() called, which delegates "
            "to integrate_basis_times_field()"
        )
        mat = self.integrate_basis_times_field(
            self.basis_fields(),
            None if indices is None else indices[: len(self.basis_shape())],
            jacobian_scale,
        )
        return (
            mat if indices is None else mat.basis[*indices[len(self.basis_shape()) :]]
        )

    def integrate_grad_basis_dot_field(
        self,
        field: FieldType,
        indices: colltypes.Sequence | None = None,
        jacobian_scale: FieldType = FieldType((), (), 1),  # NOQA: B008
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot f ~ dV$
        for a field $f$, on this element. The dot product takes the last axis of
        `field`, which must have size equal to the domain dimension.
        The basis for the vector field is assumed to be in global coordinates,
        which is assumed to be over the kronecker dot product.

        Args:
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        self._verify_field_compatibilities(field=field, jacobian_scale=jacobian_scale)
        field, jacobian_scale, _ = FieldType.broadcast_field_compatibility(
            field,
            jacobian_scale,
            FieldType(self.basis_shape(), (), np.array(0)),
        )
        return self._integrate_grad_basis_dot_field(field, indices, jacobian_scale)

    @abc.abstractmethod
    def _integrate_grad_basis_dot_field(
        self,
        field: FieldType,
        indices: colltypes.Sequence | None,
        jacobian_scale: FieldType,
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot f ~ dV$
        for a field $f$, on this element. The dot product takes the last axis of
        `field`, which must have size equal to the domain dimension.
        The basis for the vector field is assumed to be in global coordinates,
        which is assumed to be over the kronecker dot product.

        Args:
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        raise NotImplementedError

    def integrate_grad_basis_dot_grad_field(
        self,
        field: FieldType,
        indices: colltypes.Sequence | None = None,
        jacobian_scale: FieldType = FieldType((), (), 1),  # NOQA: B008
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot \\nabla f ~ dV$
        for a field $f$, on this element. The dot product is over the gradients.

        Args:
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        self._verify_field_compatibilities(field=field, jacobian_scale=jacobian_scale)
        field, jacobian_scale, _ = FieldType.broadcast_field_compatibility(
            field,
            jacobian_scale,
            FieldType(self.basis_shape(), (), np.array(0)),
        )
        return self._integrate_grad_basis_dot_grad_field(field, indices, jacobian_scale)

    @abc.abstractmethod
    def _integrate_grad_basis_dot_grad_field(
        self,
        field: FieldType,
        indices: colltypes.Sequence | None,
        jacobian_scale: FieldType,
    ) -> FieldType:
        """
        Computes the integral $\\int \\alpha \\nabla \\phi_i \\cdot \\nabla f ~ dV$
        for a field $f$, on this element. The dot product is over the gradients.

        Args:
            field (Field): an array of shape (*basis_shape,...,*fieldshape)
                representing the field to be interpolated.
            indices (colltypes.Sequence[ArrayLike] | None, optional): Indices of the basis
                    functions to integrate against, or None if the integral should be
                    computed against every basis field. Defaults to None.
            jacobian_scale (Field, optional): an array of shape
                `(*basis_shape,...)` representing the scalar factor $\\alpha$.
                Defaults to 1.

        Returns:
            NDArray: The resultant integral, an array of shape
                `(indices.shape,...,*fieldshape)`, or `(*basis_shape,...,*fieldshape)`
                if `indices` is None.
        """
        raise NotImplementedError

    # TODO boundary integrals,
