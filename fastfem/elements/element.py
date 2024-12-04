import abc
import warnings

import numpy as np
from numpy.typing import NDArray

from fastfem.fields.field import Field as FieldType


class ElementBase(abc.ABC):
    """
    Template to handle management of element operations.
    Element classes define operations to compute integrals on them.
    The elements themselves are handled in a data-driven
    way, where positions and fields are handled as arrays of coefficients of the shape
    functions defined by the element class. Derived classes must create
    the definitions for interpolation/integration, since ElementBase does not assume
    dimension, or special conditions.
    """

    def _dev_warn(self, message: str):
        """To be used by developers, only.

        This method is called by the base `Element` class to warn the use of default
        (most likely unoptimized, or potentially undesired)
        strategies to compute certain values.

        For example, `Element.mass_matrix()` computes the mass matrix by delegating to
        `integrate_basis_times_field(...,field=basis_fields())`, which may not be very
        efficient.

        This method can be overridden by the developer to suppress these warnings,
        and new ones can be called oustide of the `Element` base class without issue.

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        warnings.warn(f"Element developer message: {message}")


class IsoparametricElement(abc.ABC):
    """This type is for elements that are isoparametric (shape functions and position
    fields are of the same function space).

    From this, we can specify a consistent shape corresponding to the basis of
    this space.
    """

    @abc.abstractmethod
    def basis_shape(self) -> tuple[int, ...]:
        """
        Each element has a basis with a corresponding multi-index. The shape
        corresponding to the multi-index is given by this method. Any function defined
        on elements is passed into `element`'s methods as a tensor `f` of coefficients
        for the basis, the indices of which should be the leading indices of `f.shape`.

        Returns:
            tuple[int, ...]: a tuple representing the shape of the array corresponding
                to the basis coefficients. A scalar field `f`, given as an array is
                expected to have shape `f.shape == element.basis_shape()`.
        """
        raise NotImplementedError

    # this emulates a member class definition
    def Field(
        self, field: NDArray, is_const: bool, field_shape: tuple[int, ...] = tuple()
    ) -> FieldType:
        """Converts a `numpy` array to a `Field` object that can be used by this
        element class.

        Args:
            field (NDArray): An array representing the field.
            is_const (bool): If `True`, the field is considered
                constant, so `field.shape` is the same as its pointwise shape. For
                scalar fields, this means that `field.shape` is the field's stack shape.
                Otherwise, the first `len(element.basis_shape())` axes of `field` are
                taken to be the basis axes.
            field_shape (tuple[int, ...], optional): The shape of the field (pointwise,
            per-element). Defaults to tuple() for a scalar field.

        Raises:
            ValueError: when `is_const` is `False`, and the first
                `len(element.basis_shape())` axes of `field` are incompatible with
                `element.basis_shape()`.

        Returns:
            Field: An object corresponding to the given field.
        """
        basis_shape = self.basis_shape()
        return FieldType(
            basis_shape,
            field_shape,
            (
                field[*tuple(np.newaxis for _ in range(len(basis_shape))), ...]
                if is_const
                else field
            ),
        )

    def basis_fields(self) -> FieldType:
        """
        Returns a stack of scalar fields representing the basis elements.
        This is the identity matrix within the `basis_shape` indexing. That is,
        for two multi-indices, `I` and `J`,
        `basis_fields.coefficients[*I,*J] = (I == J)`.

        Returns:
            Field: The Field object with a stack of shape `basis_shape()` representing
                each basis function as a scalar field.
        """
        basis_shape = self.basis_shape()
        field = np.zeros(basis_shape + basis_shape)
        basis_size = np.prod(basis_shape, dtype=int)
        enumeration = np.unravel_index(np.arange(basis_size), basis_shape)
        field[*enumeration, *enumeration] = 1
        return FieldType(basis_shape, tuple(), field)

    def _verify_field_compatibilities(
        self, *fields: FieldType, **named_fields: FieldType
    ):
        compare_field = FieldType(self.basis_shape(), tuple(), 0)
        if not FieldType.are_compatible(compare_field, *fields, *named_fields.values()):
            shapestrs = [str(f.shape) for f in fields] + [
                f"{name}: {str(f.shape)}" for name, f in named_fields.items()
            ]
            raise ValueError("Incompatible shapes: " + ", ".join(shapestrs))
