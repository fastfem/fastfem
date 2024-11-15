from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray
import typing
from typing import Literal

import itertools


def _is_broadcastable(base: tuple[int, ...], *shapes: tuple[int, ...]) -> bool:
    """Checks if shapes are broadcastable into base by numpy broadcasting rules.

    Args:
        base (tuple[int,...]): the first (target) shape
        *shapes (tuple[int,...]): the shapes to be broadcasted

    Returns:
        bool: `True` if the shapes are compatible, and false otherwise.
    """
    return all(
        map(
            lambda x: x[0] is not None
            and all(xi == x[0] or xi == 1 or xi is None for xi in x[1:]),
            itertools.zip_longest(
                reversed(base), *(reversed(shape) for shape in shapes), fillvalue=None
            ),
        )
    )


def _is_compatible(*shapes: tuple[int, ...]) -> bool:
    """Checks if shapes are compatible by numpy broadcasting rules.

    Returns:
        bool: `True` if the shapes are compatible, and false otherwise.
    """
    return all(
        map(
            lambda xi: all(  # every shape[i] in x[i] must be compatible
                map(  # recover compatibility boolean
                    lambda fjxi: fjxi[1],
                    # fj(x[i]) = (axsize , axsize compatible with x[i][j]);  j = 0,...
                    itertools.accumulate(
                        xi,
                        func=lambda a, b: (
                            b if b != 1 else a[0],
                            (a[0] == b or a[0] == 1 or b == 1),
                        ),
                        initial=(1, True),
                    ),
                )
            ),
            # x[i] = (shape[i] for shape in shapes) : i = 0,...
            itertools.zip_longest(*(reversed(shape) for shape in shapes), fillvalue=1),
        )
    )


class FieldShapeError(Exception):
    pass


class FieldConstructionError(FieldShapeError):
    """Called when constructing a field fails."""

    def __init__(self, basis_shape, field_shape, coeff_shape):
        errmsg = (
            f"Cannot construct Field object with basis_shape {basis_shape},"
            f" field_shape {field_shape} given the coefficient shape {coeff_shape}. "
        )
        super().__init__(errmsg)


class FieldBasisAccessor:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, slices):
        # do validation
        basis = np.broadcast_to(0, self.parent.basis_shape)
        new_basis_shape = basis[slices].shape
        return Field(
            new_basis_shape, self.parent.field_shape, self.parent.coefficients[slices]
        )


class FieldStackAccessor:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, slices):
        # do validation
        stack = np.broadcast_to(0, self.parent.stack_shape)
        new_stack_shape = stack[slices].shape  # noqa: F841
        slicepad = (slice(None),) * len(self.parent.basis_shape)
        if not isinstance(slices, tuple):
            slices = (slices,)
        return Field(
            self.parent.basis_shape,
            self.parent.field_shape,
            self.parent.coefficients[*slicepad, *slices],
        )


class FieldElementAccessor:
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, slices):
        # do validation
        element = np.broadcast_to(0, self.parent.field_shape)
        new_field_shape = element[slices].shape
        slicepad = (slice(None),) * (
            len(self.parent.basis_shape) + len(self.parent.stack_shape)
        )
        if not isinstance(slices, tuple):
            slices = (slices,)
        return Field(
            self.parent.basis_shape,
            new_field_shape,
            self.parent.coefficients[*slicepad, *slices],
        )


@dataclass(eq=False, frozen=True, unsafe_hash=False, init=False)
class Field:
    """
    A class responsible for storing fields on elements as an `NDArray` of coefficients.
    There are 3 relevant shapes / axis sets to a field:

    - `basis_shape` - The shape of the basis. These axes represent the multi-index for
            the basis function.

    - `stack_shape` - The shape of the element stack. These axes represent the
            multi-index for the element.

    - `field_shape` - The shape of the field. These axes represent the pointwise,
            per-element tensor index.

    The shape of `coefficients` will be `(*basis_shape,*stack_shape,*field_shape)`
    """

    basis_shape: tuple[int, ...]
    stack_shape: tuple[int, ...]
    field_shape: tuple[int, ...]
    coefficients: NDArray

    def __init__(
        self,
        basis_shape: tuple[int, ...],
        field_shape: tuple[int, ...],
        coefficients: ArrayLike,
    ):
        if not isinstance(coefficients, np.ndarray):
            coefficients = np.array(coefficients)
        cshape_orig = np.shape(coefficients)
        if len(cshape_orig) < len(basis_shape) + len(field_shape):
            coefficients = coefficients[
                *(
                    (np.newaxis,)
                    * (len(basis_shape) + len(field_shape) - len(cshape_orig))
                ),
                ...,
            ]
            cshape = np.shape(coefficients)
        else:
            cshape = cshape_orig
        bmarker = len(basis_shape)  # where basis_shape ends (excl)
        fmarker = len(cshape) - len(field_shape)  # where f_shape begins (incl)
        if (not _is_broadcastable(basis_shape, cshape[:bmarker])) or (
            not _is_broadcastable(field_shape, cshape[fmarker:])
        ):
            raise FieldConstructionError(basis_shape, field_shape, cshape_orig)
        stack_shape = cshape[bmarker:fmarker]
        object.__setattr__(
            self,
            "coefficients",
            np.broadcast_to(coefficients, basis_shape + stack_shape + field_shape),
        )
        object.__setattr__(self, "basis_shape", basis_shape)
        object.__setattr__(self, "field_shape", field_shape)
        object.__setattr__(self, "stack_shape", stack_shape)

    @typing.overload
    def __getattr__(
        self, name: Literal["shape"]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    @typing.overload
    def __getattr__(self, name: Literal["basis"]) -> FieldBasisAccessor: ...
    @typing.overload
    def __getattr__(self, name: Literal["stack"]) -> FieldStackAccessor: ...
    @typing.overload
    def __getattr__(self, name: Literal["element"]) -> FieldElementAccessor: ...

    def __getattr__(self, name):
        if name == "shape":
            return (self.basis_shape, self.stack_shape, self.field_shape)
        elif name == "basis":
            return FieldBasisAccessor(self)
        elif name == "stack":
            return FieldStackAccessor(self)
        elif name == "element":
            return FieldElementAccessor(self)
        raise AttributeError

    def broadcast_to_shape(
        self,
        basis_shape: tuple[int, ...],
        stack_shape: tuple[int, ...],
        field_shape: tuple[int, ...],
    ) -> "Field":
        if (
            not _is_broadcastable(basis_shape, self.basis_shape)
            or not _is_broadcastable(stack_shape, self.stack_shape)
            or not _is_broadcastable(field_shape, self.field_shape)
        ):
            raise FieldShapeError(
                f"Cannot broadcast field of shape {self.shape} into"
                f" shape {(basis_shape,stack_shape,field_shape)}"
            )
        return Field(
            basis_shape,
            field_shape,
            self.coefficients[
                *(np.newaxis for _ in range(len(basis_shape) - len(self.basis_shape))),
                *(slice(None) for _ in range(len(self.basis_shape))),
                *(np.newaxis for _ in range(len(stack_shape) - len(self.stack_shape))),
                *(slice(None) for _ in range(len(self.stack_shape))),
                *(np.newaxis for _ in range(len(field_shape) - len(self.field_shape))),
                *(slice(None) for _ in range(len(self.field_shape))),
            ],
        )

    @staticmethod
    def are_broadcastable(*fields: "Field") -> bool:
        return Field.are_compatible(*fields) and _is_compatible(
            *(field.field_shape for field in fields)
        )

    @staticmethod
    def broadcast_fields_full(*fields: "Field") -> tuple["Field", ...]:
        if Field.are_broadcastable(*fields):
            basis_shape = np.broadcast_shapes(*[field.basis_shape for field in fields])
            stack_shape = np.broadcast_shapes(*[field.stack_shape for field in fields])
            field_shape = np.broadcast_shapes(*[field.field_shape for field in fields])
            return tuple(
                field.broadcast_to_shape(basis_shape, stack_shape, field_shape)
                for field in fields
            )
        raise FieldShapeError("Cannot broadcast fields with incompatible shapes.")

    @staticmethod
    def are_compatible(*fields: "Field") -> bool:
        """Two fields a and b are compatible if they have compatible bases
        (basis_shape equal or at least one of them is size 1 representing a constant)
        and they have broadcastable stack_shapes.
        """
        return all(
            map(
                lambda x: x[1],  # accumulator -> did nonempty tuple change?
                itertools.accumulate(
                    (field.basis_shape for field in fields),
                    func=lambda a, b: (
                        a[0] if np.prod(b, dtype=int) == 1 else b,  # nonempty tuple
                        (np.prod(a[0], dtype=int) == 1)
                        or (np.prod(b, dtype=int) == 1)
                        or a[0] == b,  # if nonempty, did shape change?
                    ),
                    initial=(tuple(), True),
                ),
            )
        ) and _is_compatible(*(field.stack_shape for field in fields))

    @staticmethod
    def broadcast_field_compatibility(*fields: "Field") -> tuple["Field", ...]:
        if Field.are_compatible(*fields):
            basis_shape = np.broadcast_shapes(*[field.basis_shape for field in fields])
            stack_shape = np.broadcast_shapes(*[field.stack_shape for field in fields])
            return tuple(
                field.broadcast_to_shape(basis_shape, stack_shape, field.field_shape)
                for field in fields
            )
        raise FieldShapeError("Cannot broadcast fields with incompatible shapes.")

    def __eq__(self, other) -> bool:
        if not Field.are_broadcastable(self, other):
            return False

        return np.array_equiv(
            *(f.coefficients for f in Field.broadcast_fields_full(self, other))
        )
