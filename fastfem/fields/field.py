import itertools
import typing
from dataclasses import dataclass
import dataclasses
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from enum import IntEnum

import jax

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

class ShapeComponent(IntEnum):
    STACK = 0
    BASIS = 1
    FIELD = 2

@dataclass(frozen=True)
class FieldAxisIndex:
    component: ShapeComponent
    index: int
    @typing.overload
    def __getitem__(self,ind:Literal[0]) -> ShapeComponent: ...
    @typing.overload
    def __getitem__(self,ind:Literal[1]) -> int: ...

    def __getitem__(self,ind):
        if ind == 0:
            return self.component
        else:
            return self.index

FieldAxisIndexType = tuple[ShapeComponent,int] | FieldAxisIndex

def _verify_is_permutation(p: tuple[int,...]) -> None:
    if not isinstance(p,tuple):
        raise FieldShapeError("shape_order is not a permutation! (must be a tuple)")
    n = len(p) #size of the permutation
    exists = [False] * n
    for i in p:
        if not isinstance(i,int):
            raise FieldShapeError("shape_order is not a permutation! (all entries must be integers)")
        exists[i] = True

    if not all(exists):
        raise FieldShapeError("shape_order is not a permutation! (must be a bijection)")
        



class FieldConstructionError(FieldShapeError):
    """Called when constructing a field fails."""

    def __init__(self, basis_shape, field_shape, coeff_shape, shape_order, hint = None):
        errmsg = (
            f"Cannot construct Field object with basis_shape {basis_shape},"
            f" field_shape {field_shape} given the coefficient shape {coeff_shape} and shape order {shape_order}."
        )
        if hint is not None:
            errmsg += f" ({hint})"
        super().__init__(errmsg)


class FieldBasisAccessor:
    """This class is the type returned in the Field __get_attr__ for basis access.
    The sole purpose of this class is to provide the syntax for basis-targeted slicing
    in the same format as numpy.
    """

    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, slices):
        # do validation
        basis = np.broadcast_to(0, self.parent.basis_shape)
        new_basis_shape = basis[slices].shape
        slicepad = (slice(None),) * (
            (len(self.parent.stack_shape) if self.parent.shape_order[ShapeComponent.STACK] < self.parent.shape_order[ShapeComponent.BASIS] else 0)
            +(len(self.parent.field_shape) if self.parent.shape_order[ShapeComponent.FIELD] < self.parent.shape_order[ShapeComponent.BASIS] else 0)
            )
        if not isinstance(slices, tuple):
            slices = (slices,)
        return Field(
            new_basis_shape, self.parent.field_shape, self.parent.coefficients[*slicepad, *slices]
        )


class FieldStackAccessor:
    """This class is the type returned in the Field __get_attr__ for stack access.
    The sole purpose of this class is to provide the syntax for stack-targeted slicing
    in the same format as numpy.
    """

    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, slices):
        # do validation
        stack = np.broadcast_to(0, self.parent.stack_shape)
        new_stack_shape = stack[slices].shape  # noqa: F841
        slicepad = (slice(None),) * (
            (len(self.parent.basis_shape) if self.parent.shape_order[ShapeComponent.BASIS] < self.parent.shape_order[ShapeComponent.STACK] else 0)
            +(len(self.parent.field_shape) if self.parent.shape_order[ShapeComponent.FIELD] < self.parent.shape_order[ShapeComponent.STACK] else 0)
            )
        if not isinstance(slices, tuple):
            slices = (slices,)
        return Field(
            self.parent.basis_shape,
            self.parent.field_shape,
            self.parent.coefficients[*slicepad, *slices],
        )


class FieldElementAccessor:
    """This class is the type returned in the Field __get_attr__ for field-element
    access.
    The sole purpose of this class is to provide the syntax for field-targeted slicing
    in the same format as numpy.
    """

    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, slices):
        # do validation
        element = np.broadcast_to(0, self.parent.field_shape)
        new_field_shape = element[slices].shape
        slicepad = (slice(None),) * (
            (len(self.parent.basis_shape) if self.parent.shape_order[ShapeComponent.BASIS] < self.parent.shape_order[ShapeComponent.FIELD] else 0)
            +(len(self.parent.stack_shape) if self.parent.shape_order[ShapeComponent.STACK] < self.parent.shape_order[ShapeComponent.FIELD] else 0)
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

    The shape of `coefficients` will be some permutation of
    `stack_shape + field_shape + basis_shape`. The order is specified by `shape_order`,
    which is a 3-tuple `(stack_location, field_location, basis_location)`, where each
    entry is an integer specifying the position relative to the other two shapes.
    """

    basis_shape: tuple[int, ...]
    stack_shape: tuple[int, ...]
    field_shape: tuple[int, ...]
    coefficients: NDArray | jax.Array
    shape_order: tuple[int,int,int] = dataclasses.field(repr=False,init=False)
    use_jax: bool = dataclasses.field(repr=False,init=False)

    def __init__(
        self,
        basis_shape: tuple[int, ...],
        field_shape: tuple[int, ...],
        coefficients: ArrayLike,
        shape_order: tuple[int,int,int] = (0,1,2),
        use_jax: bool|None = None,
    ):  
        _verify_is_permutation(shape_order)
        if (not isinstance(coefficients, np.ndarray)
                and not isinstance(coefficients, jax.Array)):
            coefficients = np.array(coefficients)
        if use_jax is None:
            use_jax = isinstance(coefficients, jax.Array)
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
        
        #here, coefficients is at least as large as basis and field shapes combined

        # we need to place two markers to index the separations between basis, field,
        # and stack shapes; start with basis_shape (if not in middle)
        stack_start = 0
        stack_end = len(cshape)
        def cshape_slice_positives(a,b):
            return cshape[a:b]
        def cshape_slice_negatives(a,b):
            return cshape[a:(b if b != 0 else None)] if a != 0 else tuple()
        if shape_order[ShapeComponent.BASIS] == 0:
            if not _is_broadcastable(basis_shape,cshape_slice_positives(0,len(basis_shape))):
                raise FieldConstructionError(basis_shape,field_shape,cshape_orig,shape_order,hint = "basis_shape cannot be broadcasted at the beginning")
            stack_start = len(basis_shape)
        elif shape_order[ShapeComponent.BASIS] == 2:
            if not _is_broadcastable(basis_shape,cshape_slice_negatives(-len(basis_shape),0)):
                raise FieldConstructionError(basis_shape,field_shape,cshape_orig,shape_order,hint = "basis_shape cannot be broadcasted at the end")
            stack_end -= len(basis_shape)
        # then do field_shape
        if shape_order[ShapeComponent.FIELD] == 0:
            if not _is_broadcastable(field_shape,cshape_slice_positives(0,len(field_shape))):
                raise FieldConstructionError(basis_shape,field_shape,cshape_orig,shape_order,hint = "field_shape cannot be broadcasted at the beginning")
            # if basis_shape was in center, we now have the right offset for it
            if shape_order[ShapeComponent.BASIS] == 1:
                if not _is_broadcastable(basis_shape,cshape_slice_positives(len(field_shape),(len(basis_shape) + len(field_shape)))):
                    raise FieldConstructionError(basis_shape,field_shape,cshape_orig,shape_order,hint = "basis_shape cannot be broadcasted in the center")
                stack_start = len(basis_shape) + len(field_shape)
            else:
                stack_start = len(field_shape)
        elif shape_order[ShapeComponent.FIELD] == 2:
            if not _is_broadcastable(field_shape,cshape_slice_negatives(-len(field_shape),0)):
                raise FieldConstructionError(basis_shape,field_shape,cshape_orig,shape_order,hint = "field_shape cannot be broadcasted at the end")
            # if basis_shape was in center, we now have the right offset for it
            if shape_order[ShapeComponent.BASIS] == 1:
                if not _is_broadcastable(basis_shape,cshape_slice_negatives(-(len(basis_shape) + len(field_shape)),-len(field_shape))):
                    raise FieldConstructionError(basis_shape,field_shape,cshape_orig,shape_order,hint = "basis_shape cannot be broadcasted in the center")
                stack_end -= len(basis_shape)+ len(field_shape)
            else:
                stack_end -= len(field_shape)
        elif shape_order[ShapeComponent.FIELD] == 1:
            #cases by basis_location
            if shape_order[ShapeComponent.BASIS] == 0:
                if not _is_broadcastable(field_shape,cshape_slice_positives(len(basis_shape),(len(basis_shape) + len(field_shape)))):
                    raise FieldConstructionError(basis_shape,field_shape,cshape_orig,shape_order,hint = "field_shape cannot be broadcasted in the center")
                stack_start = len(basis_shape) + len(field_shape)
            else:
                if not _is_broadcastable(field_shape,cshape_slice_negatives(-(len(basis_shape) + len(field_shape)),-len(basis_shape))):
                    raise FieldConstructionError(basis_shape,field_shape,cshape_orig,shape_order,hint = "field_shape cannot be broadcasted in the center")
                stack_end -= len(basis_shape) + len(field_shape)
        
        
        stack_shape = cshape[stack_start:stack_end]
        shapes:list[tuple[int,...]] = [tuple()] * 3
        shapes[ShapeComponent.BASIS] = basis_shape
        shapes[ShapeComponent.STACK] = stack_shape
        shapes[ShapeComponent.FIELD] = field_shape
        object.__setattr__(self,"coefficients", np.broadcast_to(coefficients,shapes[0]+shapes[1]+shapes[2]))
        object.__setattr__(self, "basis_shape", basis_shape)
        object.__setattr__(self, "field_shape", field_shape)
        object.__setattr__(self, "stack_shape", stack_shape)
        object.__setattr__(self, "shape_order", shape_order)
        object.__setattr__(self, "use_jax", use_jax)

    @typing.overload
    def __getattr__(
        self, name: Literal["shape"]
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]: ...
    @typing.overload
    def __getattr__(self, name: Literal["basis"]) -> FieldBasisAccessor: ...
    @typing.overload
    def __getattr__(self, name: Literal["stack"]) -> FieldStackAccessor: ...
    @typing.overload
    def __getattr__(self, name: Literal["point"]) -> FieldElementAccessor: ...

    def __getattr__(self, name):
        if name == "shape":
            return (self.stack_shape, self.basis_shape, self.field_shape)
        elif name == "basis":
            return FieldBasisAccessor(self)
        elif name == "stack":
            return FieldStackAccessor(self)
        elif name == "point":
            return FieldElementAccessor(self)
        raise AttributeError

    def get_shape(self,component:ShapeComponent) -> tuple[int,...]:
        return self.basis_shape if component == ShapeComponent.BASIS else (self.stack_shape if component == ShapeComponent.STACK else self.field_shape)
    def broadcast_to_shape(
        self,
        stack_shape: tuple[int, ...],
        basis_shape: tuple[int, ...],
        field_shape: tuple[int, ...],
    ) -> "Field":
        if (
            not _is_broadcastable(basis_shape, self.basis_shape)
            or not _is_broadcastable(stack_shape, self.stack_shape)
            or not _is_broadcastable(field_shape, self.field_shape)
        ):
            raise FieldShapeError(
                f"Cannot broadcast field of shape {self.shape} into"
                f" shape {(stack_shape,basis_shape,field_shape)}"
            )
        slices: list[typing.Any] = [None,None,None]
        slices[ShapeComponent.BASIS] = itertools.chain((np.newaxis for _ in range(len(basis_shape) - len(self.basis_shape))),(slice(None) for _ in range(len(self.basis_shape))))
        slices[ShapeComponent.STACK] = itertools.chain((np.newaxis for _ in range(len(stack_shape) - len(self.stack_shape))),(slice(None) for _ in range(len(self.stack_shape))))
        slices[ShapeComponent.FIELD] = itertools.chain((np.newaxis for _ in range(len(field_shape) - len(self.field_shape))),(slice(None) for _ in range(len(self.field_shape))))
        return Field(
            basis_shape,
            field_shape,
            self.coefficients[
                *itertools.chain(*slices)
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
                field.broadcast_to_shape(stack_shape, basis_shape, field_shape)
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
                field.broadcast_to_shape(stack_shape, basis_shape, field.field_shape)
                for field in fields
            )
        raise FieldShapeError("Cannot broadcast fields with incompatible shapes.")

    def _axis_field_to_numpy(self, index: FieldAxisIndexType, out_of_bounds_check: bool = True):
        comp: ShapeComponent = index[0]
        ind: int = index[1]
        shape = self.get_shape(comp)

        if ind < 0:
            if out_of_bounds_check and ind < -len(shape):
                raise IndexError(f"Attempting to access axis {ind} ({-1-ind}) of shape {shape}.")
            ind = -1-ind
        elif out_of_bounds_check and ind >= len(shape):
            raise IndexError(f"Attempting to access axis {ind} of shape {shape}.")

        prec = self.shape_order[comp]
        if self.shape_order[ShapeComponent.BASIS] < prec:
            ind += len(self.basis_shape)
        if self.shape_order[ShapeComponent.STACK] < prec:
            ind += len(self.stack_shape)
        if self.shape_order[ShapeComponent.FIELD] < prec:
            ind += len(self.field_shape)
        return ind


    def __eq__(self, other) -> bool:
        if not Field.are_broadcastable(self, other):
            return False

        return np.array_equiv(
            *(f.coefficients for f in Field.broadcast_fields_full(self, other))
        )
