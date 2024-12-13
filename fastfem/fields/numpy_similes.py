import itertools
from collections.abc import Sequence
from typing import Literal, cast

import numpy  # NOQA: ICN001
from numpy.typing import ArrayLike

from fastfem.fields.field import (
    Field,
    FieldAxisIndex,
    FieldAxisIndexType,
    ShapeComponent,
    _reshape,
)
from fastfem.fields.field import np_or_jnp as np

from . import numpy_similes_linalg as linalg


def moveaxis(
    field: Field, source: FieldAxisIndexType, destination: FieldAxisIndexType
) -> Field:
    """This attempts to replicate the numpy `moveaxis` function. Currently, multiple axes at the same time are not supported.
    https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html

    Args:
        field (Field): The field whose axes should be reordered
        source (FieldAxisIndexType | typing.Sequence[FieldAxisIndexType]): Original positions of the axes to move. These must be unique.
        destination (FieldAxisIndexType | typing.Sequence[FieldAxisIndexType]): Destination positions of the axes to move. These must also be unique.
    """

    shapes = {
        ShapeComponent.BASIS: field.basis_shape,
        ShapeComponent.STACK: field.stack_shape,
        ShapeComponent.POINT: field.point_shape,
    }

    src_pos_np = field._axis_field_to_numpy(source)
    srcshape = shapes[source[0]]

    src_pt = source[1] if source[1] >= 0 else len(srcshape) + source[1]
    rem_axis = srcshape[source[1]]  # size of the removed axis
    # when counting from the right, be sure to add one for the new length
    dest_pt = (
        destination[1]
        if destination[1] >= 0
        else len(shapes[destination[0]])
        + (0 if source[0] == destination[0] else 1)
        + destination[1]
    )
    dest_pos_np = field._axis_field_to_numpy((destination[0], dest_pt), False)
    coefs = np(field.use_jax).moveaxis(
        field.coefficients,
        src_pos_np,
        # dest_pos_np - (1 if dest_pos_np > src_pos_np else 0), #account for removals shifting indices
        dest_pos_np
        - (
            1 if field.shape_order[source[0]] < field.shape_order[destination[0]] else 0
        ),
    )
    shapes[source[0]] = srcshape[:src_pt] + srcshape[(src_pt + 1) :]
    destshape = shapes[destination[0]]
    shapes[destination[0]] = destshape[:dest_pt] + (rem_axis,) + destshape[dest_pt:]
    return Field(
        shapes[ShapeComponent.BASIS],
        shapes[ShapeComponent.POINT],
        coefs,
        shape_order=field.shape_order,
    )


def sum(
    field: Field,
    axes: FieldAxisIndexType | tuple[FieldAxisIndexType, ...] | ShapeComponent | None,
) -> Field:
    """This attempts to replicate the numpy "sum" function. In
    https://numpy.org/doc/stable/reference/generated/numpy.sum.html

    Args:
        field (Field): _description_
        axes (FieldAxisIndex | tuple[FieldAxisIndex,...] | ShapeComponent | None): _description_

    Returns:
        Field: _description_
    """
    if axes is None:
        return Field((), (), np(field.use_jax).sum(field.coefficients))
    axes_: tuple[FieldAxisIndexType, ...] = ()
    if isinstance(axes, ShapeComponent):
        axes_ = tuple(
            FieldAxisIndex(axes, i) for i in range(len(field.get_shape(axes)))
        )
    elif isinstance(axes, FieldAxisIndex):
        axes_ = (axes,)
    elif isinstance(axes[0], tuple | FieldAxisIndex):
        # ^ force tuple[tuple]              ^ force tuple[FieldAxisIndex]

        # for some reason pyright doesn't approve
        axes_ = cast(tuple[FieldAxisIndexType, ...], axes)
    else:
        axes_ = cast(tuple[FieldAxisIndexType, ...], (axes,))

    coefs = np(field.use_jax).sum(
        field.coefficients, tuple(map(field._axis_field_to_numpy, axes_))
    )
    shapes = {
        ShapeComponent.BASIS: field.basis_shape,
        ShapeComponent.STACK: field.stack_shape,
        ShapeComponent.POINT: field.point_shape,
    }
    for ax in axes_:
        shape = shapes[ax[0]]
        shapes[ax[0]] = shape[: ax[1]] + shape[(ax[1] + 1) :]
    return Field(shapes[ShapeComponent.BASIS], shapes[ShapeComponent.POINT], coefs)


def reshape(
    field: Field,
    component_selector: ShapeComponent,
    shape: int | tuple[int],
    order: Literal["C", "F", "A"] = "C",
    copy: bool | None = None,
) -> Field:
    """This attempts to replicate the numpy "reshape" function.
    https://numpy.org/doc/stable/reference/generated/numpy.reshape.html

    reshape() applies numpy "reshape" to a given component. This function is also called
    when using the field accessor reshape methods. That is, `reshape(field,BASIS,s)` is
    the same as `field.basis.reshape(s)`
    Args:
        field (Field): The field to reshape
        component_selector (ShapeComponent): Which component to reshape.
        shape (int | tuple[int]): The target shape of the component. For any integer i,
            i is equivalent to (i,)
        order ({'C','F','A'}, optional): See the numpy documentation. Defaults to 'C'.
        copy (bool | None, optional): See the numpy documentation. Defaults to None.

    Raises:
        ValueError: when a copy operation is required, but `copy` is False.
    Returns:
        Field: The reshaped field. This is always a new object, but if data is copied,
        the underlying array is a view of the original.
    """
    return _reshape(field, component_selector, shape, order, copy)


def einsum(
    component_selector: ShapeComponent,
    subscripts: str,
    *operands: ArrayLike | Field,
) -> Field:
    fieldops = tuple(op for op in operands if isinstance(op, Field))
    if len(fieldops) == 0:
        message = "Please provide at least one Field!"
        raise ValueError(message)

    valid_indices = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    desired_order = fieldops[0].shape_order

    # which components to reshape to single lines; we need indices for them:
    comp1rs = ShapeComponent(1 if component_selector == 0 else 0)
    comp2rs = ShapeComponent(
        (comp1rs + 2) if component_selector == (comp1rs + 1) else comp1rs + 1
    )
    # order so that the first one appears first
    if desired_order[comp1rs] > desired_order[comp2rs]:
        comp1rs, comp2rs = comp2rs, comp1rs

    # broadcast all components except the one being selected
    broadcast_shape = [()] * 3
    for comp in comp1rs, comp2rs:
        broadcast_shape[comp] = numpy.broadcast_shapes(  # type: ignore
            *(field.get_shape(comp) for field in fieldops)
        )
    broadcasted = []
    for field in fieldops:
        broadcast_shape[component_selector] = field.get_shape(component_selector)  # type: ignore
        broadcasted.append(
            field.broadcast_to_shape(
                broadcast_shape[ShapeComponent.STACK],
                broadcast_shape[ShapeComponent.BASIS],
                broadcast_shape[ShapeComponent.POINT],
            )
        )

    if desired_order[comp1rs] < desired_order[component_selector]:
        comp1label = valid_indices[0]
        if desired_order[comp2rs] < desired_order[component_selector]:
            comp2label = valid_indices[1]
        else:
            comp2label = valid_indices[-1]
    else:
        comp1label = valid_indices[-2]
        comp2label = valid_indices[-1]

    if comp1label in subscripts or comp2label in subscripts:
        message = (
            f"Subscripts {comp1label} and {comp2label} have not been reserved for"
            " non-selected indices!"
        )
        raise ValueError(message)

    # the shapes of the result
    output_shapes: list[tuple[int, ...]] = [()] * 3
    output_shapes[desired_order[comp1rs]] = broadcasted[0].get_shape(comp1rs)
    output_shapes[desired_order[comp2rs]] = broadcasted[0].get_shape(comp2rs)
    use_jax = any(f.use_jax for f in broadcasted)
    # inject broadcasted operands back
    broadcasted_iter = iter(broadcasted)
    operands = tuple(
        (
            reshape(
                reshape(
                    next(broadcasted_iter),
                    comp1rs,
                    -1,
                ),
                comp2rs,
                -1,
            )
            if isinstance(op, Field)
            else op
        )
        for op in operands
    )

    def component_inject(order) -> tuple[str, str]:
        compstr = (
            comp1label + comp2label
            if order[comp1rs] < order[comp2rs]
            else comp2label + comp1label
        )
        ind = order[component_selector]
        return compstr[:ind], compstr[ind:]

    ops_iter = iter(
        itertools.chain(
            (
                (component_inject(op.shape_order) if isinstance(op, Field) else None)
                for op in operands
            ),
            [component_inject(desired_order)],
        )
    )
    einstr = "->".join(
        ",".join(
            (subs if ins is None else ins[0] + subs + ins[1])
            for subs, ins in zip(sec.split(","), ops_iter, strict=False)
        )
        for sec in subscripts.split("->")
    )
    coefs = np(*broadcasted).einsum(
        einstr,
        *(op.coefficients if isinstance(op, Field) else op for op in operands),
    )
    if desired_order[component_selector] == 0:
        output_shapes[desired_order[component_selector]] = coefs.shape[:-2]
    elif desired_order[component_selector] == 1:
        output_shapes[desired_order[component_selector]] = coefs.shape[1:-1]
    else:
        output_shapes[desired_order[component_selector]] = coefs.shape[2:]
    return Field(
        output_shapes[desired_order[ShapeComponent.BASIS]],
        output_shapes[desired_order[ShapeComponent.POINT]],
        coefs.reshape(
            output_shapes[0] + output_shapes[1] + output_shapes[2],
        ),
        shape_order=broadcasted[0].shape_order,
        use_jax=use_jax,
    )


def abs(field: Field) -> Field:
    """absolute value of field. TODO link to numpy

    Args:
        field (Field): _description_

    Returns:
        Field: _description_
    """
    return Field(
        field.basis_shape,
        field.point_shape,
        np(field).abs(field.coefficients),
        shape_order=field.shape_order,
        use_jax=field.use_jax,
    )


def stack(fields: Sequence[Field], axis: FieldAxisIndexType) -> Field:
    """stack multiple fields. TODO link to numpy

    Args:
        fields (Sequence[Field]): _description_
        axis (FieldAxisIndexType): _description_

    Returns:
        Field: _description_
    """
    _np = np(*fields)
    shapes = [()] * 3
    shapes[ShapeComponent.BASIS] = fields[0].basis_shape  # type: ignore
    shapes[ShapeComponent.STACK] = fields[0].stack_shape  # type: ignore
    shapes[ShapeComponent.POINT] = fields[0].point_shape  # type: ignore
    shapes[axis[0]] = (  # type: ignore
        shapes[axis[0]][: axis[1]] + (len(fields),) + shapes[axis[0]][axis[1] :]
    )
    return Field(
        shapes[ShapeComponent.BASIS],
        shapes[ShapeComponent.POINT],
        _np.stack(
            [field.coefficients for field in fields],
            axis=fields[0]._axis_field_to_numpy(axis, False),
        ),
        shape_order=fields[0].shape_order,
        use_jax=fields[0].use_jax,
    )


__all__ = [
    "abs",
    "einsum",
    "linalg",
    "moveaxis",
    "reshape",
    "sum",
]
