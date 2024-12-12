from numpy.typing import NDArray
import typing
import numpy as _np

from fastfem.fields.field import Field, ShapeComponent

ShapeEntryType = tuple[int, ...]


def assemble_field_add(
    disassembled_field: Field,
    node_indices: NDArray[_np.intp],
    assembled_basis_shape: int | ShapeEntryType,
) -> Field:
    """This function is used for mesh assemblies, where fields on subelements must
    be accumulated into the larger mesh element. For integration, the same node may
    need to take the value of a sum of integrals on subelements. This requires
    an atomic add operation.

    Given an assembly with a 1-axis basis shape B1, with a stack B2 of subelements
    with shape B3, the node index `i` in B1 corresponds to the index K of subelement J
    for which `node_indices[*J,*K] == i`.

    Args:
        disassembled_field (Field): The field with basis shape B3 corresponding to the
            stack of disassembled fields. The leading axes in the stack must match B2.
        node_indices (NDArray[np.intp]): The array of indices demonstrating the
            assembly.
        assembled_basis_shape (int | tuple[int, ...]): The shape of the assembled field.
            This should be 1D.

    Returns:
        Field: The assembled field.
    """
    # verify assembly shapes are compatible
    subelem_basis_shape = disassembled_field.basis_shape
    if node_indices.shape[-len(subelem_basis_shape) :] != subelem_basis_shape:
        raise ValueError(
            "The last axes of node_indices must match the basis shape!"
            f" ({node_indices.shape[-len(subelem_basis_shape):]} != basis shape"
            f" {subelem_basis_shape}; node_indices.shape = {node_indices.shape})"
        )
    assembly_stack_shape = node_indices.shape[: -len(subelem_basis_shape)]
    if (
        assembly_stack_shape
        != disassembled_field.stack_shape[: len(assembly_stack_shape)]
    ):
        raise ValueError(
            "The first axes of node_indices must match the first axes of the stack"
            f" shape! ({node_indices.shape[:-len(subelem_basis_shape)]} !="
            f" {disassembled_field.stack_shape[:len(assembly_stack_shape)]};"
            f" node_indices.shape = {node_indices.shape}; stack_shape ="
            f" {disassembled_field.stack_shape})"
        )
    if isinstance(assembled_basis_shape, int):
        assembled_basis_shape = (assembled_basis_shape,)
    if len(assembled_basis_shape) != 1:
        raise ValueError(
            "assembled_basis_shape must have 1-dimension! (assembled_basis_shape ="
            f" {assembled_basis_shape})"
        )

    if disassembled_field.use_jax:
        raise NotImplementedError(
            "JAX-friendly accumulation methods not yet implemented!"
        )

    outshape = [tuple()] * 3
    outshape[ShapeComponent.BASIS] = assembled_basis_shape
    outshape[ShapeComponent.STACK] = disassembled_field.stack_shape[
        len(assembly_stack_shape) :
    ]
    outshape[ShapeComponent.POINT] = disassembled_field.point_shape
    accum = _np.zeros(
        outshape[disassembled_field.shape_order_inverse[0]]
        + outshape[disassembled_field.shape_order_inverse[1]]
        + outshape[disassembled_field.shape_order_inverse[2]],
        dtype=disassembled_field.coefficients.dtype,
    )
    src_basis_ind = disassembled_field._axis_field_to_numpy((ShapeComponent.BASIS, 0))
    src_stack_ind = disassembled_field._axis_field_to_numpy((ShapeComponent.STACK, 0))
    end_basis_ind = src_basis_ind - (
        len(assembly_stack_shape)
        if disassembled_field.shape_order[ShapeComponent.STACK]
        < disassembled_field.shape_order[ShapeComponent.BASIS]
        else 0
    )

    src_shift_origin = (
        *range(src_stack_ind, src_stack_ind + len(assembly_stack_shape)),
        *range(src_basis_ind, src_basis_ind + len(subelem_basis_shape)),
    )
    src_shift_target = (
        *range(len(assembly_stack_shape)),
        *range(
            len(assembly_stack_shape),
            len(assembly_stack_shape) + len(subelem_basis_shape),
        ),
    )
    _np.add.at(
        _np.moveaxis(accum, end_basis_ind, 0),
        node_indices,
        _np.moveaxis(
            typing.cast(NDArray, disassembled_field.coefficients),
            src_shift_origin,
            src_shift_target,
        ),
    )

    return Field(
        outshape[ShapeComponent.BASIS],
        outshape[ShapeComponent.POINT],
        accum,
        shape_order=disassembled_field.shape_order,
        use_jax=False,
    )
