from fastfem.fields.field import Field, ShapeComponent, FieldAxisIndexType, FieldAxisIndex
import jax.numpy as jnp

def moveaxis(field: Field, source: FieldAxisIndexType, destination: FieldAxisIndexType) -> Field:
    """ This attempts to replicate the numpy `moveaxis` function. Currently, multiple axes at the same time are not supported.
    https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html

    Args:
        field (Field): The field whose axes should be reordered
        source (FieldAxisIndexType | typing.Sequence[FieldAxisIndexType]): Original positions of the axes to move. These must be unique.
        destination (FieldAxisIndexType | typing.Sequence[FieldAxisIndexType]): Destination positions of the axes to move. These must also be unique.
    """


    coefs = jnp.moveaxis(field.coefficients,field._axis_field_to_numpy(source),field._axis_field_to_numpy(destination,out_of_bounds_check=False))
    shapes = {
        ShapeComponent.BASIS:field.basis_shape,
        ShapeComponent.STACK:field.stack_shape,
        ShapeComponent.FIELD:field.field_shape,
    }
    srcshape = shapes[source[0]]
    rem_axis = srcshape[source[1]] #size of the removed axis
    shapes[source[0]] = srcshape[:source[1]] + srcshape[(source[1]+1):]
    destshape = shapes[destination[0]]
    shapes[destination[0]] = destshape[:destination[1]] + (rem_axis,) + destshape[destination[1]:]
    return Field(shapes[ShapeComponent.BASIS],shapes[ShapeComponent.FIELD],coefs)


def sum(field: Field, axes: FieldAxisIndex | tuple[FieldAxisIndex,...] | ShapeComponent | None) -> Field:
    if axes is None:
        return Field(tuple(),tuple(),jnp.sum(field.coefficients))
    if isinstance(axes, ShapeComponent):
        axes = tuple(FieldAxisIndex(axes,i) for i in range(len(field.get_shape(axes))))
    elif isinstance(axes,FieldAxisIndex):
        axes = (axes,)
    
    coefs = jnp.sum(field.coefficients,tuple(map(field._axis_field_to_numpy,axes)))
    shapes = {
        ShapeComponent.BASIS:field.basis_shape,
        ShapeComponent.STACK:field.stack_shape,
        ShapeComponent.FIELD:field.field_shape,
    }
    for ax in axes:
        shape = shapes[ax[0]]
        shapes[ax[0]] = shape[:ax[1]] + shape[(ax[1]+1):]
    return Field(shapes[ShapeComponent.BASIS], shapes[ShapeComponent.FIELD], coefs)
    