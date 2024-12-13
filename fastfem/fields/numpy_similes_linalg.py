from fastfem.fields.field import Field, FieldShapeError, ShapeComponent
from fastfem.fields.field import np_or_jnp as np


def det(
    field: Field, component_selector: ShapeComponent = ShapeComponent.POINT
) -> Field:
    """This is analogous to [`numpy.linalg.det`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html)

    Computes the determinant of a field along the given component. By default, the determinant is point-wise.
    Raises a FieldShapeError if that component is not a square matrix.

    Args:
        field (Field): The field to compute the determinant of
        component_selector (ShapeComponent, optional): The component to take the determinant over. Defaults to ShapeComponent.POINT.
    Raises:
        FieldShapeError: if the selected component is not square.

    Returns:
        Field: A field representing the determinant, with the selected component contracted.
    """
    selshape = field.get_shape(component_selector)
    if len(selshape) != 2 and selshape[0] != selshape[1]:
        message = (
            "Cannot take the determinant along component"
            f" {component_selector} ({selshape}). Must be square!"
        )
        raise FieldShapeError(message)
    shapes = [()] * 3
    shapes[ShapeComponent.BASIS] = field.basis_shape  # type: ignore
    shapes[ShapeComponent.POINT] = field.point_shape  # type: ignore
    shapes[ShapeComponent.STACK] = field.stack_shape  # type: ignore
    shapes[component_selector] = ()
    _np = np(field)
    return Field(
        shapes[ShapeComponent.BASIS],
        shapes[ShapeComponent.POINT],
        _np.linalg.det(
            _np.moveaxis(
                field.coefficients,
                (
                    field._axis_field_to_numpy((component_selector, 0)),
                    field._axis_field_to_numpy((component_selector, 1)),
                ),
                (-2, -1),
            )
        ),
        shape_order=field.shape_order,
        use_jax=field.use_jax,
    )


def inv(
    field: Field, component_selector: ShapeComponent = ShapeComponent.POINT
) -> Field:
    """This is analogous to [`numpy.linalg.inv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html)

    Computes the matrix inverse of a field along the given component. By default, the inverse is point-wise.
    Raises a FieldShapeError if that component is not a square matrix.

    Args:
        field (Field): The field to compute the matrix inverse of
        component_selector (ShapeComponent, optional): The component to take the inverse over. Defaults to ShapeComponent.POINT.
    Raises:
        FieldShapeError: if the selected component is not square.

    Returns:
        Field: A field representing the inverse, with the same shape.
    """
    selshape = field.get_shape(component_selector)
    if len(selshape) != 2 and selshape[0] != selshape[1]:
        message = (
            "Cannot take the determinant along component"
            f" {component_selector} ({selshape}). Must be square!"
        )
        raise FieldShapeError(message)
    _np = np(field)
    return Field(
        field.basis_shape,
        field.point_shape,
        _np.moveaxis(
            _np.linalg.inv(
                _np.moveaxis(
                    field.coefficients,
                    (
                        field._axis_field_to_numpy((component_selector, 0)),
                        field._axis_field_to_numpy((component_selector, 1)),
                    ),
                    (-2, -1),
                )
            ),
            (-2, -1),
            (
                field._axis_field_to_numpy((component_selector, 0)),
                field._axis_field_to_numpy((component_selector, 1)),
            ),
        ),
        shape_order=field.shape_order,
        use_jax=field.use_jax,
    )


__all__ = ["det", "inv"]
