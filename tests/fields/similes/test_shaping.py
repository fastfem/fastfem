import numpy as np

import fastfem.fields.numpy_similes as fnp
from fastfem.fields.field import Field, ShapeComponent


def test_moveaxis(permutation):
    def sum_shapes(shapes):
        return shapes[0] + shapes[1] + shapes[2]

    shapes = [(0, 1, 2), (3, 4, 5), (6, 7, 8)]
    perm_shapes = [()] * 3
    for i in range(len(perm_shapes)):
        perm_shapes[permutation[i]] = shapes[i]
    arr = np.broadcast_to(0, sum_shapes(perm_shapes))
    f = Field(
        shapes[ShapeComponent.BASIS],
        shapes[ShapeComponent.POINT],
        arr,
        shape_order=permutation,
    )

    def assert_cascade_down(p1, p2, s1, s2):
        shapes_swp = np.array(sum_shapes(shapes), dtype=int)
        rem = shapes_swp[s1]
        shapes_swp[s1:s2] = shapes_swp[s1 + 1 : s2 + 1]
        shapes_swp[s2] = rem
        moved = fnp.moveaxis(f, p1, p2).shape
        assert sum_shapes(moved) == tuple(int(k) for k in shapes_swp)
        shape_sizes = list(len(k) for k in shapes)
        shape_sizes[p1[0]] -= 1
        shape_sizes[p2[0]] += 1
        assert tuple(shape_sizes) == tuple(len(k) for k in moved)

    def assert_cascade_up(p1, p2, s1, s2):
        shapes_swp = np.array(sum_shapes(shapes), dtype=int)
        rem = shapes_swp[s2]
        shapes_swp[s1 + 1 : s2 + 1] = shapes_swp[s1:s2]
        shapes_swp[s1] = rem
        moved = fnp.moveaxis(f, p1, p2).shape
        assert sum_shapes(moved) == tuple(int(k) for k in shapes_swp)
        shape_sizes = list(len(k) for k in shapes)
        shape_sizes[p1[0]] -= 1
        shape_sizes[p2[0]] += 1
        assert tuple(shape_sizes) == tuple(len(k) for k in moved)

    assert_cascade_down((ShapeComponent.STACK, 0), (ShapeComponent.STACK, 1), 0, 1)
    assert_cascade_down((ShapeComponent.STACK, 0), (ShapeComponent.STACK, 2), 0, 2)
    assert_cascade_down((ShapeComponent.STACK, 0), (ShapeComponent.STACK, -1), 0, 2)
    assert_cascade_down((ShapeComponent.STACK, 0), (ShapeComponent.BASIS, 0), 0, 2)
    assert_cascade_down((ShapeComponent.STACK, 0), (ShapeComponent.BASIS, 1), 0, 3)
    assert_cascade_down((ShapeComponent.STACK, 0), (ShapeComponent.BASIS, 2), 0, 4)
    assert_cascade_down((ShapeComponent.STACK, 0), (ShapeComponent.BASIS, 3), 0, 5)
    assert_cascade_down((ShapeComponent.STACK, 0), (ShapeComponent.BASIS, -1), 0, 5)
    assert_cascade_down((ShapeComponent.STACK, 0), (ShapeComponent.POINT, 0), 0, 5)
    assert_cascade_down((ShapeComponent.STACK, 0), (ShapeComponent.POINT, 1), 0, 6)
    assert_cascade_down((ShapeComponent.STACK, 1), (ShapeComponent.STACK, 0), 0, 1)
    assert_cascade_up((ShapeComponent.POINT, -1), (ShapeComponent.STACK, 0), 0, 8)
