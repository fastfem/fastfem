import numpy as np
import pytest

import fastfem.fields.field as field


def shapes_generator(
    dimcountlims: tuple[int, int] = (0, 4),
    dimsizelims: tuple[int, int] = (0, 2),
    totalsizelims: tuple[int, int | None] = (0, 20),
):
    if (
        (dimcountlims[1] < dimcountlims[0] or dimcountlims[1] < 0)
        or (dimsizelims[1] < dimsizelims[0] or dimsizelims[1] < 0)
        or (
            totalsizelims[1] is not None and totalsizelims[1] < max(0, totalsizelims[0])
        )
    ):
        return
    if (dimcountlims[1] == 0) and (
        totalsizelims[0] <= 0 and (totalsizelims[1] is None or totalsizelims[1] >= 0)
    ):
        yield tuple()
        return

    yield from shapes_generator(
        (dimcountlims[0], dimcountlims[1] - 1), dimsizelims, totalsizelims
    )
    for shape in shapes_generator(
        (dimcountlims[1] - 1, dimcountlims[1] - 1), dimsizelims, totalsizelims
    ):
        for k in range(dimsizelims[0], dimsizelims[1] + 1):
            newshape = shape + (k,)
            size = np.prod(newshape, dtype=int)
            if totalsizelims[0] <= size and (
                totalsizelims[1] is None or size <= totalsizelims[1]
            ):
                yield newshape


def shapes():
    yield from shapes_generator(dimcountlims=(0, 3))
    yield from ((1,) * i + (3, 2) for i in range(3, 5))
    yield from ((1,) * i + (3, 2, 2) for i in range(3, 5))


def shapes_small():
    # yield from shapes_generator(dimcountlims=(0,2),dimsizelims=(1,2))
    # yield from ((1,)*i + (3,2) for i in range(3,5))
    # yield from ((1,)*i + (3,2,2) for i in range(3,5))
    # yield from shapes_generator(dimcountlims=(1,2),dimsizelims=(0,1))
    yield tuple()
    yield from ((i,) for i in range(3))
    yield from ((1,) * i + (3, 5) for i in range(2))
    yield from ((1,) * i + (5, 3) for i in range(2))
    yield (2, 1, 2)
    yield (0, 1)


def test_field_construction():
    # attr should fail for other things
    with pytest.raises(AttributeError):
        # this should fail
        field.Field(tuple(), tuple(), 1)._shape  # pyright: ignore
    for basis_shape in shapes_small():
        for point_shape in shapes_small():
            f = field.Field(
                basis_shape, point_shape, np.empty(point_shape)
            )  # this will work
            assert f.shape == (tuple(), basis_shape, point_shape)
            for stack_shape in shapes_small():
                coefs = np.empty(stack_shape + basis_shape + point_shape)
                f = field.Field(
                    basis_shape,
                    point_shape,
                    coefs,
                )  # this will work
                if f.shape != (stack_shape, basis_shape, point_shape):
                    f = field.Field(
                        basis_shape,
                        point_shape,
                        coefs,
                    )  # this will work

                assert f.shape == (stack_shape, basis_shape, point_shape)

                # these wont
                if np.prod(basis_shape, dtype=int) > 1:
                    basis_shape_ = tuple(ax + 1 for ax in basis_shape)
                    with pytest.raises(field.FieldConstructionError):
                        field.Field(basis_shape_, point_shape, coefs)
                if np.prod(point_shape, dtype=int) > 1:
                    point_shape_ = tuple(ax + 1 for ax in point_shape)
                    with pytest.raises(field.FieldConstructionError):
                        field.Field(basis_shape, point_shape_, coefs)


def test_shape_broadcastability_and_compatibility(comparison_three_shapes):
    for a, b, c, np_compatible in comparison_three_shapes:
        broadcastable = field._is_broadcastable(a, b, c)
        compatible = field._is_compatible(a, b, c)
        if np_compatible:
            target = np.broadcast_shapes(a, b, c)
            assert compatible, (
                f"Numpy broadcasts {a}, {b}, {c} -> {target}, but"
                f" _is_compatible(...) == {compatible}, which should be True."
            )
            assert (target == a) == broadcastable, (
                f"Numpy broadcasts {a}, {b}, {c} -> {target}, but"
                f" _is_broadcastable(...) == {compatible}, which should be"
                f" {target == a}."
            )
        else:
            assert not (broadcastable or compatible), (
                f"Numpy failed to broadcast {a}, {b}, {c}, but this disagrees"
                " with field functions: _is_broadcastable(...) =="
                f" {broadcastable}, _is_compatible(...) == {compatible}."
            )


def test_field_broadcast_compatibility_on_triples(comparison_three_shapes):
    for a, b, c, _ in comparison_three_shapes:
        for a_, b_, c_, _ in comparison_three_shapes:
            nonconst = [s for s in (a, b, c) if np.prod(s, dtype=int) != 1]
            compatibility = field._is_compatible(a_, b_, c_) and (
                len(nonconst) == 0 or all(nonconst[0] == s for s in nonconst)
            )
            fa = field.Field(a, tuple(), np.random.rand(*(a_ + a)))
            fb = field.Field(b, tuple(), np.random.rand(*(b_ + b)))
            fc = field.Field(c, tuple(), np.random.rand(*(c_ + c)))
            assert field.Field.are_compatible(fa, fb, fc) == compatibility, (
                f"{fa.shape}, {fb.shape} and"
                f" {fc.shape} broadcastibility should be"
                f" {compatibility}."
            )
            if compatibility:
                fa2, fb2, fc2 = field.Field.broadcast_field_compatibility(fa, fb, fc)
                assert fa == fa2
                assert fb == fb2
                assert fc == fc2
            else:
                with pytest.raises(field.FieldShapeError):
                    field.Field.broadcast_field_compatibility(fa, fb, fc)


def test_field_reshaping_errors():
    for a in shapes_small():
        for a_ in shapes_small():
            for a__ in shapes_small():
                fa = field.Field(a, a__, np.random.rand(*(a_ + a + a__)))
                if np.prod(a, dtype=int) > 1:
                    adiff = tuple(ax + 1 for ax in a)
                    with pytest.raises(field.FieldShapeError):
                        fa.broadcast_to_shape(a_, adiff, a__)
                if np.prod(a_, dtype=int) > 1:
                    a_diff = tuple(ax + 1 for ax in a_)
                    with pytest.raises(field.FieldShapeError):
                        fa.broadcast_to_shape(a_diff, a, a__)
                if np.prod(a__, dtype=int) > 1:
                    a__diff = tuple(ax + 1 for ax in a__)
                    with pytest.raises(field.FieldShapeError):
                        fa.broadcast_to_shape(a_, a, a__diff)


def test_field_broadcast_full_on_doubles(comparison_two_shapes):
    for val1, val2, _ in comparison_two_shapes:
        for a, b, a_, b_, a__, b__ in [
            (val1, val2, val1, val1, val1, val1),
            (val1, val1, val1, val2, val1, val1),
            (val1, val1, val1, val1, val1, val2),
        ]:
            broadcastibility = (
                field._is_compatible(a_, b_)
                and field._is_compatible(a__, b__)
                and (np.prod(b, dtype=int) == 1 or np.prod(a, dtype=int) == 1 or a == b)
            )
            fa = field.Field(a, a__, np.random.rand(*(a_ + a + a__)))
            fb = field.Field(b, b__, np.random.rand(*(b_ + b + b__)))
            assert field.Field.are_broadcastable(fa, fb) == broadcastibility, (
                f"{fa.shape} and {fb.shape} broadcastibility should be"
                f" {broadcastibility}."
            )
            if broadcastibility:
                fa2, fb2 = field.Field.broadcast_fields_full(fa, fb)
                assert fa == fa2
                assert fb == fb2
                fa3 = fa.broadcast_to_shape(
                    fa2.stack_shape, fa2.basis_shape, fa2.point_shape
                )
                assert fa3 == fa2
            else:
                assert fa != fb
                with pytest.raises(field.FieldShapeError):
                    field.Field.broadcast_fields_full(fa, fb)


def test_field_accessors():
    def random_accessors(n, base_shape):
        # yields n accessors for base_shape
        def rand_symbol(dimsize):
            if np.random.rand() < 0.5 and dimsize > 0:
                return np.random.randint(dimsize)
            return slice(None)

        shapesize = len(base_shape)
        if shapesize > 0:
            for _ in range(n):
                k = np.random.randint(shapesize)
                yield tuple(rand_symbol(base_shape[i]) for i in range(k))

    def prod_chain(A, B, C):
        return (A[..., *(None for _ in B.shape)] * B)[..., *(None for _ in C.shape)] * C

    for a in shapes_small():
        for b in shapes_small():
            for c in shapes_small():
                coefs_A = np.array(np.random.rand(*a))
                coefs_B = np.array(np.random.rand(*b))
                coefs_C = np.array(np.random.rand(*c))
                f = field.Field(b, c, prod_chain(coefs_A, coefs_B, coefs_C))
                if len(f.coefficients.shape) == 0:
                    continue
                for amod in random_accessors(5, a):
                    np.testing.assert_allclose(
                        f.stack[*amod].coefficients,
                        prod_chain(coefs_A[amod], coefs_B, coefs_C),
                    )
                for bmod in random_accessors(5, b):
                    np.testing.assert_allclose(
                        f.basis[*bmod].coefficients,
                        prod_chain(coefs_A, coefs_B[bmod], coefs_C),
                    )
                for cmod in random_accessors(5, c):
                    np.testing.assert_allclose(
                        f.point[*cmod].coefficients,
                        prod_chain(coefs_A, coefs_B, coefs_C[cmod]),
                    )


def test_axis_recovery(field_of_different_shape_orders_testsep):
    def validate_shapes(shapes):
        arr = np.empty(shapes[0] + shapes[1] + shapes[2])
        for f in field_of_different_shape_orders_testsep(shapes, arr):
            assert f._component_offset(f.shape_order_inverse[0]) == 0
            assert f._component_offset(f.shape_order_inverse[1]) == len(shapes[0])
            assert f._component_offset(f.shape_order_inverse[2]) == len(
                shapes[0]
            ) + len(shapes[1])

    validate_shapes([(2, 3, 2), (2, 2), (3, 3)])
    validate_shapes([(2, 3), (4, 1), (1, 2)])


def test_reshape_axis_down(field_of_different_shape_orders_testsep):
    shapes = [(2, 3, 2), (2, 3), (2, 3)]
    reshapes = [(12, 2, 3, 2, 3), (2, 3, 2, 6, 2, 3), (2, 3, 2, 2, 3, 6)]
    arr = np.random.rand(*shapes[0], *shapes[1], *shapes[2])
    for f in field_of_different_shape_orders_testsep(shapes, arr):
        ind = f.shape_order[field.ShapeComponent.STACK]
        np.testing.assert_allclose(
            f.stack.reshape(-1).coefficients, np.reshape(arr, reshapes[ind])
        )

        ind = f.shape_order[field.ShapeComponent.BASIS]
        np.testing.assert_allclose(
            f.basis.reshape(-1).coefficients, np.reshape(arr, reshapes[ind])
        )

        ind = f.shape_order[field.ShapeComponent.POINT]
        np.testing.assert_allclose(
            f.point.reshape(-1).coefficients, np.reshape(arr, reshapes[ind])
        )
