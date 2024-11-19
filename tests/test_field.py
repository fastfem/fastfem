import fastfem.fields.field as field
import numpy as np

import pytest


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
        for field_shape in shapes_small():
            f = field.Field(
                basis_shape, field_shape, np.empty(field_shape)
            )  # this will work
            assert f.shape == (basis_shape, tuple(), field_shape)
            for stack_shape in shapes_small():
                coefs = np.empty(basis_shape + stack_shape + field_shape)
                f = field.Field(
                    basis_shape,
                    field_shape,
                    coefs,
                )  # this will work
                assert f.shape == (basis_shape, stack_shape, field_shape)

                # these wont
                if np.prod(basis_shape, dtype=int) > 1:
                    basis_shape_ = tuple(ax + 1 for ax in basis_shape)
                    with pytest.raises(field.FieldConstructionError):
                        field.Field(basis_shape_, field_shape, coefs)
                if np.prod(field_shape, dtype=int) > 1:
                    field_shape_ = tuple(ax + 1 for ax in field_shape)
                    with pytest.raises(field.FieldConstructionError):
                        field.Field(basis_shape, field_shape_, coefs)


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
            fa = field.Field(a, tuple(), np.random.rand(*(a + a_)))
            fb = field.Field(b, tuple(), np.random.rand(*(b + b_)))
            fc = field.Field(c, tuple(), np.random.rand(*(c + c_)))
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
                fa = field.Field(a, a__, np.random.rand(*(a + a_ + a__)))
                if np.prod(a, dtype=int) > 1:
                    adiff = tuple(ax + 1 for ax in a)
                    with pytest.raises(field.FieldShapeError):
                        fa.broadcast_to_shape(adiff, a_, a__)
                if np.prod(a_, dtype=int) > 1:
                    a_diff = tuple(ax + 1 for ax in a_)
                    with pytest.raises(field.FieldShapeError):
                        fa.broadcast_to_shape(a, a_diff, a__)
                if np.prod(a__, dtype=int) > 1:
                    a__diff = tuple(ax + 1 for ax in a__)
                    with pytest.raises(field.FieldShapeError):
                        fa.broadcast_to_shape(a, a_, a__diff)


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
            fa = field.Field(a, a__, np.random.rand(*(a + a_ + a__)))
            fb = field.Field(b, b__, np.random.rand(*(b + b_ + b__)))
            assert field.Field.are_broadcastable(fa, fb) == broadcastibility, (
                f"{fa.shape} and {fb.shape} broadcastibility should be"
                f" {broadcastibility}."
            )
            if broadcastibility:
                fa2, fb2 = field.Field.broadcast_fields_full(fa, fb)
                assert fa == fa2
                assert fb == fb2
                fa3 = fa.broadcast_to_shape(
                    fa2.basis_shape, fa2.stack_shape, fa2.field_shape
                )
                assert fa3 == fa2
            else:
                assert fa != fb
                with pytest.raises(field.FieldShapeError):
                    field.Field.broadcast_fields_full(fa, fb)


def test_field_accessors():

    def random_accessors(n, base_shape):
        def rand_symbol(dimsize):
            if np.random.rand() < 0.5 and dimsize > 0:
                return np.random.randint(dimsize)
            return slice(None)

        shapesize = len(base_shape)
        if shapesize > 0:
            for _ in range(n):
                k = np.random.randint(shapesize)
                yield tuple(rand_symbol(base_shape[i]) for i in range(k))

    for a in shapes_small():
        for b in shapes_small():
            for c in shapes_small():
                f = field.Field(a, c, np.random.rand(*(a + b + c)))
                if len(f.coefficients.shape) == 0:
                    continue
                for amod in random_accessors(5, a):
                    np.testing.assert_allclose(
                        f.basis[*amod].coefficients, f.coefficients[*amod]
                    )
                slicepad = tuple(slice(None) for _ in a)
                for bmod in random_accessors(5, b):
                    np.testing.assert_allclose(
                        f.stack[*bmod].coefficients, f.coefficients[*slicepad, *bmod]
                    )
                slicepad = tuple(slice(None) for _ in a + b)
                for cmod in random_accessors(5, c):
                    np.testing.assert_allclose(
                        f.element[*cmod].coefficients, f.coefficients[*slicepad, *cmod]
                    )
