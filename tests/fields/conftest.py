import numpy as np
import pytest

from fastfem.fields.field import Field, ShapeComponent

_SHAPES_COMPARE_: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
    (tuple(), tuple()),
    (tuple(), (0,)),
    (tuple(), (1,)),
    (tuple(), (2,)),
    (tuple(), (3, 2)),
    ((1,), (0,)),
    ((1,), (1,)),
    ((1,), (2,)),
    ((1,), (3, 2)),
    ((2,), (3,)),
    ((1, 2), (2, 1)),
    ((1, 1, 1, 5), (2, 2, 2, 5)),
    ((1, 1, 1, 5), (2, 2, 2, 4)),
    ((2, 2, 1, 1), (3,)),
]
_SHAPES_COMPARE_TRIPLES_: list[
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
] = [
    *((a, b, tuple()) for a, b in _SHAPES_COMPARE_),
    ((1,), (2,), (3,)),
]


def are_shapes_compatible(*shapes: tuple[int, ...]):
    try:
        np.broadcast_shapes(*shapes)
    except ValueError:
        return False
    return True


_SHAPES_COMPARE: list[tuple[tuple[int, ...], tuple[int, ...], bool]] = [
    (a, b, are_shapes_compatible(a, b)) for a, b in _SHAPES_COMPARE_
]
_SHAPES_COMPARE_TRIPLES: list[
    tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], bool]
] = [(a, b, c, are_shapes_compatible(a, b, c)) for a, b, c in _SHAPES_COMPARE_TRIPLES_]


# probably a better way of doing this,
# but we need to create a creator of generators, since one may call
# "for ... in itergen" multiple times.
class _itergen:
    def __init__(self, genfunc):
        self.genfunc = genfunc

    def __iter__(self):
        return self.genfunc()


@pytest.fixture
def comparison_two_shapes():
    def loop_shapes():
        for a, b, valid in _SHAPES_COMPARE:
            yield (a, b, valid)
            yield (b, a, valid)

    return _itergen(loop_shapes)


@pytest.fixture
def comparison_three_shapes():
    def loop_shapes():
        for a, b, c, valid in _SHAPES_COMPARE_TRIPLES:
            # yes, this isn't exhaustive of all permutations.
            yield (a, b, c, valid)
            yield (a, c, b, valid)
            yield (c, b, a, valid)

    return _itergen(loop_shapes)


def field_from_perm(shapes, coef, perm):
    return Field(
        shapes[perm[ShapeComponent.BASIS]],
        shapes[perm[ShapeComponent.POINT]],
        coef,
        perm,
    )


@pytest.fixture(
    params=[
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ]
)
def permutation(request):
    return request.param


@pytest.fixture
def field_of_different_shape_orders_testsep(permutation):
    def f(shapes, coef):
        yield field_from_perm(shapes, coef, permutation)

    return f


@pytest.fixture
def field_of_different_shape_orders():
    def f(shapes, coef):
        yield field_from_perm(shapes, coef, (0, 1, 2))
        yield field_from_perm(shapes, coef, (0, 2, 1))
        yield field_from_perm(shapes, coef, (1, 0, 2))
        yield field_from_perm(shapes, coef, (1, 2, 0))
        yield field_from_perm(shapes, coef, (2, 0, 1))
        yield field_from_perm(shapes, coef, (2, 1, 0))

    return f
