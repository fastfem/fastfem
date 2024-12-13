from collections.abc import Callable, Iterable
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from fastfem.fields.field import Field


def transform_posmatrix_linearly(pos_matrix, mod, *args):
    if mod == "translate":
        if len(args) < 2:
            raise ValueError(f"modifier '{mod}' expects 2 arguments! (dx,dy)")
        vec = np.array([args[0], args[1]])
        pos_matrix = pos_matrix + vec  # no in-place op, since we want a copy
    elif mod == "rotate":
        if len(args) < 1:
            raise ValueError(f"modifier '{mod}' expects 1 argument! (angle)")
        t = args[0]
        rotmat = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        pos_matrix = (rotmat @ np.expand_dims(pos_matrix, -1)).squeeze(-1)
    elif mod == "scale":
        if len(args) < 2:
            raise ValueError(f"modifier '{mod}' expects 2 arguments! (scalex,scaley)")
        vec = np.array([args[0], args[1]])
        pos_matrix = pos_matrix * vec  # no in-place op, since we want a copy
    elif mod == "lin_trans":
        if len(args) < 4:
            raise ValueError(f"modifier '{mod}' expects 4 arguments! (m00,m01,m10,m11)")
        A = np.array([[args[0], args[1]], [args[2], args[3]]])
        pos_matrix = (A @ np.expand_dims(pos_matrix, -1)).squeeze(-1)
    else:
        raise ValueError(f"'{mod}' not acceptable element modifier!")
    return pos_matrix


_PRESET_TRANSFORMS_AFFINE = {
    "id": lambda x: x,
    "translated": lambda x: transform_posmatrix_linearly(x, "translate", 5, -2),
    "rotated": lambda x: transform_posmatrix_linearly(x, "rotate", 1),
    "x-scaled": lambda x: transform_posmatrix_linearly(x, "scale", 2, 1),
    "y-scaled": lambda x: transform_posmatrix_linearly(x, "scale", 1, 2),
    "combo1": lambda x: transform_posmatrix_linearly(
        transform_posmatrix_linearly(x, "lin_trans", 2, 1, -1, 1), "translate", -4, 2
    ),
    "combo2": lambda x: transform_posmatrix_linearly(
        transform_posmatrix_linearly(x, "lin_trans", 0.5, 1.3, 10, 0.3),
        "translate",
        300,
        600,
    ),
}


class AffineTransformHandler:
    def set_target_type(
        self,
        T: Callable[[NDArray], NDArray],
        target_type: Literal["point"] | Literal["vector"] | Literal["form"],
    ) -> Callable[[NDArray], NDArray]:
        if target_type == "point":
            return T
        if target_type == "vector":
            return lambda x: T(x) - T(np.zeros(2))
        if target_type == "form":
            shift = T(np.zeros(2))
            A_Tinv = np.linalg.inv(T(np.eye(2)) - shift).T
            return lambda x: (A_Tinv @ np.expand_dims(x, -1)).squeeze(-1)
        raise ValueError(
            f"target_type {target_type} not known. Must be 'point', 'vector', or 'form'"
        )

    def iterate_presets(
        self,
        target_type: Literal["point"] | Literal["vector"] | Literal["form"] = "point",
    ) -> Iterable[tuple[str, Callable[[NDArray], NDArray]]]:
        for k, T in _PRESET_TRANSFORMS_AFFINE.items():
            yield k, self.set_target_type(T, target_type)

    def __iter__(self):
        yield from _PRESET_TRANSFORMS_AFFINE.items()


@pytest.fixture
def affine_transforms():
    return AffineTransformHandler()


@pytest.fixture(scope="module", params=_PRESET_TRANSFORMS_AFFINE.keys())
def transformation(request):
    name = request.param
    return _PRESET_TRANSFORMS_AFFINE[name]


@pytest.fixture(scope="module", params=[0, 1, 2, 3])
def transform_stack(request):
    transforms = _PRESET_TRANSFORMS_AFFINE.values()
    # param is number of dims for element position array
    ndims = request.param
    stackshape = tuple(3 for _ in range(ndims))
    stacksize = np.prod(stackshape, dtype=int)

    def stack_transform(x: Field):
        y = np.empty(x.basis_shape + stackshape + x.stack_shape + x.point_shape)
        transformed = [f(x.coefficients) for f in transforms]
        for i in range(stacksize):
            y[..., *np.unravel_index(i, stackshape), ...] = transformed[
                i % len(transforms)
            ]
        return Field(x.basis_shape, x.point_shape, y)

    return stack_transform
