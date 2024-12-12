import numpy as np

import fastfem.fields.numpy_similes as fnp


def test_einsum_with_single_field(field_of_different_shape_orders_testsep):
    shapes = [(2, 3), (4, 1), (1, 2)]
    arr = np.random.rand(*shapes[0], *shapes[1], *shapes[2])
    factors = [np.random.rand(s[1]) for s in shapes]
    result = np.einsum("ijklmn,j,l,n", arr, *factors)
    for field in field_of_different_shape_orders_testsep(shapes, arr):
        tmp = fnp.einsum(field.shape_order_inverse[0], "ij,j", field, factors[0])
        tmp = fnp.einsum(field.shape_order_inverse[1], "ij,j", tmp, factors[1])
        tmp = fnp.einsum(field.shape_order_inverse[2], "ij,j", tmp, factors[2])
        np.testing.assert_allclose(tmp.coefficients, result)
