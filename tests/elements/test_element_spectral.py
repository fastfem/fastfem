from typing import Callable
import pytest
import numpy as np

from fastfem.elements import spectral_element
from fastfem.fields.field import Field


@pytest.fixture(scope="module", params=[3, 4, 5])
def element(request):
    order = request.param
    elem = spectral_element.SpectralElement2D(order)
    return elem


@pytest.fixture(scope="module")
def transformed_element(element, transformation):
    return (
        element,
        Field(
            element.basis_shape(),
            (2,),
            transformation(element.reference_element_position_matrix().coefficients),
        ),
        transformation,
    )


@pytest.fixture(scope="module")
def transformed_element_stack(transform_stack, element):

    transformed = transform_stack(element.reference_element_position_matrix())
    ndims = len(transformed.shape) - len(element.basis_shape())
    pts_stack = np.permute_dims(
        transformed, (ndims, ndims + 1) + tuple(range(ndims)) + (-1,)
    )
    return element, pts_stack, transform_stack


@pytest.fixture(
    params=[(0, 0), (-1, -1), (1, -1), (1, 1), (-1, 1), (0.5, 0.5), (-0.33, 0.84)]
)
def ref_coords(request):
    return request.param


@pytest.fixture(
    params=[
        np.array((0, 0)),
        np.array(((-1, -1), (1, -1), (1, 1), (-1, 1))),
        np.array((((-0.5, -0.3), (0.7, -0.2)), ((0.2, 0.8), (-0.1, 1)))),
    ]
)
def ref_coords_arr(request):
    return request.param


_broadcastable_pairs = [
    ((5,), (5,)),
    ((2, 6), (2, 6)),
    ((2, 4, 3), (2, 4, 3)),
    ((3,), (4, 3)),
    ((4, 3), (3,)),
    ((1, 3), (4, 3)),
    ((4, 3), (1, 3)),
    ((4, 1, 2), (2, 1)),
    ((2, 1), (4, 1, 2)),
]


@pytest.fixture(params=_broadcastable_pairs)
def broadcastable_shapes(request):
    return request.param[0], request.param[1], np.broadcast_shapes(*request.param)


@pytest.fixture
def broadcastable_shape_triples(broadcastable_shapes):
    a, b, target = broadcastable_shapes

    num_entries = 3  # for triples

    # what each entry can be (this fixture fails if this is >= 10)
    candidates = [tuple(), a, b]
    # the tuple needs at least these
    requires = [1, 2]

    def is_valid(inds):
        # do checks:
        if any(r not in inds for r in requires):
            # we do not have a required entry
            return False

        # placeholder for more checks

        return True

    ind_gen = (
        [int(c) for c in np.base_repr(i, len(candidates))]
        for i in range(len(candidates) ** num_entries)
    )
    return target, ([candidates[i] for i in inds] for inds in ind_gen if is_valid(inds))


# ===================


def test_lagrange_poly_coefs1D(element):
    elem = element[0]
    elem._lagrange_derivs = dict()
    elem._lagrange_derivs[0] = elem._lagrange_polys

    sigfigs = 5
    # we will use central finite difference which has O(h^2) error
    h = 10 ** -((sigfigs + 3) // 2)

    for deriv_order in range(0, elem.degree + 1):
        # verify that the lagrange derivatives are being set in the dictionary.
        P = elem.lagrange_poly1D(deriv_order)
        np.testing.assert_almost_equal(
            elem._lagrange_derivs[deriv_order],
            P,
            err_msg=(
                f"derivative L^{({deriv_order})} is not being set properly in the"
                " dictionary!"
            ),
        )
        assert set(elem._lagrange_derivs) == set(
            np.arange(deriv_order + 1)
        ), "Expected dictionary keys not found! Are they being improperly stored?"

    for deriv_order in range(1, elem.degree + 1):
        num_terms = elem.degree + 1 - deriv_order
        # polynomials of degree p are uniquely defined by their values at p+1 unique points. This is how we check equality.
        test_x = np.linspace(-1, 1, num_terms)

        # L_i^(deriv_order), compare to L_i^(...-1) with finite difference
        # polys are stored as P[i,k] : component c in term cx^k of poly P_i
        # L_i^(deriv_order-1)
        def L(x):
            return np.einsum(
                "ia,...a",
                elem.lagrange_poly1D(deriv_order - 1),
                np.expand_dims(x, -1) ** np.arange(num_terms + 1),
            )

        # L_i^(deriv_order)
        def Lp(x):
            return np.einsum(
                "ia,...a",
                elem.lagrange_poly1D(deriv_order),
                np.expand_dims(x, -1) ** np.arange(num_terms),
            )

        np.testing.assert_almost_equal(
            Lp(test_x),
            (L(test_x + h) - L(test_x - h)) / (2 * h),
            decimal=sigfigs,
            err_msg=(
                f"derivative L^({deriv_order}) does not match the central difference on"
                f" L^({deriv_order-1})!"
            ),
        )


def test_lagrange_evals1D(element, broadcastable_shapes):
    elem = element[0]

    for deriv_order in range(0, elem.degree + 1):
        # verify that the lagrange derivatives are evaluated correctly
        P = elem.lagrange_poly1D(deriv_order)
        test_points = np.linspace(-1, 1, np.prod(broadcastable_shapes[1])).reshape(
            broadcastable_shapes[1]
        )
        test_indices = (
            np.arange(np.prod(broadcastable_shapes[0])).reshape(broadcastable_shapes[0])
            % P.shape[0]
        )

        eval_pts = elem.lagrange_eval1D(deriv_order, test_indices, test_points)
        assert (
            eval_pts.shape == broadcastable_shapes[2]
        ), "Did not broadcast into the right shape!"

        test_points = np.broadcast_to(test_points, broadcastable_shapes[2])
        test_indices = np.broadcast_to(test_indices, broadcastable_shapes[2])

        it = np.nditer(eval_pts, flags=["multi_index"])
        degp1 = elem.degree + 1 - deriv_order  # degree+1 of P
        for Px in it:
            ind = test_indices[it.multi_index]
            x = test_points[it.multi_index]
            np.testing.assert_almost_equal(
                Px,
                np.dot(P[ind, :], x ** np.arange(degp1)),
                err_msg=(
                    f"index {it.multi_index}: L_{ind}^({deriv_order}) ({x})"
                    " disagreement"
                ),
            )


# @pytest.mark.skip
def test_interpolate_field(element, ref_coords):
    field = element.basis_fields()
    x, y = ref_coords
    interp_vals = element.interpolate_field(field, x, y)
    # relies on correctness of lagrange_eval1D
    Lx = element.lagrange_eval1D(0, None, x=x)
    Ly = element.lagrange_eval1D(0, None, x=y)

    np.testing.assert_almost_equal(
        interp_vals,
        Lx[:, np.newaxis] * Ly[np.newaxis, :],
    )


def test_real_to_reference_interior(transformed_element, ref_coords):
    elem = transformed_element[0]
    points = transformed_element[1]
    transformation = transformed_element[2]

    true_pos = transformation(ref_coords)

    recover_ref = elem.locate_point(
        points, true_pos[0], true_pos[1], tol=1e-10, ignore_out_of_bounds=True
    )
    assert recover_ref[1], (
        "Test with ignore_out_of_bounds flag. Should be True for point being found"
        " (loss(recover_ref) < tol)."
    )
    np.testing.assert_almost_equal(
        recover_ref[0], ref_coords, err_msg="Test with ignore_out_of_bounds flag"
    )

    recover_ref = elem.locate_point(
        points, true_pos[0], true_pos[1], tol=1e-10, ignore_out_of_bounds=False
    )
    assert recover_ref[1], (
        "Test without ignore_out_of_bounds flag. Should be True for point being found"
        " (loss(recover_ref) < tol)."
    )
    np.testing.assert_almost_equal(
        recover_ref[0], ref_coords, err_msg="Test without ignore_out_of_bounds flag"
    )


def test_real_to_reference_degen_elem(element):
    points = element.reference_element_position_matrix().coefficients
    points[..., 0] = np.abs(points[..., 0])
    try:
        element.locate_point(points, 1e-10, 0.3)
    except spectral_element.DeformationGradient2DBadnessException as e:
        assert e.x == pytest.approx(0, abs=1e-5), "error should be on local x=0"
        return
    assert False, "Correct Exception not thrown!"


@pytest.mark.skip("update code and move it to general element tests")
def test_field_grad(transformed_element):
    elem, points, transformation = transformed_element
    X = np.linspace(-1, 1, elem.num_nodes)[:, np.newaxis] + np.zeros(
        (1, elem.num_nodes)
    )
    Y = np.linspace(-1, 1, elem.num_nodes)[np.newaxis, :] + np.zeros(
        (elem.num_nodes, 1)
    )

    sigfigs = 5
    # we will use central finite difference which has O(h^2) error
    h = 10 ** -((sigfigs + 3) // 2)

    # this hinges on linearity of field values
    field = np.zeros((elem.num_nodes, elem.num_nodes, elem.num_nodes, elem.num_nodes))
    fieldshape = (elem.num_nodes, elem.num_nodes)
    enumeration = (
        np.arange(elem.num_nodes**2) % elem.num_nodes,
        np.arange(elem.num_nodes**2) // elem.num_nodes,
    )
    field[enumeration[0], enumeration[1], enumeration[0], enumeration[1]] = 1
    grads = elem.interpolate_field_gradient(field, X, Y, fieldshape=fieldshape)
    cartgrads = elem.interpolate_field_gradient(
        field, X, Y, pos_matrix=points, fieldshape=fieldshape
    )
    x_derivs = (
        elem.interpolate_field(field, X + h, Y, fieldshape=fieldshape)
        - elem.interpolate_field(field, X - h, Y, fieldshape=fieldshape)
    ) / (2 * h)
    y_derivs = (
        elem.interpolate_field(field, X, Y + h, fieldshape=fieldshape)
        - elem.interpolate_field(field, X, Y - h, fieldshape=fieldshape)
    ) / (2 * h)
    grads_comp = np.stack((x_derivs, y_derivs), -1)
    np.testing.assert_almost_equal(
        grads,
        grads_comp,
        decimal=sigfigs,
        err_msg="Local-coordinate gradient disagreement",
    )

    def_grad = elem.interpolate_deformation_gradient(points, X, Y)
    grads_cart_to_lag = np.einsum("...ij,...i->...j", def_grad, cartgrads)
    np.testing.assert_almost_equal(
        grads_cart_to_lag,
        grads,
        err_msg="Local->global->local disagrees with local",
    )

    # values are correct, now test shaped accessing: first, single values
    np.testing.assert_almost_equal(
        elem.interpolate_field_gradient(field, X[0, 0], Y[0, 0], fieldshape=fieldshape),
        grads[0, 0, ...],
        err_msg="Shape-invariance test: Single floats fail",
    )

    # values are correct, now test shaped accessing: use single values to verify
    for a, b in _broadcastable_pairs:
        c = np.broadcast_shapes(a, b)
        accessor_a = (np.arange(np.prod(a) * 2) % elem.num_nodes).reshape((2,) + a)
        accessor_b = (np.arange(np.prod(b) * 2) % elem.num_nodes).reshape((2,) + b)
        Xa = X[*accessor_a]
        Yb = Y[*accessor_b]
        res = elem.interpolate_field_gradient(field, Xa, Yb, fieldshape=fieldshape)
        Xc = Xa + 0 * Yb
        Yc = Yb + 0 * Xa
        it = np.nditer(Xc, flags=["multi_index"])
        for x in it:
            ind = it.multi_index
            np.testing.assert_almost_equal(
                res[*ind, ...],
                elem.interpolate_field_gradient(
                    field, x, Yc[ind], fieldshape=fieldshape
                ),
                err_msg=f"Shape-invariance test: pair {a}-{b} (broadcast to {c}) fails",
            )


def test_integrate_field(
    transformed_element: tuple[spectral_element.SpectralElement2D, Field, Callable],
):
    element, pointfield, transform = transformed_element
    field = element.basis_fields()
    jac_scale = Field(
        element.basis_shape(), tuple(), field.coefficients[..., np.newaxis, np.newaxis]
    )
    result = element.integrate_field(pointfield, field, jac_scale)

    kronecker = field.coefficients.copy()
    kronecker *= element.weights[:, np.newaxis, np.newaxis, np.newaxis]
    kronecker *= element.weights[np.newaxis, :, np.newaxis, np.newaxis]

    def_grad = element.compute_field_gradient(pointfield)
    jac = np.abs(np.linalg.det(def_grad.coefficients))
    kronecker *= jac[:, :, np.newaxis, np.newaxis]

    np.testing.assert_almost_equal(result, kronecker)


def test_integrate_basis_times_field(
    transformed_element: tuple[spectral_element.SpectralElement2D, Field, Callable],
):
    element, pointfield, transform = transformed_element
    field = element.basis_fields()
    jac_scale = Field(
        element.basis_shape(), tuple(), field.coefficients[..., np.newaxis, np.newaxis]
    )

    result = element.integrate_basis_times_field(
        pointfield, field, jacobian_scale=jac_scale
    )

    kronecker = field.coefficients.copy()

    triple_kronecker = np.einsum("abcd,cdef->abcdef", kronecker, kronecker)
    triple_kronecker *= element.weights[
        :, np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis
    ]
    triple_kronecker *= element.weights[
        np.newaxis, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis
    ]
    def_grad = element.compute_field_gradient(pointfield)
    jac = np.abs(np.linalg.det(def_grad.coefficients))
    triple_kronecker *= jac[:, :, np.newaxis, np.newaxis, np.newaxis, np.newaxis]

    np.testing.assert_almost_equal(result, triple_kronecker)


def test_integrate_grad_basis_dot_field(
    transformed_element: tuple[spectral_element.SpectralElement2D, Field, Callable],
):
    element, pointfield, transform = transformed_element
    field = element.basis_fields()
    jac_scale = Field(
        element.basis_shape(), tuple(), field.coefficients[..., np.newaxis, np.newaxis]
    )
    e_field = Field(
        element.basis_shape(),
        (2,),
        np.eye(2)[np.newaxis, np.newaxis, :, *((np.newaxis,) * 4), :]
        * field.coefficients[:, :, *((np.newaxis,) * 3), :, :, np.newaxis],
    )

    grad_basis = element.compute_field_gradient(
        element.basis_fields(), pointfield
    )  # ^g
    def_grad = element.compute_field_gradient(pointfield)  # [F]^g_l
    kronecker = field.coefficients.copy()

    # int(grad_basis * field * jac_scale)
    expect = np.einsum(
        "abcdg,abef,abhi,a,b,ab->cdgefhi",
        grad_basis.coefficients,
        kronecker,
        kronecker,
        element.weights,
        element.weights,
        np.abs(np.linalg.det(def_grad.coefficients)),
    )

    result = element.integrate_grad_basis_dot_field(
        pointfield, e_field, jacobian_scale=jac_scale
    )

    np.testing.assert_almost_equal(result, expect)


def meshquad(X, Y, F):
    """
    In case this is ever needed, this is a function to compute the integral on a 2D mesh.
    Integrals are estimated by estimating integrals on triangles, which are defined by
    indices ([i,j] - [i+1,j] - [i,j+1]) ([i+1,j+1] - [i+1,j] - [i,j+1])
    """

    def cross_mag(XA, YA, XB, YB):
        return np.abs(XA * YB - XB * YA)

    F00 = F[:-1, :-1]
    F01 = F[:-1, 1:]
    F10 = F[1:, :-1]
    F11 = F[1:, 1:]
    X00 = X[:-1, :-1]
    X01 = X[:-1, 1:]
    X10 = X[1:, :-1]
    X11 = X[1:, 1:]
    Y00 = Y[:-1, :-1]
    Y01 = Y[:-1, 1:]
    Y10 = Y[1:, :-1]
    Y11 = Y[1:, 1:]
    axes = np.arange(X.ndim)
    return np.sum(
        cross_mag(X10 - X00, Y10 - Y00, X01 - X00, Y01 - Y00) * (F00 + F01 + F10) / 6,
        axes,
    ) + np.sum(
        cross_mag(X10 - X11, Y10 - Y11, X01 - X11, Y01 - Y11) * (F11 + F01 + F10) / 6,
        axes,
    )


@pytest.mark.skip("we need to migrate this code over and generalize.")
def test_stiffness_matrix(transformed_element):
    elem, points, transformation = transformed_element
    # weights [i,j] times jacobian
    w = (
        elem.weights[:, np.newaxis]
        * elem.weights[np.newaxis, :]
        * np.abs(
            np.linalg.det(
                elem.interpolate_deformation_gradient(
                    points,
                    np.arange(elem.num_nodes),
                    np.arange(elem.num_nodes)[np.newaxis, :],
                )
            )
        )
    )

    # F(x_i,x_j) = delta_{im}delta_{jn}

    # equiv: # L_m(x_i)L_n(x_j)
    field = np.zeros((elem.num_nodes, elem.num_nodes, elem.num_nodes, elem.num_nodes))
    fieldshape = (elem.num_nodes, elem.num_nodes)
    enumeration = (
        np.arange(elem.num_nodes**2) % elem.num_nodes,
        np.arange(elem.num_nodes**2) // elem.num_nodes,
    )
    field[enumeration[0], enumeration[1], enumeration[0], enumeration[1]] = 1

    # [i,j, m,n, dim] partial_dim phi_{mn}(xi,xj)
    field_grad = elem.interpolate_field_gradient(
        field,
        elem.knots[:, np.newaxis],
        elem.knots[np.newaxis, :],
        pos_matrix=points,
        fieldshape=fieldshape,
    )

    # sum_{ij} w_{ij}( partial_dim phi_{mn}(xi,xj) )( partial_dim F_{ab}(xi,xj) )
    stiff = np.einsum("ij,ijmnd,ijabd->mnab", w, field_grad, field_grad)

    np.testing.assert_almost_equal(
        elem.integrate_grad_basis_dot_grad_field(points, field, fieldshape=fieldshape),
        stiff,
    )

    np.testing.assert_almost_equal(
        elem.integrate_grad_basis_dot_grad_field(points), np.einsum("mnmn->mn", stiff)
    )


@pytest.mark.parametrize("deg", [1, 2, 5, 10])
def test_gll_build(deg):
    x, w, L = spectral_element._build_GLL(deg)

    # verify shapes
    assert x.shape == (deg + 1,)
    assert w.shape == (deg + 1,)
    assert L.shape == (deg + 1, deg + 1)

    # verify quad degree
    for i in range(deg * 2 - 1):

        def F(x):
            return ((x + 1) / 2) ** i

        quad = sum(F(x) * w)
        res_true = 2 / (i + 1)
        assert quad == pytest.approx(res_true, abs=1e-8), (
            "GLL quadrature must be exact for polynomials up to degree 2n-1"
            f" ({2*deg - 1})! "
            + f"Failed at degree {i}."
        )

    # verify L
    evals = np.einsum(
        "ik,jk->ij", L, x[:, np.newaxis] ** np.arange(deg + 1)[np.newaxis, :]
    )
    np.testing.assert_almost_equal(evals, np.eye(deg + 1), decimal=8)


if __name__ == "__main__":
    pass
