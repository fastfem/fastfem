import pytest


from fastfem.elements import spectral_element
from fastfem.elements.element import Element2D


# ======================================================================================
#                                  Adding elements
#     When adding new elements, insert them into the dictionary `elements_to_test`. The
# key used to insert the element will be referenced in the name of the test
# parameterization. This test suite will verify the basic functionality of the element,
# while things like integration will only be tested insofar as validating things such
# as field stacking. For an element to have a complete collection of tests, one should
# additionally have the following tested separately:
#
#   - interpolate_field(field,x,y). Agreement between different configurations of
#       stack_shape, field_shape or x,y shapes will be tested,
#       as well as linearity of solutions. So, the developer only needs to test a
#       spanning set of the scalar field space.
#
#   - compute_field_gradient(field, pos_matrix). As above, the developer only needs to
#       test a spanning set of the scalar field space. This test can be skipped by
#       appending the name of the element to
#       `elements_for_field_gradient_finitediff_test`, which tests
#       compute_field_gradient according to interpolate_field(),
#       reference_element_position_matrix() and finite differencing. It is assumed
#       that compute_field_gradient yields an exact value, and that nonlinear
#       transformations need not be checked.
#
#   - integrate_field(pos_matrix, field, jacobian_scale),
#   - integrate_basis_times_field(pos_matrix, field, indices, jacobian_scale)
#   - integrate_grad_basis_dot_field(pos_matrix,field,is_field_upper_index,indices,
#       jacobian_scale)
#   - integrate_grad_basis_dot_grad_field(pos_matrix,field,indices,jacobian_scale)
#       The same agreements will be verified (except with bilinearity of field and
#       jacobian_scale instead of linearity), so the developer only needs to test a
#       spanning set of the triple Cartesian product of the scalar field space and a
#       satisfactory set of transformed elements. Linear transformations on pos_matrix
#       will be tested by this suite, so only nonlinear transformations need to be
#       verified. Additionally, agreement with different index configurations will be
#       verified, so only indices=None needs to be taken as a case. Since
#       grad_basis_dot_field requires a vector field, so testing needs only be tested
#       with multiples of the elementary basis of R2.
#
#   -  for each
#       stack_shape == tuple(), field.field_shape == tuple(), indices == None
#   -  for each stack_shape == tuple(), indices == None, both
# ======================================================================================
elements_to_test = dict()
elements_for_field_gradient_finitediff_test = set()

# register spectral elements of different orders
for i in [3, 4, 5]:
    testname = f"spectral{i}"
    elements_to_test[testname] = spectral_element.SpectralElement2D(i)
    elements_for_field_gradient_finitediff_test.add(testname)


# ======================================================================================


@pytest.fixture(scope="module", params=elements_to_test.keys())
def element(request):
    return elements_to_test[request.param]


def test_basis_and_reference_shapes(element: Element2D):
    """Validates the shapes of basis_fields() and reference_element_position_matrix()
    against basis_shape().
    """
    shape = element.basis_shape()
    basis = element.basis_fields()

    assert basis.basis_shape == shape, (
        "basis_fields() should have basis_shape == basis_shape(). "
        + f"basis_fields().basis_shape: {basis.basis_shape} basis_shape(): {shape}."
    )
    assert basis.field_shape == tuple(), (
        "basis_fields() should be a stack of scalar fields. Instead, field_shape =="
        f" {basis.field_shape}."
    )

    ref_elem_pts = element.reference_element_position_matrix()

    assert ref_elem_pts.basis_shape == shape, (
        "reference_element_position_matrix() should have basis_shape == basis_shape(). "
        + "reference_element_position_matrix().basis_shape:"
        f" {ref_elem_pts.basis_shape} basis_shape(): {shape}."
    )


def test_reference_deformation_gradient(element: Element2D):
    """Validates the reference position matrix's deformation gradient, which should
    be the identity.
    """


# TODO all of this stuff


# def test_compute_field_gradient(element: Element2D, transform_stack):
#     ref_coords = element.reference_element_position_matrix()
#     coord_stack = transform_stack(ref_coords)
