import numpy as np
import pytest

import dgd


def test_support_function_hint2_defaults_and_mutation():
    """Checks SupportFunctionHint2 default values and field mutation."""
    hint = dgd.SupportFunctionHint2()

    np.testing.assert_allclose(hint.n_prev, np.zeros(2))
    assert hint.idx_ws == -1

    hint.n_prev = np.array([1.0, 2.0], dtype=float)
    hint.idx_ws = 7

    np.testing.assert_allclose(hint.n_prev, np.array([1.0, 2.0]))
    assert hint.idx_ws == 7


def test_base_point_hint3_defaults_and_mutation():
    """Checks BasePointHint3 fields can be assigned and read back."""
    hint = dgd.BasePointHint3()

    hint.bc = np.array([0.2, 0.3, 0.5], dtype=float)
    hint.idx = np.array([4, 5, 6], dtype=int)

    np.testing.assert_allclose(hint.bc, np.array([0.2, 0.3, 0.5]))
    np.testing.assert_array_equal(hint.idx, np.array([4, 5, 6]))


def test_support_patch_hull_and_normal_cone_span_2d_fields():
    """Checks 2D support patch and normal cone scalar fields."""
    sph = dgd.SupportPatchHull2()
    ncs = dgd.NormalConeSpan2()

    sph.aff_dim = 1
    ncs.span_dim = 1

    assert sph.aff_dim == 1
    assert ncs.span_dim == 1


def test_support_patch_hull_and_normal_cone_span_3d_fields():
    """Checks 3D support patch and normal cone basis field behavior."""
    sph = dgd.SupportPatchHull3()
    ncs = dgd.NormalConeSpan3()

    sph.aff_dim = 1
    sph.basis = np.eye(3, 1, dtype=float)
    ncs.span_dim = 2
    ncs.basis = np.array([[1.0], [0.0], [0.0]], dtype=float)

    assert sph.aff_dim == 1
    assert sph.basis.shape == (3,)
    np.testing.assert_allclose(sph.basis, np.array([1.0, 0.0, 0.0]))
    assert ncs.span_dim == 2
    assert ncs.basis.shape == (3,)
    np.testing.assert_allclose(ncs.basis, np.array([1.0, 0.0, 0.0]))


def test_convex_set_class_level_api_and_abstract_instantiation():
    """Checks ConvexSet class API exposure and abstract-type behavior."""
    assert dgd.ConvexSet2.dimension() == 2
    assert dgd.ConvexSet3.dimension() == 3
    assert dgd.ConvexSet2.eps_sp() > 0.0
    assert dgd.ConvexSet3.eps_sp() > 0.0

    for method_name in [
        "support_function",
        "support_function_hess",
        "compute_local_geometry",
        "projection_derivative",
        "bounds",
        "require_unit_normal",
        "is_polytopic",
        "print_info",
    ]:
        assert hasattr(dgd.ConvexSet2, method_name)
        assert hasattr(dgd.ConvexSet3, method_name)

    with pytest.raises(TypeError):
        dgd.ConvexSet2()

    with pytest.raises(TypeError):
        dgd.ConvexSet3()
