import numpy as np
import pytest

import dgd

# ---------------------------------------------------------------------------
# SolutionStatus
# ---------------------------------------------------------------------------


def test_solution_status_values():
    """Checks that all SolutionStatus enum members are accessible."""
    assert dgd.SolutionStatus.Optimal != dgd.SolutionStatus.MaxIterReached
    assert (
        dgd.SolutionStatus.CoincidentCenters != dgd.SolutionStatus.IllConditionedInputs
    )


# ---------------------------------------------------------------------------
# Output2
# ---------------------------------------------------------------------------


def test_output2_defaults():
    """Checks default values and array shapes for Output2."""
    o = dgd.Output2()

    assert o.s1.shape == (2, 2)
    assert o.s2.shape == (2, 2)
    assert o.bc.shape == (2,)
    assert o.normal.shape == (2,)
    assert o.z1.shape == (2,)
    assert o.z2.shape == (2,)
    assert o.idx_s1.shape == (2,)
    assert o.idx_s2.shape == (2,)

    np.testing.assert_array_equal(o.s1, np.zeros((2, 2)))
    np.testing.assert_array_equal(o.bc, np.zeros(2))
    np.testing.assert_array_equal(o.normal, np.zeros(2))

    assert o.growth_dist_ub == dgd.INF
    assert o.growth_dist_lb == 0.0
    assert o.iter == 0
    assert o.status == dgd.SolutionStatus.MaxIterReached


def test_output2_field_mutation():
    """Checks that Output2 fields can be assigned."""
    o = dgd.Output2()

    o.growth_dist_ub = 1.5
    o.growth_dist_lb = 1.2
    o.iter = 7
    o.status = dgd.SolutionStatus.Optimal

    assert o.growth_dist_ub == pytest.approx(1.5)
    assert o.growth_dist_lb == pytest.approx(1.2)
    assert o.iter == 7
    assert o.status == dgd.SolutionStatus.Optimal

    o.normal = np.array([1.0, 0.0])
    np.testing.assert_array_equal(o.normal, np.array([1.0, 0.0]))


# ---------------------------------------------------------------------------
# Output3
# ---------------------------------------------------------------------------


def test_output3_defaults():
    """Checks default values and array shapes for Output3."""
    o = dgd.Output3()

    assert o.s1.shape == (3, 3)
    assert o.s2.shape == (3, 3)
    assert o.bc.shape == (3,)
    assert o.normal.shape == (3,)
    assert o.z1.shape == (3,)
    assert o.z2.shape == (3,)
    assert o.idx_s1.shape == (3,)
    assert o.idx_s2.shape == (3,)

    assert o.growth_dist_ub == dgd.INF
    assert o.growth_dist_lb == 0.0
    assert o.iter == 0


# ---------------------------------------------------------------------------
# DirectionalDerivative2 / DirectionalDerivative3
# ---------------------------------------------------------------------------


def test_directional_derivative2_defaults():
    """Checks default values and shapes for DirectionalDerivative2."""
    dd = dgd.DirectionalDerivative2()

    assert dd.z_nullspace.shape == (2,)
    assert dd.n_nullspace.shape == (2, 2)
    assert dd.d_normal.shape == (2,)
    assert dd.d_z1.shape == (2,)
    assert dd.d_z2.shape == (2,)

    assert dd.d_gd == 0.0
    assert dd.z_nullity == 0
    assert dd.n_nullity == 0
    assert dd.value_differentiable is False
    assert dd.differentiable is False


def test_directional_derivative3_defaults():
    """Checks default values and shapes for DirectionalDerivative3."""
    dd = dgd.DirectionalDerivative3()

    assert dd.z_nullspace.shape == (3, 2)
    assert dd.n_nullspace.shape == (3, 3)
    assert dd.d_normal.shape == (3,)
    assert dd.d_z1.shape == (3,)
    assert dd.d_z2.shape == (3,)

    assert dd.d_gd == 0.0
    assert dd.z_nullity == 0


# ---------------------------------------------------------------------------
# TotalDerivative2 / TotalDerivative3
# ---------------------------------------------------------------------------


def test_total_derivative2_defaults():
    """Checks default values and shapes for TotalDerivative2."""
    td = dgd.TotalDerivative2()

    assert td.d_normal_tf1.shape == (2, 3)
    assert td.d_normal_tf2.shape == (2, 3)
    assert td.d_z1_tf1.shape == (2, 3)
    assert td.d_z1_tf2.shape == (2, 3)
    assert td.d_z2_tf1.shape == (2, 3)
    assert td.d_z2_tf2.shape == (2, 3)
    assert td.d_gd_tf1.shape == (3,)
    assert td.d_gd_tf2.shape == (3,)

    np.testing.assert_array_equal(td.d_gd_tf1, np.zeros(3))
    np.testing.assert_array_equal(td.d_gd_tf2, np.zeros(3))


def test_total_derivative3_defaults():
    """Checks default values and shapes for TotalDerivative3."""
    td = dgd.TotalDerivative3()

    assert td.d_normal_tf1.shape == (3, 6)
    assert td.d_normal_tf2.shape == (3, 6)
    assert td.d_z1_tf1.shape == (3, 6)
    assert td.d_gd_tf1.shape == (6,)
    assert td.d_gd_tf2.shape == (6,)
