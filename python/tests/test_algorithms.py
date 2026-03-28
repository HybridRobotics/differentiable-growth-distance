import numpy as np
import pytest

import dgd


def _tf2_identity() -> np.ndarray:
    return np.eye(3, dtype=float)


def _tf3_identity() -> np.ndarray:
    return np.eye(4, dtype=float)


def test_growth_distance_and_solver_variants_2d():
    """Checks 2D GrowthDistance bindings for all solver entry points."""
    set1 = dgd.Circle(1.0)
    set2 = dgd.Circle(0.5)

    tf1 = _tf2_identity()
    tf2 = _tf2_identity()
    tf2[:2, 2] = np.array([3.0, 0.0])

    settings = dgd.Settings()

    out = dgd.Output2()
    gd = dgd.growth_distance(set1, tf1, set2, tf2, settings, out)
    assert gd > 0.0
    assert out.status == dgd.SolutionStatus.Optimal

    out_cp = dgd.Output2()
    gd_cp = dgd.growth_distance_cp(set1, tf1, set2, tf2, settings, out_cp)
    assert out_cp.status == dgd.SolutionStatus.Optimal

    out_trn = dgd.Output2()
    gd_trn = dgd.growth_distance_trn(set1, tf1, set2, tf2, settings, out_trn)
    assert out_trn.status == dgd.SolutionStatus.Optimal

    assert gd_cp == pytest.approx(gd, rel=1e-8, abs=1e-8)
    assert gd_trn == pytest.approx(gd, rel=1e-8, abs=1e-8)


def test_growth_distance_and_collision_halfspace_3d():
    """Checks 3D Halfspace overloads for growth distance and collision."""
    set1 = dgd.Cuboid(1.0, 2.0, 3.0, 0.1)
    set2 = dgd.Halfspace3(0.4)

    tf1 = _tf3_identity()
    tf2 = _tf3_identity()

    settings = dgd.Settings()
    out = dgd.Output3()

    tf1[:3, 3] = np.array([8.0, -7.0, 0.7])
    gd = dgd.growth_distance(set1, tf1, set2, tf2, settings, out)
    assert gd == pytest.approx(0.2, rel=1e-8, abs=1e-8)
    assert out.status == dgd.SolutionStatus.Optimal

    colliding = dgd.detect_collision(set1, tf1, set2, tf2, settings, out)
    assert colliding is True

    tf1[:3, 3] = np.array([8.0, -7.0, -0.1])
    dgd.growth_distance(set1, tf1, set2, tf2, settings, out)
    assert out.status == dgd.SolutionStatus.CoincidentCenters


def test_compute_kkt_nullspace_and_derivative_interfaces_3d():
    """Checks 3D KKT nullspace, gradient, and directional derivative bindings."""
    set1 = dgd.Sphere(1.0)
    set2 = dgd.Cuboid(1.0, 1.0, 1.0, 0.0)

    tf1 = _tf3_identity()
    tf2 = _tf3_identity()
    tf2[:3, 3] = np.array([3.0, 0.0, 0.0])

    settings = dgd.Settings()
    out = dgd.Output3()
    gd = dgd.growth_distance(set1, tf1, set2, tf2, settings, out)
    assert out.status == dgd.SolutionStatus.Optimal
    assert gd == pytest.approx(1.5, rel=1e-8, abs=1e-8)

    dd = dgd.DirectionalDerivative3()
    kkt_nullity = dgd.compute_kkt_nullspace(set1, tf1, set2, tf2, settings, out, dd)
    assert kkt_nullity == 1
    assert dd.z_nullity == 0
    assert dd.n_nullity == 1
    assert dd.value_differentiable is True

    td = dgd.TotalDerivative3()
    dgd.gd_gradient(tf1, tf2, settings, out, td)
    assert td.d_gd_tf1.shape == (6,)
    assert td.d_gd_tf2.shape == (6,)

    state1 = dgd.KinematicState3(tf=tf1, tw=np.zeros(6))
    state2 = dgd.KinematicState3(tf=tf2, tw=np.zeros(6))
    d_gd = dgd.gd_derivative(state1, state2, settings, out, dd)
    assert d_gd == pytest.approx(0.0, abs=1e-10)
    assert dd.d_gd == pytest.approx(0.0, abs=1e-10)


def test_factorize_solution_derivative_and_jacobian_3d():
    """Checks 3D factorization-based solution derivative and Jacobian APIs."""
    set1 = dgd.Sphere(1.0)
    set2 = dgd.Cuboid(1.0, 1.0, 1.0)

    tf1 = _tf3_identity()
    tf2 = _tf3_identity()
    tf2[:3, 3] = np.array([3.0, 0.0, 0.0])

    settings = dgd.Settings()
    out = dgd.Output3()
    dgd.growth_distance(set1, tf1, set2, tf2, settings, out)

    dd = dgd.DirectionalDerivative3()
    differentiable = dgd.factorize_kkt_system(set1, tf1, set2, tf2, settings, out, dd)
    assert isinstance(differentiable, bool)
    assert differentiable is True

    state1 = dgd.KinematicState3(tf=tf1, tw=np.array([0.1, -0.2, 0.3, 0.2, 0.0, -0.1]))
    state2 = dgd.KinematicState3(tf=tf2, tw=np.array([-0.3, 0.1, 0.0, -0.2, 0.2, 0.4]))
    dgd.gd_solution_derivative(state1, state2, settings, out, dd)

    td = dgd.TotalDerivative3()
    dgd.gd_jacobian(tf1, tf2, settings, out, dd, td)

    assert td.d_normal_tf1.shape == (3, 6)
    assert td.d_normal_tf2.shape == (3, 6)
    assert np.isfinite(td.d_normal_tf1).all()
    assert np.isfinite(td.d_normal_tf2).all()

    np.testing.assert_allclose(
        dd.d_normal,
        td.d_normal_tf1 @ state1.tw + td.d_normal_tf2 @ state2.tw,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        dd.d_z1, td.d_z1_tf1 @ state1.tw + td.d_z1_tf2 @ state2.tw, atol=1e-8
    )
    np.testing.assert_allclose(
        dd.d_z2, td.d_z2_tf1 @ state1.tw + td.d_z2_tf2 @ state2.tw, atol=1e-8
    )
    np.testing.assert_allclose(
        dd.d_gd, td.d_gd_tf1.dot(state1.tw) + td.d_gd_tf2.dot(state2.tw), atol=1e-8
    )


def test_warm_start_output_reuse_for_growth_distance_cp_2d():
    """Checks warm-start workflow by reusing the same output object in 2D."""
    set1 = dgd.Circle(1.0)
    set2 = dgd.Rectangle(1.0, 0.5)

    tf1 = _tf2_identity()
    tf2 = _tf2_identity()
    tf2[:2, 2] = np.array([4.0, 0.1])

    settings = dgd.Settings()
    out = dgd.Output2()

    gd0 = dgd.growth_distance_cp(set1, tf1, set2, tf2, settings, out, warm_start=False)
    assert out.status == dgd.SolutionStatus.Optimal

    tf2[:2, 2] = np.array([3.95, 0.1])
    gd1 = dgd.growth_distance_cp(set1, tf1, set2, tf2, settings, out, warm_start=True)
    assert out.status == dgd.SolutionStatus.Optimal
    assert gd1 > 0.0
    assert abs(gd1 - gd0) < 0.2
