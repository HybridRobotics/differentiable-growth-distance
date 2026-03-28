import numpy as np
import pytest

import dgd


def _tf2_identity() -> np.ndarray:
    return np.eye(3, dtype=float)


def _tf3_identity() -> np.ndarray:
    return np.eye(4, dtype=float)


def test_solution_error_struct_defaults_and_mutation():
    """Checks SolutionError fields are Python-visible and mutable."""
    err = dgd.SolutionError()

    err.prim_dual_gap = 0.1
    err.prim_infeas_err = 0.2
    err.dual_infeas_err = 0.0

    assert err.prim_dual_gap == pytest.approx(0.1)
    assert err.prim_infeas_err == pytest.approx(0.2)
    assert err.dual_infeas_err == pytest.approx(0.0)


def test_compute_solution_error_for_optimal_solution():
    """Checks compute_solution_error returns near-zero errors for optimal output."""
    set1 = dgd.Circle(1.0)
    set2 = dgd.Rectangle(1.0, 0.5)

    tf1 = _tf2_identity()
    tf2 = _tf2_identity()
    tf2[:2, 2] = np.array([4.0, 0.1])

    settings = dgd.Settings()
    out = dgd.Output2()
    gd = dgd.growth_distance_cp(set1, tf1, set2, tf2, settings, out)

    assert gd > 0.0
    assert out.status == dgd.SolutionStatus.Optimal

    err = dgd.compute_solution_error(set1, tf1, set2, tf2, out)
    assert err.prim_dual_gap >= 0.0
    assert err.prim_infeas_err >= 0.0
    assert err.dual_infeas_err == pytest.approx(0.0)
    assert err.prim_dual_gap < 1e-7
    assert err.prim_infeas_err < 1e-8


def test_assert_collision_status_noncollision_and_collision_cases():
    """Checks assert_collision_status for both separated and colliding cases."""
    set1 = dgd.Sphere(1.0)
    set2 = dgd.Cuboid(1.0, 1.0, 1.0)

    tf1 = _tf3_identity()
    tf2 = _tf3_identity()

    settings = dgd.Settings()
    out = dgd.Output3()

    tf2[:3, 3] = np.array([6.0, 0.0, 0.0])
    collision = dgd.detect_collision(set1, tf1, set2, tf2, settings, out)
    assert collision is False
    assert dgd.assert_collision_status(set1, tf1, set2, tf2, out, collision)

    tf2[:3, 3] = np.array([0.1, 0.0, 0.0])
    collision = dgd.detect_collision(set1, tf1, set2, tf2, settings, out)
    assert collision is True
    assert dgd.assert_collision_status(set1, tf1, set2, tf2, out, collision)
