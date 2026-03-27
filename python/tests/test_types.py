import math

import numpy as np

import dgd


def test_module_constants_are_exposed():
    """Checks that the scalar constants are present and valid."""
    assert math.isinf(dgd.INF)
    assert dgd.EPS > 0.0
    assert dgd.SQRT_EPS > 0.0
    assert abs(dgd.PI - math.pi) < 1e-12


def test_kinematic_state2_defaults_and_writeback():
    """Checks default state values and mutable NumPy-backed writeback in 2D."""
    state = dgd.KinematicState2()

    assert state.tf.shape == (3, 3)
    assert state.tw.shape == (3,)
    np.testing.assert_array_equal(state.tf, np.eye(3))
    np.testing.assert_array_equal(state.tw, np.zeros(3))

    state.tw[1] = 2.5
    assert state.tw[1] == 2.5

    state.tw = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(state.tw, np.array([1.0, 2.0, 3.0]))


def test_kinematic_state3_defaults_and_writeback():
    """Checks default state values and mutable NumPy-backed writeback in 3D."""
    state = dgd.KinematicState3()

    assert state.tf.shape == (4, 4)
    assert state.tw.shape == (6,)
    np.testing.assert_array_equal(state.tf, np.eye(4))
    np.testing.assert_array_equal(state.tw, np.zeros(6))

    state.tw[3] = 0.5
    assert state.tw[3] == 0.5

    updated = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    state.tw = updated
    np.testing.assert_array_equal(state.tw, updated)


def test_kinematic_state2_repr_round_trip():
    """Checks that repr output can be evaluated into an equivalent object."""
    original = dgd.KinematicState2(
        tf=np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]]),
        tw=np.array([0.5, 1.5, -2.0]),
    )

    recovered = eval(repr(original), {"dgd": dgd})
    np.testing.assert_array_equal(recovered.tf, original.tf)
    np.testing.assert_array_equal(recovered.tw, original.tw)


def test_kinematic_state3_repr_round_trip():
    """Checks that repr output can be evaluated into an equivalent object."""
    original = dgd.KinematicState3(
        tf=np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
        tw=np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]),
    )

    recovered = eval(repr(original), {"dgd": dgd})
    np.testing.assert_array_equal(recovered.tf, original.tf)
    np.testing.assert_array_equal(recovered.tw, original.tw)
