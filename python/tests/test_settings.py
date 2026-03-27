import dgd


def test_warmstarttype_values():
    """Checks that WarmStartType enum members are accessible."""
    assert dgd.WarmStartType.Primal != dgd.WarmStartType.Dual
    assert dgd.WarmStartType.__members__["Primal"] == dgd.WarmStartType.Primal
    assert dgd.WarmStartType.__members__["Dual"] == dgd.WarmStartType.Dual


def test_twistframe_values():
    """Checks that TwistFrame enum members are accessible."""
    assert dgd.TwistFrame.Spatial != dgd.TwistFrame.Hybrid
    assert dgd.TwistFrame.Hybrid != dgd.TwistFrame.Body
    assert dgd.TwistFrame.__members__["Spatial"] == dgd.TwistFrame.Spatial
    assert dgd.TwistFrame.__members__["Hybrid"] == dgd.TwistFrame.Hybrid
    assert dgd.TwistFrame.__members__["Body"] == dgd.TwistFrame.Body


def test_settings_defaults():
    """Checks that Settings defaults match the C++ header values."""
    s = dgd.Settings()

    assert s.min_center_dist == dgd.SQRT_EPS
    assert s.rel_tol == 1.0 + dgd.SQRT_EPS
    assert s.nullspace_tol == dgd.SQRT_EPS
    assert s.jac_tol == dgd.SQRT_EPS
    assert s.max_iter == 100
    assert s.ws_type == dgd.WarmStartType.Primal
    assert s.twist_frame == dgd.TwistFrame.Hybrid


def test_settings_field_mutation():
    """Checks that all Settings fields can be written and read back."""
    s = dgd.Settings()

    s.min_center_dist = 1e-6
    assert s.min_center_dist == 1e-6

    s.rel_tol = 1.001
    assert s.rel_tol == 1.001

    s.nullspace_tol = 1e-5
    assert s.nullspace_tol == 1e-5

    s.jac_tol = 1e-4
    assert s.jac_tol == 1e-4

    s.max_iter = 50
    assert s.max_iter == 50

    s.ws_type = dgd.WarmStartType.Dual
    assert s.ws_type == dgd.WarmStartType.Dual

    s.twist_frame = dgd.TwistFrame.Body
    assert s.twist_frame == dgd.TwistFrame.Body


def test_settings_repr_contains_key_fields():
    """Checks that __repr__ mentions the main fields."""
    s = dgd.Settings()
    r = repr(s)
    assert "max_iter=100" in r
    assert "WarmStartType.Primal" in r
    assert "TwistFrame.Hybrid" in r
