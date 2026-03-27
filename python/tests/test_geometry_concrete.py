from pathlib import Path

import numpy as np
import pytest

import dgd


def test_halfspace_defaults_and_validation():
    """Checks Halfspace defaults, mutability, and constructor validation."""
    hs2 = dgd.Halfspace2()
    hs3 = dgd.Halfspace3(0.2)

    assert hs2.margin == pytest.approx(0.0)
    assert hs3.margin == pytest.approx(0.2)
    assert dgd.Halfspace2.dimension() == 2
    assert dgd.Halfspace3.dimension() == 3

    hs2.margin = 0.5
    assert hs2.margin == pytest.approx(0.5)

    with pytest.raises(Exception, match="negative"):
        dgd.Halfspace2(-1.0)


def test_rectangle_and_ellipse_support_outputs():
    """Checks representative support outputs for 2D concrete classes."""
    ell = dgd.Ellipse(3.0, 2.0)
    sv_e, sp_e, differentiable, d_sp_n = ell.support_function_hess(np.array([1.0, 0.0]))
    assert sv_e == pytest.approx(3.0)
    np.testing.assert_allclose(sp_e, np.array([3.0, 0.0]))
    assert differentiable is True
    assert d_sp_n.shape == (2, 2)

    rect = dgd.Rectangle(2.0, 1.0)
    sv_r, sp_r = rect.support_function(np.array([1.0, 0.0]))
    assert sv_r == pytest.approx(2.0)
    assert sp_r.shape == (2,)
    assert sp_r[0] == pytest.approx(2.0)


def test_polygon_construction_support_and_vertices_roundtrip():
    """Checks Polygon construction and support function return shape."""
    vertices = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
    poly = dgd.Polygon(vertices, inradius=1.0)

    out_vertices = poly.vertices()
    assert out_vertices.shape == (4, 2)
    assert poly.nvertices() == 4

    sv, sp = poly.support_function(np.array([1.0, 0.0]))
    assert sv == pytest.approx(1.0)
    assert sp.shape == (2,)
    assert sp[0] == pytest.approx(1.0)


def test_3d_primitives_support_outputs():
    """Checks representative support outputs for 3D concrete classes."""
    cuboid = dgd.Cuboid(1.0, 2.0, 3.0)
    sv_c, sp_c = cuboid.support_function(np.array([0.0, 0.0, 1.0]))
    assert sv_c == pytest.approx(3.0)
    assert sp_c.shape == (3,)
    assert sp_c[2] == pytest.approx(3.0)

    capsule = dgd.Capsule(1.0, 0.5)
    sv_cap, sp_cap = capsule.support_function(np.array([1.0, 0.0, 0.0]))
    assert sv_cap == pytest.approx(1.5)
    np.testing.assert_allclose(sp_cap, np.array([1.5, 0.0, 0.0]))

    sphere = dgd.Sphere(2.5)
    sv_s, sp_s = sphere.support_function(np.array([1.0, 0.0, 0.0]))
    assert sv_s == pytest.approx(2.5)
    np.testing.assert_allclose(sp_s, np.array([2.5, 0.0, 0.0]))


def test_cone_and_frustum_offsets():
    """Checks cone/frustum bind offset accessors and basic construction."""
    cone = dgd.Cone(radius=1.0, height=2.0)
    frustum = dgd.Frustum(base_radius=1.0, top_radius=0.5, height=2.0)

    assert cone.offset() > 0.0
    assert frustum.offset() > 0.0


def test_polytope_from_vertices_and_base_properties():
    """Checks Polytope constructor and inherited mutable base properties."""
    vertices = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    poly = dgd.Polytope(vertices, inradius=1.0)

    assert poly.nvertices() == 8
    assert poly.vertices().shape == (8, 3)

    poly.eps_p = 1e-6
    poly.eps_d = 2e-6
    poly.inradius = 0.8
    assert poly.eps_p == pytest.approx(1e-6)
    assert poly.eps_d == pytest.approx(2e-6)
    assert poly.inradius == pytest.approx(0.8)


def test_mesh_bindings_with_meshloader_outputs():
    """Checks Mesh constructor with MeshLoader graph outputs."""
    cube_obj = Path(__file__).resolve().parents[2] / "tests" / "cube.obj"

    ml = dgd.MeshLoader()
    ml.load_obj(str(cube_obj), True)
    valid, vertices, graph = ml.make_vertex_graph()
    assert valid is True

    inradius, _ = ml.compute_inradius()
    assert inradius > 0.0

    mesh = dgd.Mesh(vertices, graph, inradius)
    assert mesh.nvertices() == 8
    assert mesh.vertices().shape == (8, 3)
    assert mesh.graph().ndim == 1

    sv, sp = mesh.support_function(np.array([1.0, 0.0, 0.0]))
    assert sv > 0.0
    assert sp.shape == (3,)
    assert sp[0] == pytest.approx(sv)


def test_concrete_constructor_validation_errors():
    """Checks invalid concrete geometry parameters raise clear exceptions."""
    with pytest.raises(Exception, match="Invalid"):
        dgd.Rectangle(0.0, 1.0)

    with pytest.raises(Exception, match="Invalid"):
        dgd.Cuboid(1.0, -1.0, 1.0)

    with pytest.raises(Exception, match="Invalid"):
        dgd.Cone(1.0, -2.0)

    with pytest.raises(Exception, match="Invalid"):
        dgd.Frustum(0.0, 0.0, 1.0)
