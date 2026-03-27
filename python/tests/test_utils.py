from pathlib import Path

import numpy as np
import pytest

import dgd


def test_graham_scan_returns_ccw_hull_vertices():
    """Checks graham_scan returns CCW hull vertices with duplicates removed."""
    points = np.array(
        [
            [0.0, 0.0],
            [3.0, 2.0],
            [2.0, 2.0],
            [1.5, 1.0],
            [3.0, 2.0],
            [2.5, 2.0],
            [0.0, 4.0],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    hull = dgd.graham_scan(points)
    expected = np.array([[0.0, 0.0], [3.0, 2.0], [0.0, 4.0]], dtype=float)

    assert hull.shape == (3, 2)
    np.testing.assert_allclose(hull, expected)


def test_compute_polygon_inradius_triangle():
    """Checks inradius computation on a known triangle/interior point."""
    vertices = np.array([[0.0, 0.0], [6.0, 4.0], [0.0, 4.0]], dtype=float)
    interior = np.array([3.0, 3.0], dtype=float)

    inradius = dgd.compute_polygon_inradius(vertices, interior)
    expected = np.sqrt(9.0 / 13.0)

    assert inradius == pytest.approx(expected)


def test_mesh_loader_process_points_and_vertex_graph():
    """Checks ProcessPoints and MakeVertexGraph outputs/shapes."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # Duplicate point to test handling.
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    ml = dgd.MeshLoader()
    ml.process_points(points)

    assert ml.npts() == 5

    valid, vertices, graph = ml.make_vertex_graph()
    assert valid is True
    assert vertices.shape == (5, 3)
    assert graph.ndim == 1
    assert int(graph[0]) == vertices.shape[0]
    assert int(graph[1]) == 6
    assert graph.size == 2 + 2 * 5 + 3 * 6


def test_mesh_loader_make_facet_graph_and_inradius():
    """Checks facet graph outputs and both inradius paths."""
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # Duplicate point to test handling.
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )

    ml = dgd.MeshLoader()
    ml.process_points(points)

    valid, normals, offsets, graph, interior = ml.make_facet_graph()
    assert valid is True
    assert normals.shape == (5, 3)
    assert offsets.shape == (5,)
    assert int(graph[0]) == 5
    assert int(graph[1]) == 8
    assert graph.size == 2 + 2 * 5 + 2 * 8
    assert interior.shape == (3,)

    inradius_h = ml.compute_inradius_from_halfspaces(normals, offsets, interior)
    inradius_qh, interior_qh = ml.compute_inradius()

    assert inradius_h > 0.0
    assert inradius_qh > 0.0
    assert interior_qh.shape == (3,)


def test_mesh_loader_load_obj_from_file():
    """Checks OBJ loading from tests/cube.obj and graph extraction."""
    cube_obj = Path(__file__).resolve().parents[2] / "tests" / "cube.obj"

    ml = dgd.MeshLoader()
    ml.load_obj(str(cube_obj), True)

    assert ml.npts() == 8

    valid, vertices, graph = ml.make_vertex_graph()
    assert valid is True
    assert vertices.shape == (8, 3)
    assert int(graph[0]) == 8
    assert int(graph[1]) == 12
    assert graph.size == 2 + 2 * 8 + 3 * 12


def test_mesh_loader_exception_translation():
    """Checks invalid inputs map to Python exceptions with clear messages."""
    ml = dgd.MeshLoader()

    bad_points = np.array([[0.0, 0.0, np.nan], [1.0, 0.0, 0.0]])
    with pytest.raises(RuntimeError, match="not finite"):
        ml.process_points(bad_points)

    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    offsets = np.array([0.0])
    interior = np.zeros(3)
    with pytest.raises(ValueError, match="matching first dimensions"):
        ml.compute_inradius_from_halfspaces(normals, offsets, interior)
