// Copyright 2025 Akshay Thirugnanam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file utils.cc
 * @brief Bindings for geometry helper utilities.
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <stdexcept>

#include "binding_helpers.h"
#include "dgd/graham_scan.h"
#include "dgd/mesh_loader.h"

namespace py = pybind11;
using namespace dgd;

using pybind_helpers::MatX2r;
using pybind_helpers::MatX3r;
using pybind_helpers::MatXToVec;
using pybind_helpers::VeciToVecXi;
using pybind_helpers::VecrToVecXr;
using pybind_helpers::VecToMatX;
using pybind_helpers::VecXrToVecr;

void bind_utils(py::module_& m) {
  // ------------------------------------------------------------------
  // Graham scan function
  // ------------------------------------------------------------------
  m.def(
      "graham_scan",
      [](const MatX2r& points) {
        const auto pts = MatXToVec<2>(points);
        std::vector<Vec2r> hull;
        GrahamScan(pts, hull);
        return VecToMatX<2>(hull);
      },
      py::arg("points"),
      R"doc(
Computes the 2D convex hull using Graham scan.

Parameters
----------
points : numpy.ndarray, shape (n, 2)
		Input 2D points.

Returns
-------
numpy.ndarray, shape (m, 2)
		Convex hull vertices in CCW order with collinear duplicates removed.
)doc");

  // ------------------------------------------------------------------
  // Polygon inradius function
  // ------------------------------------------------------------------
  m.def(
      "compute_polygon_inradius",
      [](const MatX2r& vertices, const Vec2r& interior_point) {
        const auto vert = MatXToVec<2>(vertices);
        return ComputePolygonInradius(vert, interior_point);
      },
      py::arg("vertices"), py::arg("interior_point"),
      R"doc(
Computes polygon inradius at a given interior point.

Parameters
----------
vertices : numpy.ndarray, shape (n, 2)
		Polygon vertices in CCW order.
interior_point : numpy.ndarray, shape (2,)
		A point in the polygon interior.

Returns
-------
float
		Inradius at the interior point (negative if the point is outside).
)doc");

  // ------------------------------------------------------------------
  // MeshLoader
  // ------------------------------------------------------------------
  py::class_<MeshLoader>(m, "MeshLoader",
                         R"doc(
Loads 3D meshes and computes convex-hull graph representations.
)doc")
      .def(py::init<int>(), py::arg("maxhullvert") = 10000,
           "Constructs a MeshLoader.")
      .def("load_obj", &MeshLoader::LoadObj, py::arg("input"),
           py::arg("is_file") = true,
           "Loads from an OBJ filename or an OBJ string.")
      .def(
          "process_points",
          [](MeshLoader& self, const MatX3r& points) {
            self.ProcessPoints(MatXToVec<3>(points));
          },
          py::arg("points"),
          "Converts points to internal format and removes duplicates.")
      .def(
          "make_vertex_graph",
          [](MeshLoader& self) {
            std::vector<Vec3r> vertices;
            std::vector<int> graph;
            const bool valid = self.MakeVertexGraph(vertices, graph);
            return py::make_tuple(valid, VecToMatX<3>(vertices),
                                  VeciToVecXi(graph));
          },
          R"doc(
Builds the convex-hull vertex graph.

Returns
-------
tuple[bool, numpy.ndarray, numpy.ndarray]
		(valid, vertices, graph) where:
		- vertices has shape (nvert, 3)
		- graph has shape (k,)

Graph layout
------------
graph[0] = nvert
graph[1] = nface
graph[2 : 2 + nvert] = vert_edgeadr
graph[2 + nvert : ] = edge_localid records terminated by -1
)doc")
      .def(
          "make_facet_graph",
          [](MeshLoader& self) {
            std::vector<Vec3r> normals;
            std::vector<Real> offsets;
            std::vector<int> graph;
            Vec3r interior_point = Vec3r::Zero();
            const bool valid =
                self.MakeFacetGraph(normals, offsets, graph, interior_point);
            return py::make_tuple(valid, VecToMatX<3>(normals),
                                  VecrToVecXr(offsets), VeciToVecXi(graph),
                                  interior_point);
          },
          R"doc(
Builds the convex-hull facet graph and returns an interior point.

Returns
-------
tuple[bool, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
		(valid, normals, offsets, graph, interior_point) where:
		- normals has shape (nfacet, 3)
		- offsets has shape (nfacet,)
		- graph has shape (k,)
		- interior_point has shape (3,)

Graph layout
------------
graph[0] = nfacet
graph[1] = nridge
graph[2 : 2 + nfacet] = facet_ridgeadr
graph[2 + nfacet : ] = ridge_localid records terminated by -1
)doc")
      .def(
          "compute_inradius_from_halfspaces",
          [](const MeshLoader& self, const MatX3r& normals,
             const VecXr& offsets, const Vec3r& interior_point) {
            if (normals.rows() != offsets.size()) {
              throw std::invalid_argument(
                  "normals and offsets must have matching first dimensions");
            }
            const auto n = MatXToVec<3>(normals);
            const auto d = VecXrToVecr(offsets);
            return self.ComputeInradius(n, d, interior_point);
          },
          py::arg("normals"), py::arg("offsets"), py::arg("interior_point"),
          "Computes inradius from half-space representation and an interior "
          "point.")
      .def(
          "compute_inradius",
          [](MeshLoader& self, py::object interior_point, bool use_given_ip) {
            Vec3r ip = Vec3r::Zero();
            if (!interior_point.is_none()) ip = interior_point.cast<Vec3r>();
            const Real inradius = self.ComputeInradius(ip, use_given_ip);
            return py::make_tuple(inradius, ip);
          },
          py::arg("interior_point") = py::none(),
          py::arg("use_given_ip") = false,
          R"doc(
Computes inradius from internally stored points.

Parameters
----------
interior_point : numpy.ndarray, shape (3,), optional
		Interior point used when `use_given_ip` is True.
use_given_ip : bool, default False
		If False, a new interior point is computed.

Returns
-------
tuple[float, numpy.ndarray]
		(inradius, interior_point_used)
)doc")
      .def("npts", &MeshLoader::npts, "Returns number of stored points.");
}
