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
 * @file geometry_concrete.cc
 * @brief Bindings for concrete geometry classes.
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "binding_helpers.h"
#include "dgd/geometry/2d/ellipse.h"
#include "dgd/geometry/2d/polygon.h"
#include "dgd/geometry/2d/rectangle.h"
#include "dgd/geometry/3d/cone.h"
#include "dgd/geometry/3d/cuboid.h"
#include "dgd/geometry/3d/cylinder.h"
#include "dgd/geometry/3d/ellipsoid.h"
#include "dgd/geometry/3d/frustum.h"
#include "dgd/geometry/3d/mesh.h"
#include "dgd/geometry/3d/polytope.h"
#include "dgd/geometry/halfspace.h"
#include "dgd/geometry/xd/capsule.h"
#include "dgd/geometry/xd/sphere.h"

namespace py = pybind11;
using namespace dgd;

using pybind_helpers::MatX2r;
using pybind_helpers::MatX3r;
using pybind_helpers::MatXToVec;
using pybind_helpers::VeciToVecXi;
using pybind_helpers::VecToMatX;
using pybind_helpers::VecXi;
using pybind_helpers::VecXiToVeci;

void bind_geometry_concrete(py::module_& m) {
  // ------------------------------------------------------------------
  // Halfspace
  // ------------------------------------------------------------------
  py::class_<Halfspace<2>>(m, "Halfspace2", "2D half-space.")
      .def(py::init<Real>(), py::arg("margin") = Real(0.0))
      .def_readwrite("margin", &Halfspace<2>::margin, "Safety margin.")
      .def("print_info", &Halfspace<2>::PrintInfo, "Prints half-space info.")
      .def_static("dimension", &Halfspace<2>::dimension, "Returns dimension.");

  py::class_<Halfspace<3>>(m, "Halfspace3", "3D half-space.")
      .def(py::init<Real>(), py::arg("margin") = Real(0.0))
      .def_readwrite("margin", &Halfspace<3>::margin, "Safety margin.")
      .def("print_info", &Halfspace<3>::PrintInfo, "Prints half-space info.")
      .def_static("dimension", &Halfspace<3>::dimension, "Returns dimension.");

  // ------------------------------------------------------------------
  // 2D convex sets
  // ------------------------------------------------------------------
  py::class_<Ellipse, ConvexSet<2>, std::shared_ptr<Ellipse>>(
      m, "Ellipse", "2D axis-aligned ellipse.")
      .def(py::init<Real, Real, Real>(), py::arg("hlx"), py::arg("hly"),
           py::arg("margin") = Real(0.0));

  py::class_<Polygon, ConvexSet<2>, std::shared_ptr<Polygon>>(
      m, "Polygon", "2D convex polygon.")
      .def(py::init([](const MatX2r& vertices, Real inradius, Real margin) {
             return std::make_shared<Polygon>(MatXToVec<2>(vertices), inradius,
                                              margin);
           }),
           py::arg("vertices"), py::arg("inradius"),
           py::arg("margin") = Real(0.0))
      .def(
          "vertices",
          [](const Polygon& self) { return VecToMatX<2>(self.vertices()); },
          "Returns polygon vertices as an (n, 2) array.")
      .def("nvertices", &Polygon::nvertices, "Returns number of vertices.");

  py::class_<Rectangle, ConvexSet<2>, std::shared_ptr<Rectangle>>(
      m, "Rectangle", "2D axis-aligned rectangle.")
      .def(py::init<Real, Real, Real>(), py::arg("hlx"), py::arg("hly"),
           py::arg("margin") = Real(0.0));

  py::class_<Stadium, ConvexSet<2>, std::shared_ptr<Stadium>>(
      m, "Stadium", "2D stadium capsule.")
      .def(py::init<Real, Real, Real>(), py::arg("hlx"), py::arg("radius"),
           py::arg("margin") = Real(0.0));

  py::class_<Circle, ConvexSet<2>, std::shared_ptr<Circle>>(m, "Circle",
                                                            "2D circle.")
      .def(py::init<Real>(), py::arg("radius"));

  // ------------------------------------------------------------------
  // 3D convex sets
  // ------------------------------------------------------------------
  py::class_<Cone, ConvexSet<3>, std::shared_ptr<Cone>>(m, "Cone", "3D cone.")
      .def(py::init<Real, Real, Real>(), py::arg("radius"), py::arg("height"),
           py::arg("margin") = Real(0.0))
      .def("offset", &Cone::offset, "Returns the base z-offset.");

  py::class_<Cuboid, ConvexSet<3>, std::shared_ptr<Cuboid>>(
      m, "Cuboid", "3D axis-aligned cuboid.")
      .def(py::init<Real, Real, Real, Real>(), py::arg("hlx"), py::arg("hly"),
           py::arg("hlz"), py::arg("margin") = Real(0.0));

  py::class_<Cylinder, ConvexSet<3>, std::shared_ptr<Cylinder>>(
      m, "Cylinder", "3D axis-aligned cylinder.")
      .def(py::init<Real, Real, Real>(), py::arg("hlx"), py::arg("radius"),
           py::arg("margin") = Real(0.0));

  py::class_<Ellipsoid, ConvexSet<3>, std::shared_ptr<Ellipsoid>>(
      m, "Ellipsoid", "3D axis-aligned ellipsoid.")
      .def(py::init<Real, Real, Real, Real>(), py::arg("hlx"), py::arg("hly"),
           py::arg("hlz"), py::arg("margin") = Real(0.0));

  py::class_<Frustum, ConvexSet<3>, std::shared_ptr<Frustum>>(m, "Frustum",
                                                              "3D frustum.")
      .def(py::init<Real, Real, Real, Real>(), py::arg("base_radius"),
           py::arg("top_radius"), py::arg("height"),
           py::arg("margin") = Real(0.0))
      .def("offset", &Frustum::offset, "Returns the base z-offset.");

  py::class_<Mesh, ConvexSet<3>, std::shared_ptr<Mesh>>(m, "Mesh", "3D mesh.")
      .def(py::init([](const MatX3r& vertices, const VecXi& graph,
                       Real inradius, Real margin, Real thresh, int guess_level,
                       const std::string& name) {
             return std::make_shared<Mesh>(MatXToVec<3>(vertices),
                                           VecXiToVeci(graph), inradius, margin,
                                           thresh, guess_level, name);
           }),
           py::arg("vertices"), py::arg("graph"), py::arg("inradius"),
           py::arg("margin") = Real(0.0), py::arg("thresh") = Real(0.9),
           py::arg("guess_level") = 1, py::arg("name") = "__Mesh__")
      .def(
          "vertices",
          [](const Mesh& self) { return VecToMatX<3>(self.vertices()); },
          "Returns mesh hull vertices as an (n, 3) array.")
      .def(
          "graph", [](const Mesh& self) { return VeciToVecXi(self.graph()); },
          "Returns mesh graph as an integer array.")
      .def("nvertices", &Mesh::nvertices, "Returns number of vertices.");

  py::class_<Polytope, ConvexSet<3>, std::shared_ptr<Polytope>>(
      m, "Polytope", "3D convex polytope.")
      .def(py::init([](const MatX3r& vertices, Real inradius, Real margin,
                       Real thresh) {
             return std::make_shared<Polytope>(MatXToVec<3>(vertices), inradius,
                                               margin, thresh);
           }),
           py::arg("vertices"), py::arg("inradius"),
           py::arg("margin") = Real(0.0), py::arg("thresh") = Real(0.75))
      .def(
          "vertices",
          [](const Polytope& self) { return VecToMatX<3>(self.vertices()); },
          "Returns polytope vertices as an (n, 3) array.")
      .def("nvertices", &Polytope::nvertices, "Returns number of vertices.");

  py::class_<Capsule, ConvexSet<3>, std::shared_ptr<Capsule>>(m, "Capsule",
                                                              "3D capsule.")
      .def(py::init<Real, Real, Real>(), py::arg("hlx"), py::arg("radius"),
           py::arg("margin") = Real(0.0));

  py::class_<Sphere, ConvexSet<3>, std::shared_ptr<Sphere>>(m, "Sphere",
                                                            "3D sphere.")
      .def(py::init<Real>(), py::arg("radius"));
}
