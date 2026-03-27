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
 * @file types.cc
 * @brief Bindings for scalar constants and the KinematicState struct from
 * data_types.h.
 *
 * Eigen matrix/vector types (Vecr, Matr, Transformr, ...) are automatically
 * converted to/from NumPy arrays by <pybind11/eigen.h> and do not need
 * explicit class bindings.
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include <sstream>

#include "dgd/data_types.h"

namespace py = pybind11;
using namespace dgd;

namespace {

template <int dim>
inline std::string VectorToLiteral(const Vecr<dim>& v) {
  std::ostringstream oss;
  oss << "[" << v(0);
  for (int i = 1; i < dim; ++i) {
    oss << ", " << v(i);
  }
  oss << "]";
  return oss.str();
}

template <int row, int col>
inline std::string MatrixToLiteral(const Matr<row, col>& m) {
  std::ostringstream oss;
  oss << "[";
  for (int i = 0; i < row; ++i) {
    if (i > 0) oss << ", ";
    oss << "[" << m(i, 0);
    for (int j = 1; j < col; ++j) {
      oss << ", " << m(i, j);
    }
    oss << "]";
  }
  oss << "]";
  return oss.str();
}

}  // namespace

void bind_types(py::module_& m) {
  // ------------------------------------------------------------------
  // Scalar constants
  // ------------------------------------------------------------------
  m.attr("INF") = kInf;
  m.attr("EPS") = kEps;
  m.attr("SQRT_EPS") = kSqrtEps;
  m.attr("PI") = kPi;

  // ------------------------------------------------------------------
  // KinematicState<2>
  // ------------------------------------------------------------------
  py::class_<KinematicState<2>>(m, "KinematicState2",
                                R"doc(
Kinematic state of a 2D rigid body.

Attributes
----------
tf : numpy.ndarray, shape (3, 3)
    Rigid body transformation (rotation + translation) in homogeneous
    coordinates.  Initialized to identity.
tw : numpy.ndarray, shape (3,)
    Rigid body twist ``(v_x, v_y, omega_z)``.  Initialized to zero.
)doc")
      .def(py::init<>())
      .def(py::init([](const Transformr<2>& tf, const Twistr<2>& tw) {
             KinematicState<2> s;
             s.tf = tf;
             s.tw = tw;
             return s;
           }),
           py::arg("tf"), py::arg("tw"))
      .def_property(
          "tf", [](KinematicState<2>& s) -> Transformr<2>& { return s.tf; },
          [](KinematicState<2>& s, const Transformr<2>& v) { s.tf = v; },
          py::return_value_policy::reference_internal)
      .def_property(
          "tw", [](KinematicState<2>& s) -> Twistr<2>& { return s.tw; },
          [](KinematicState<2>& s, const Twistr<2>& v) { s.tw = v; },
          py::return_value_policy::reference_internal)
      .def("__repr__", [](const KinematicState<2>& s) {
        return "dgd.KinematicState2(tf=" + MatrixToLiteral(s.tf) +
               ", tw=" + VectorToLiteral(s.tw) + ")";
      });

  // ------------------------------------------------------------------
  // KinematicState<3>
  // ------------------------------------------------------------------
  py::class_<KinematicState<3>>(m, "KinematicState3",
                                R"doc(
Kinematic state of a 3D rigid body.

Attributes
----------
tf : numpy.ndarray, shape (4, 4)
    Rigid body transformation (rotation + translation) in homogeneous
    coordinates.  Initialized to identity.
tw : numpy.ndarray, shape (6,)
    Rigid body twist ``(v_x, v_y, v_z, omega_x, omega_y, omega_z)``.
    Initialized to zero.
)doc")
      .def(py::init<>())
      .def(py::init([](const Transformr<3>& tf, const Twistr<3>& tw) {
             KinematicState<3> s;
             s.tf = tf;
             s.tw = tw;
             return s;
           }),
           py::arg("tf"), py::arg("tw"))
      .def_property(
          "tf", [](KinematicState<3>& s) -> Transformr<3>& { return s.tf; },
          [](KinematicState<3>& s, const Transformr<3>& v) { s.tf = v; },
          py::return_value_policy::reference_internal)
      .def_property(
          "tw", [](KinematicState<3>& s) -> Twistr<3>& { return s.tw; },
          [](KinematicState<3>& s, const Twistr<3>& v) { s.tw = v; },
          py::return_value_policy::reference_internal)
      .def("__repr__", [](const KinematicState<3>& s) {
        return "dgd.KinematicState3(tf=" + MatrixToLiteral(s.tf) +
               ", tw=" + VectorToLiteral(s.tw) + ")";
      });
}
