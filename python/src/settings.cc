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
 * @file settings.cc
 * @brief Bindings for WarmStartType, TwistFrame, and Settings.
 */

#include "dgd/settings.h"

#include <pybind11/pybind11.h>

#include <sstream>

#include "dgd/solvers/solver_types.h"

namespace py = pybind11;
using namespace dgd;

void bind_settings(py::module_& m) {
  // ------------------------------------------------------------------
  // WarmStartType
  // ------------------------------------------------------------------
  py::enum_<WarmStartType>(m, "WarmStartType",
                           R"doc(
Warm start type for the growth distance algorithm.

Values
------
Primal : initialize from the previous primal solution.
Dual   : initialize from the previous dual solution.
)doc")
      .value("Primal", WarmStartType::Primal, "Primal solution warm start.")
      .value("Dual", WarmStartType::Dual, "Dual solution warm start.");

  // ------------------------------------------------------------------
  // TwistFrame
  // ------------------------------------------------------------------
  py::enum_<TwistFrame>(m, "TwistFrame",
                        R"doc(
Rigid body twist reference frame.

Values
------
Spatial : spatial twist in the world frame.
Hybrid  : hybrid twist in the world frame (translational velocity is the
          velocity of the origin of the local frame).
Body    : body twist in the local frame.
)doc")
      .value("Spatial", TwistFrame::Spatial,
             "Spatial twist in the world frame.")
      .value("Hybrid", TwistFrame::Hybrid, "Hybrid twist in the world frame.")
      .value("Body", TwistFrame::Body, "Body twist in the local frame.");

  // ------------------------------------------------------------------
  // Settings
  // ------------------------------------------------------------------
  py::class_<Settings>(m, "Settings",
                       R"doc(
Settings for the differentiable growth distance algorithm.

All fields default to well-chosen values and can be overridden in-place.

Attributes
----------
min_center_dist : float
    Minimum distance between the center points of the convex sets.
    Growth distance is set to zero when the centers are closer than this.
rel_tol : float
    Relative tolerance for the primal–dual gap
    (convergence criterion: ``lb <= ub <= rel_tol * lb``).
nullspace_tol : float
    Tolerance for primal/dual null-space computations in 3D.
jac_tol : float
    Tolerance for the solution derivative computation.
max_iter : int
    Maximum number of solver iterations.
ws_type : WarmStartType
    Warm start type.
twist_frame : TwistFrame
    Reference frame for input twist vectors.
)doc")
      .def(py::init<>())
      .def_readwrite("min_center_dist", &Settings::min_center_dist,
                     "Minimum distance between center points of the sets.")
      .def_readwrite("rel_tol", &Settings::rel_tol,
                     "Relative primal–dual gap tolerance.")
      .def_readwrite("nullspace_tol", &Settings::nullspace_tol,
                     "Tolerance for null-space computations (3D).")
      .def_readwrite("jac_tol", &Settings::jac_tol,
                     "Tolerance for solution derivative computation.")
      .def_readwrite("max_iter", &Settings::max_iter,
                     "Maximum number of solver iterations.")
      .def_readwrite("ws_type", &Settings::ws_type, "Warm start type.")
      .def_readwrite("twist_frame", &Settings::twist_frame,
                     "Reference frame for input twist vectors.")
      .def("__repr__", [](const Settings& s) {
        std::ostringstream oss;
        oss << "dgd.Settings(min_center_dist=" << s.min_center_dist
            << ", rel_tol=" << s.rel_tol
            << ", nullspace_tol=" << s.nullspace_tol
            << ", jac_tol=" << s.jac_tol << ", max_iter=" << s.max_iter
            << ", ws_type=WarmStartType." << WarmStartName(s.ws_type)
            << ", twist_frame=TwistFrame." << TwistFrameName(s.twist_frame)
            << ")";
        return oss.str();
      });
}
