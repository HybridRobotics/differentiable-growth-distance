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
 * @file algorithms.cc
 * @brief Bindings for core growth distance and derivative algorithms.
 */

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "dgd/dgd.h"

namespace py = pybind11;
using namespace dgd;

namespace {

template <int dim>
void bind_algorithms_dim(py::module_& m) {
  const std::string suffix = std::to_string(dim);

  m.def(
      "growth_distance",
      [](const ConvexSet<dim>& set1, const Transformr<dim>& tf1,
         const ConvexSet<dim>& set2, const Transformr<dim>& tf2,
         const Settings& settings, Output<dim>& out, bool warm_start) {
        return GrowthDistance(&set1, tf1, &set2, tf2, settings, out,
                              warm_start);
      },
      py::arg("set1"), py::arg("tf1"), py::arg("set2"), py::arg("tf2"),
      py::arg("settings"), py::arg("out"), py::arg("warm_start") = false,
      ("Growth distance for ConvexSet" + suffix + " x ConvexSet" + suffix + ".")
          .c_str());

  m.def(
      "growth_distance",
      [](const ConvexSet<dim>& set1, const Transformr<dim>& tf1,
         const Halfspace<dim>& set2, const Transformr<dim>& tf2,
         const Settings& settings, Output<dim>& out, bool warm_start) {
        return GrowthDistance(&set1, tf1, &set2, tf2, settings, out,
                              warm_start);
      },
      py::arg("set1"), py::arg("tf1"), py::arg("set2"), py::arg("tf2"),
      py::arg("settings"), py::arg("out"), py::arg("warm_start") = false,
      ("Growth distance for ConvexSet" + suffix + " x Halfspace" + suffix + ".")
          .c_str());

  m.def(
      "growth_distance_cp",
      [](const ConvexSet<dim>& set1, const Transformr<dim>& tf1,
         const ConvexSet<dim>& set2, const Transformr<dim>& tf2,
         const Settings& settings, Output<dim>& out, bool warm_start) {
        return GrowthDistanceCp(&set1, tf1, &set2, tf2, settings, out,
                                warm_start);
      },
      py::arg("set1"), py::arg("tf1"), py::arg("set2"), py::arg("tf2"),
      py::arg("settings"), py::arg("out"), py::arg("warm_start") = false,
      ("Growth distance using cutting-plane method for ConvexSet" + suffix +
       " x ConvexSet" + suffix + ".")
          .c_str());

  m.def(
      "growth_distance_trn",
      [](const ConvexSet<dim>& set1, const Transformr<dim>& tf1,
         const ConvexSet<dim>& set2, const Transformr<dim>& tf2,
         const Settings& settings, Output<dim>& out, bool warm_start) {
        return GrowthDistanceTrn(&set1, tf1, &set2, tf2, settings, out,
                                 warm_start);
      },
      py::arg("set1"), py::arg("tf1"), py::arg("set2"), py::arg("tf2"),
      py::arg("settings"), py::arg("out"), py::arg("warm_start") = false,
      ("Growth distance using trust-region Newton method for ConvexSet" +
       suffix + " x ConvexSet" + suffix + ".")
          .c_str());

  m.def(
      "detect_collision",
      [](const ConvexSet<dim>& set1, const Transformr<dim>& tf1,
         const ConvexSet<dim>& set2, const Transformr<dim>& tf2,
         const Settings& settings, Output<dim>& out, bool warm_start) {
        return DetectCollision(&set1, tf1, &set2, tf2, settings, out,
                               warm_start);
      },
      py::arg("set1"), py::arg("tf1"), py::arg("set2"), py::arg("tf2"),
      py::arg("settings"), py::arg("out"), py::arg("warm_start") = false,
      ("Collision detection for ConvexSet" + suffix + " x ConvexSet" + suffix +
       ".")
          .c_str());

  m.def(
      "detect_collision",
      [](const ConvexSet<dim>& set1, const Transformr<dim>& tf1,
         const Halfspace<dim>& set2, const Transformr<dim>& tf2,
         const Settings& settings, Output<dim>& out, bool warm_start) {
        return DetectCollision(&set1, tf1, &set2, tf2, settings, out,
                               warm_start);
      },
      py::arg("set1"), py::arg("tf1"), py::arg("set2"), py::arg("tf2"),
      py::arg("settings"), py::arg("out"), py::arg("warm_start") = false,
      ("Collision detection for ConvexSet" + suffix + " x Halfspace" + suffix +
       ".")
          .c_str());

  m.def(
      "compute_kkt_nullspace",
      [](const ConvexSet<dim>& set1, const Transformr<dim>& tf1,
         const ConvexSet<dim>& set2, const Transformr<dim>& tf2,
         const Settings& settings, const Output<dim>& out,
         DirectionalDerivative<dim>& dd) {
        return ComputeKktNullspace(&set1, tf1, &set2, tf2, settings, out, dd);
      },
      py::arg("set1"), py::arg("tf1"), py::arg("set2"), py::arg("tf2"),
      py::arg("settings"), py::arg("out"), py::arg("dd"),
      ("Computes KKT null space for ConvexSet" + suffix + " x ConvexSet" +
       suffix + ".")
          .c_str());

  m.def(
      "compute_kkt_nullspace",
      [](const ConvexSet<dim>& set1, const Transformr<dim>& tf1,
         const Halfspace<dim>& set2, const Transformr<dim>& tf2,
         const Settings& settings, const Output<dim>& out,
         DirectionalDerivative<dim>& dd) {
        return ComputeKktNullspace(&set1, tf1, &set2, tf2, settings, out, dd);
      },
      py::arg("set1"), py::arg("tf1"), py::arg("set2"), py::arg("tf2"),
      py::arg("settings"), py::arg("out"), py::arg("dd"),
      ("Computes KKT null space for ConvexSet" + suffix + " x Halfspace" +
       suffix + ".")
          .c_str());

  m.def(
      "gd_derivative",
      [](const KinematicState<dim>& state1, const KinematicState<dim>& state2,
         const Settings& settings, const Output<dim>& out,
         DirectionalDerivative<dim>* dd) {
        return GdDerivative(state1, state2, settings, out, dd);
      },
      py::arg("state1"), py::arg("state2"), py::arg("settings"), py::arg("out"),
      py::arg("dd") = nullptr,
      ("Directional derivative of growth distance for " + suffix + "D convex " +
       "sets.")
          .c_str());

  m.def(
      "gd_gradient",
      [](const Transformr<dim>& tf1, const Transformr<dim>& tf2,
         const Settings& settings, const Output<dim>& out,
         TotalDerivative<dim>& td) { GdGradient(tf1, tf2, settings, out, td); },
      py::arg("tf1"), py::arg("tf2"), py::arg("settings"), py::arg("out"),
      py::arg("td"),
      ("Gradient of growth distance for " + suffix + "D convex sets.").c_str());

  m.def(
      "factorize_kkt_system",
      [](const ConvexSet<dim>& set1, const Transformr<dim>& tf1,
         const ConvexSet<dim>& set2, const Transformr<dim>& tf2,
         const Settings& settings, const Output<dim>& out,
         DirectionalDerivative<dim>& dd) {
        return FactorizeKktSystem(&set1, tf1, &set2, tf2, settings, out, dd);
      },
      py::arg("set1"), py::arg("tf1"), py::arg("set2"), py::arg("tf2"),
      py::arg("settings"), py::arg("out"), py::arg("dd"),
      ("Factorizes KKT system for ConvexSet" + suffix + " x ConvexSet" +
       suffix + ".")
          .c_str());

  m.def(
      "factorize_kkt_system",
      [](const ConvexSet<dim>& set1, const Transformr<dim>& tf1,
         const Halfspace<dim>& set2, const Transformr<dim>& tf2,
         const Settings& settings, const Output<dim>& out,
         DirectionalDerivative<dim>& dd) {
        return FactorizeKktSystem(&set1, tf1, &set2, tf2, settings, out, dd);
      },
      py::arg("set1"), py::arg("tf1"), py::arg("set2"), py::arg("tf2"),
      py::arg("settings"), py::arg("out"), py::arg("dd"),
      ("Factorizes KKT system for ConvexSet" + suffix + " x Halfspace" +
       suffix + ".")
          .c_str());

  m.def(
      "gd_solution_derivative",
      [](const KinematicState<dim>& state1, const KinematicState<dim>& state2,
         const Settings& settings, const Output<dim>& out,
         DirectionalDerivative<dim>& dd) {
        GdSolutionDerivative(state1, state2, settings, out, dd);
      },
      py::arg("state1"), py::arg("state2"), py::arg("settings"), py::arg("out"),
      py::arg("dd"),
      ("Directional derivative of growth distance optimal solution for " +
       suffix + "D convex sets.")
          .c_str());

  m.def(
      "gd_jacobian",
      [](const Transformr<dim>& tf1, const Transformr<dim>& tf2,
         const Settings& settings, const Output<dim>& out,
         const DirectionalDerivative<dim>& dd, TotalDerivative<dim>& td) {
        GdJacobian(tf1, tf2, settings, out, dd, td);
      },
      py::arg("tf1"), py::arg("tf2"), py::arg("settings"), py::arg("out"),
      py::arg("dd"), py::arg("td"),
      ("Jacobian of growth distance optimal solution for " + suffix + "D " +
       "convex sets.")
          .c_str());
}

}  // namespace

void bind_algorithms(py::module_& m) {
  bind_algorithms_dim<2>(m);
  bind_algorithms_dim<3>(m);
}
