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
 * @file output.cc
 * @brief Bindings for SolutionStatus, Output, DirectionalDerivative, and
 * TotalDerivative.
 */

#include "dgd/output.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace dgd;

void bind_output(py::module_& m) {
  // ------------------------------------------------------------------
  // SolutionStatus
  // ------------------------------------------------------------------
  py::enum_<SolutionStatus>(m, "SolutionStatus",
                            R"doc(
Solution status at the termination of the growth distance algorithm.

Values
------
Optimal              : converged within the relative tolerance.
MaxIterReached       : maximum iteration count was reached.
CoincidentCenters    : center points of the two sets are too close.
IllConditionedInputs : input sets are ill-conditioned (inradii too small).
)doc")
      .value("Optimal", SolutionStatus::Optimal, "Optimal solution reached.")
      .value("MaxIterReached", SolutionStatus::MaxIterReached,
             "Maximum number of iterations reached.")
      .value("CoincidentCenters", SolutionStatus::CoincidentCenters,
             "Coincident center positions of the convex sets.")
      .value("IllConditionedInputs", SolutionStatus::IllConditionedInputs,
             "Ill-conditioned input sets.");

  // ------------------------------------------------------------------
  // Output<2> / Output2
  // ------------------------------------------------------------------
  py::class_<Output<2>>(m, "Output2",
                        R"doc(
Growth distance algorithm output for 2D convex sets.

Attributes
----------
s1, s2 : numpy.ndarray, shape (2, 2)
    Simplex vertices for each convex set (local frame), corresponding to the
    optimal inner polyhedral approximation.
bc : numpy.ndarray, shape (2,)
    Barycentric coordinates of the optimal simplex.
normal : numpy.ndarray, shape (2,)
    Unit normal vector of the optimal separating hyperplane (world frame),
    pointing from set 1 towards set 2.
z1, z2 : numpy.ndarray, shape (2,)
    Primal optimal solutions in the world frame.
idx_s1, idx_s2 : numpy.ndarray of int, shape (2,)
    Simplex vertex index hints.
growth_dist_ub : float
    Upper bound on the growth distance (primal solution).
growth_dist_lb : float
    Lower bound on the growth distance (dual solution).
iter : int
    Number of solver iterations taken.
status : SolutionStatus
    Solution status at termination.
)doc")
      .def(py::init<>())
      .def_readwrite("s1", &Output<2>::s1, "Simplex vertices for set 1.")
      .def_readwrite("s2", &Output<2>::s2, "Simplex vertices for set 2.")
      .def_readwrite("bc", &Output<2>::bc, "Barycentric coordinates.")
      .def_readwrite("normal", &Output<2>::normal,
                     "Unit normal dual solution (world frame).")
      .def_readwrite("z1", &Output<2>::z1, "Primal solution for set 1.")
      .def_readwrite("z2", &Output<2>::z2, "Primal solution for set 2.")
      .def_readwrite("idx_s1", &Output<2>::idx_s1,
                     "Simplex vertex indices for set 1.")
      .def_readwrite("idx_s2", &Output<2>::idx_s2,
                     "Simplex vertex indices for set 2.")
      .def_readwrite("growth_dist_ub", &Output<2>::growth_dist_ub,
                     "Upper bound on the growth distance.")
      .def_readwrite("growth_dist_lb", &Output<2>::growth_dist_lb,
                     "Lower bound on the growth distance.")
      .def_readwrite("iter", &Output<2>::iter,
                     "Number of solver iterations taken.")
      .def_readwrite("status", &Output<2>::status, "Solution status.");

  // ------------------------------------------------------------------
  // Output<3> / Output3
  // ------------------------------------------------------------------
  py::class_<Output<3>>(m, "Output3",
                        R"doc(
Growth distance algorithm output for 3D convex sets.

Attributes
----------
s1, s2 : numpy.ndarray, shape (3, 3)
    Simplex vertices for each convex set (local frame).
bc : numpy.ndarray, shape (3,)
    Barycentric coordinates of the optimal simplex.
normal : numpy.ndarray, shape (3,)
    Unit normal vector of the optimal separating hyperplane (world frame),
    pointing from set 1 towards set 2.
z1, z2 : numpy.ndarray, shape (3,)
    Primal optimal solutions in the world frame.
idx_s1, idx_s2 : numpy.ndarray of int, shape (3,)
    Simplex vertex index hints.
growth_dist_ub : float
    Upper bound on the growth distance (primal solution).
growth_dist_lb : float
    Lower bound on the growth distance (dual solution).
iter : int
    Number of solver iterations taken.
status : SolutionStatus
    Solution status at termination.
)doc")
      .def(py::init<>())
      .def_readwrite("s1", &Output<3>::s1, "Simplex vertices for set 1.")
      .def_readwrite("s2", &Output<3>::s2, "Simplex vertices for set 2.")
      .def_readwrite("bc", &Output<3>::bc, "Barycentric coordinates.")
      .def_readwrite("normal", &Output<3>::normal,
                     "Unit normal dual solution (world frame).")
      .def_readwrite("z1", &Output<3>::z1, "Primal solution for set 1.")
      .def_readwrite("z2", &Output<3>::z2, "Primal solution for set 2.")
      .def_readwrite("idx_s1", &Output<3>::idx_s1,
                     "Simplex vertex indices for set 1.")
      .def_readwrite("idx_s2", &Output<3>::idx_s2,
                     "Simplex vertex indices for set 2.")
      .def_readwrite("growth_dist_ub", &Output<3>::growth_dist_ub,
                     "Upper bound on the growth distance.")
      .def_readwrite("growth_dist_lb", &Output<3>::growth_dist_lb,
                     "Lower bound on the growth distance.")
      .def_readwrite("iter", &Output<3>::iter,
                     "Number of solver iterations taken.")
      .def_readwrite("status", &Output<3>::status, "Solution status.");

  // ------------------------------------------------------------------
  // DirectionalDerivative<2> / DirectionalDerivative2
  // ------------------------------------------------------------------
  py::class_<DirectionalDerivative<2>>(m, "DirectionalDerivative2",
                                       R"doc(
Directional (Gateaux) derivatives of the growth distance for 2D sets.

Attributes
----------
z_nullspace : numpy.ndarray, shape (2,)
    Orthonormal basis for the primal solution set affine hull (world frame).
n_nullspace : numpy.ndarray, shape (2, 2)
    Orthonormal basis for the dual solution set span (world frame, first
    column is the optimal normal vector).
d_normal : numpy.ndarray, shape (2,)
    Derivative of the dual solution.
d_z1, d_z2 : numpy.ndarray, shape (2,)
    Derivatives of the primal solutions.
d_gd : float
    Derivative of the growth distance.
z_nullity : int
    Dimension of the primal solution set null space.
n_nullity : int
    Dimension of the dual solution set null space.
value_differentiable : bool
    Whether the growth distance value is differentiable.
differentiable : bool
    Whether the full solution is differentiable.
)doc")
      .def(py::init<>())
      .def_readwrite("z_nullspace", &DirectionalDerivative<2>::z_nullspace,
                     "Primal null space basis (world frame).")
      .def_readwrite("n_nullspace", &DirectionalDerivative<2>::n_nullspace,
                     "Dual null space basis (world frame).")
      .def_readwrite("d_normal", &DirectionalDerivative<2>::d_normal,
                     "Derivative of the dual solution.")
      .def_readwrite("d_z1", &DirectionalDerivative<2>::d_z1,
                     "Derivative of the primal solution for set 1.")
      .def_readwrite("d_z2", &DirectionalDerivative<2>::d_z2,
                     "Derivative of the primal solution for set 2.")
      .def_readwrite("d_gd", &DirectionalDerivative<2>::d_gd,
                     "Derivative of the growth distance.")
      .def_readwrite("z_nullity", &DirectionalDerivative<2>::z_nullity,
                     "Dimension of the primal null space.")
      .def_readwrite("n_nullity", &DirectionalDerivative<2>::n_nullity,
                     "Dimension of the dual null space.")
      .def_readwrite("value_differentiable",
                     &DirectionalDerivative<2>::value_differentiable,
                     "Whether the growth distance value is differentiable.")
      .def_readwrite("differentiable",
                     &DirectionalDerivative<2>::differentiable,
                     "Whether the full solution is differentiable.");

  // ------------------------------------------------------------------
  // DirectionalDerivative<3> / DirectionalDerivative3
  // ------------------------------------------------------------------
  py::class_<DirectionalDerivative<3>>(m, "DirectionalDerivative3",
                                       R"doc(
Directional (Gateaux) derivatives of the growth distance for 3D sets.

Attributes
----------
z_nullspace : numpy.ndarray, shape (3, 2)
    Orthonormal basis for the primal solution set affine hull (world frame).
n_nullspace : numpy.ndarray, shape (3, 3)
    Orthonormal basis for the dual solution set span (world frame, first
    column is the optimal normal vector).
d_normal : numpy.ndarray, shape (3,)
    Derivative of the dual solution.
d_z1, d_z2 : numpy.ndarray, shape (3,)
    Derivatives of the primal solutions.
d_gd : float
    Derivative of the growth distance.
z_nullity : int
    Dimension of the primal solution set null space.
n_nullity : int
    Dimension of the dual solution set null space.
value_differentiable : bool
    Whether the growth distance value is differentiable.
differentiable : bool
    Whether the full solution is differentiable.
)doc")
      .def(py::init<>())
      .def_readwrite("z_nullspace", &DirectionalDerivative<3>::z_nullspace,
                     "Primal null space basis (world frame).")
      .def_readwrite("n_nullspace", &DirectionalDerivative<3>::n_nullspace,
                     "Dual null space basis (world frame).")
      .def_readwrite("d_normal", &DirectionalDerivative<3>::d_normal,
                     "Derivative of the dual solution.")
      .def_readwrite("d_z1", &DirectionalDerivative<3>::d_z1,
                     "Derivative of the primal solution for set 1.")
      .def_readwrite("d_z2", &DirectionalDerivative<3>::d_z2,
                     "Derivative of the primal solution for set 2.")
      .def_readwrite("d_gd", &DirectionalDerivative<3>::d_gd,
                     "Derivative of the growth distance.")
      .def_readwrite("z_nullity", &DirectionalDerivative<3>::z_nullity,
                     "Dimension of the primal null space.")
      .def_readwrite("n_nullity", &DirectionalDerivative<3>::n_nullity,
                     "Dimension of the dual null space.")
      .def_readwrite("value_differentiable",
                     &DirectionalDerivative<3>::value_differentiable,
                     "Whether the growth distance value is differentiable.")
      .def_readwrite("differentiable",
                     &DirectionalDerivative<3>::differentiable,
                     "Whether the full solution is differentiable.");

  // ------------------------------------------------------------------
  // TotalDerivative<2> / TotalDerivative2
  // ------------------------------------------------------------------
  py::class_<TotalDerivative<2>>(m, "TotalDerivative2",
                                 R"doc(
Total (Frechet) derivatives of the growth distance for 2-D sets.

All Jacobians are with respect to the SE(2) rigid body motion (3-vectors).

Attributes
----------
d_normal_tf1, d_normal_tf2 : numpy.ndarray, shape (2, 3)
    Jacobians of the dual solution w.r.t. motion of sets 1 and 2.
d_z1_tf1, d_z1_tf2 : numpy.ndarray, shape (2, 3)
    Jacobians of the primal solution z1 w.r.t. motion of sets 1 and 2.
d_z2_tf1, d_z2_tf2 : numpy.ndarray, shape (2, 3)
    Jacobians of the primal solution z2 w.r.t. motion of sets 1 and 2.
d_gd_tf1, d_gd_tf2 : numpy.ndarray, shape (3,)
    Gradients of the growth distance w.r.t. motion of sets 1 and 2.
)doc")
      .def(py::init<>())
      .def_readwrite("d_normal_tf1", &TotalDerivative<2>::d_normal_tf1,
                     "Jacobian of dual solution w.r.t. motion of set 1.")
      .def_readwrite("d_normal_tf2", &TotalDerivative<2>::d_normal_tf2,
                     "Jacobian of dual solution w.r.t. motion of set 2.")
      .def_readwrite("d_z1_tf1", &TotalDerivative<2>::d_z1_tf1,
                     "Jacobian of primal solution z1 w.r.t. motion of set 1.")
      .def_readwrite("d_z1_tf2", &TotalDerivative<2>::d_z1_tf2,
                     "Jacobian of primal solution z1 w.r.t. motion of set 2.")
      .def_readwrite("d_z2_tf1", &TotalDerivative<2>::d_z2_tf1,
                     "Jacobian of primal solution z2 w.r.t. motion of set 1.")
      .def_readwrite("d_z2_tf2", &TotalDerivative<2>::d_z2_tf2,
                     "Jacobian of primal solution z2 w.r.t. motion of set 2.")
      .def_readwrite("d_gd_tf1", &TotalDerivative<2>::d_gd_tf1,
                     "Gradient of growth distance w.r.t. motion of set 1.")
      .def_readwrite("d_gd_tf2", &TotalDerivative<2>::d_gd_tf2,
                     "Gradient of growth distance w.r.t. motion of set 2.");

  // ------------------------------------------------------------------
  // TotalDerivative<3> / TotalDerivative3
  // ------------------------------------------------------------------
  py::class_<TotalDerivative<3>>(m, "TotalDerivative3",
                                 R"doc(
Total (Frechet) derivatives of the growth distance for 3-D sets.

All Jacobians are with respect to the SE(3) rigid body motion (6-vectors).

Attributes
----------
d_normal_tf1, d_normal_tf2 : numpy.ndarray, shape (3, 6)
    Jacobians of the dual solution w.r.t. motion of sets 1 and 2.
d_z1_tf1, d_z1_tf2 : numpy.ndarray, shape (3, 6)
    Jacobians of the primal solution z1 w.r.t. motion of sets 1 and 2.
d_z2_tf1, d_z2_tf2 : numpy.ndarray, shape (3, 6)
    Jacobians of the primal solution z2 w.r.t. motion of sets 1 and 2.
d_gd_tf1, d_gd_tf2 : numpy.ndarray, shape (6,)
    Gradients of the growth distance w.r.t. motion of sets 1 and 2.
)doc")
      .def(py::init<>())
      .def_readwrite("d_normal_tf1", &TotalDerivative<3>::d_normal_tf1,
                     "Jacobian of dual solution w.r.t. motion of set 1.")
      .def_readwrite("d_normal_tf2", &TotalDerivative<3>::d_normal_tf2,
                     "Jacobian of dual solution w.r.t. motion of set 2.")
      .def_readwrite("d_z1_tf1", &TotalDerivative<3>::d_z1_tf1,
                     "Jacobian of primal solution z1 w.r.t. motion of set 1.")
      .def_readwrite("d_z1_tf2", &TotalDerivative<3>::d_z1_tf2,
                     "Jacobian of primal solution z1 w.r.t. motion of set 2.")
      .def_readwrite("d_z2_tf1", &TotalDerivative<3>::d_z2_tf1,
                     "Jacobian of primal solution z2 w.r.t. motion of set 1.")
      .def_readwrite("d_z2_tf2", &TotalDerivative<3>::d_z2_tf2,
                     "Jacobian of primal solution z2 w.r.t. motion of set 2.")
      .def_readwrite("d_gd_tf1", &TotalDerivative<3>::d_gd_tf1,
                     "Gradient of growth distance w.r.t. motion of set 1.")
      .def_readwrite("d_gd_tf2", &TotalDerivative<3>::d_gd_tf2,
                     "Gradient of growth distance w.r.t. motion of set 2.");
}
