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
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @brief Error metrics for growth distance and collision detection solutions.
 */

#ifndef DGD_ERROR_METRICS_H_
#define DGD_ERROR_METRICS_H_

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/output.h"

namespace dgd {

/// @brief Growth distance solution error metrics.
struct SolutionError {
  /**
   * @brief Relative primal-dual gap.
   *
   * The error is given by
   * \f[
   * \text{prim_dual_gap}
   * = \left|\frac{\text{growth_dist_ub}}{\text{growth_dist_lb}} - 1\right|.
   * \f]
   * When the growth distance algorithm converges, this error is less than the
   * specified tolerance.
   *
   * @see Settings.rel_tol
   */
  double prim_dual_gap;

  /**
   * @brief Normalized primal infeasibility error.
   *
   * The primal infeasibility error is given by
   * \f[
   * \text{prim_infeas_err}
   * = | p_1 - p_2 + \text{growth_dist_ub} \cdot (z_1 - p_1 - (z_2 - p_2))|_2.
   * \f]
   * The error is normalized by \f$|p_1 - p_2|_2\f$.
   */
  double prim_infeas_err;

  /**
   * @brief Dual infeasibility error.
   *
   * The growth distance algorithm always ensures dual feasibility, so this
   * error is zero.
   */
  double dual_infeas_err = 0.0;
};

/**
 * @brief Computes the primal-dual relative gap and the normalized primal
 * infeasibility error.
 *
 * @param set1,set2 Convex Sets.
 * @param tf1,tf2   Rigid body transformations for the convex sets.
 * @param out       Growth distance algorithm output.
 */
template <int dim>
SolutionError ComputeSolutionError(const ConvexSet<dim>* set1,
                                   const Transformr<dim>& tf1,
                                   const ConvexSet<dim>* set2,
                                   const Transformr<dim>& tf2,
                                   const Output<dim>& out) {
  SolutionError err{};
  if (out.status == SolutionStatus::CoincidentCenters) {
    err.prim_dual_gap = err.prim_infeas_err = 0.0;
    return err;
  } else if (out.status == SolutionStatus::IllConditionedInputs) {
    err.prim_dual_gap = static_cast<double>(kInf);
    err.prim_infeas_err = 0.0;
    return err;
  }

  const Vecr<dim> p1 = Affine(tf1);
  const Vecr<dim> p2 = Affine(tf2);

  Vecr<dim> sp;
  const Real sv1 =
      set1->SupportFunction(Linear(tf1).transpose() * out.normal, sp);
  const Real sv2 =
      set2->SupportFunction(-Linear(tf2).transpose() * out.normal, sp);
  const Real lb = (p2 - p1).dot(out.normal) / (sv1 + sv2);

  err.prim_dual_gap = static_cast<double>(out.growth_dist_ub / lb) - 1.0;
  err.prim_infeas_err = static_cast<double>(
      (p1 - p2 + out.growth_dist_ub * (out.z1 - p1 - out.z2 + p2)).norm() /
      (p1 - p2).norm());
  return err;
}

/**
 * @brief Asserts the collision status of the two convex sets.
 *
 * @param  set1,set2           Convex Sets.
 * @param  tf1,tf2             Rigid body transformations for the convex sets.
 * @param  out                 Growth distance algorithm output.
 * @param  collision           Output of the collision detection function.
 * @param  max_prim_infeas_err Maximum normalized primal infeasibility error.
 * @return true, if the collision status is correct; false otherwise.
 */
template <int dim>
bool AssertCollisionStatus(const ConvexSet<dim>* set1,
                           const Transformr<dim>& tf1,
                           const ConvexSet<dim>* set2,
                           const Transformr<dim>& tf2, const Output<dim>& out,
                           bool collision,
                           Real max_prim_infeas_err = kSqrtEps) {
  if (out.status == SolutionStatus::CoincidentCenters) {
    return collision;
  } else if ((out.status == SolutionStatus::MaxIterReached) ||
             (out.status == SolutionStatus::IllConditionedInputs)) {
    return false;
  }

  const Vecr<dim> p1 = Affine(tf1);
  const Vecr<dim> p2 = Affine(tf2);

  if (collision) {
    const Real prim_infeas_err =
        (p1 - p2 + out.growth_dist_ub * (out.z1 - p1 - out.z2 + p2)).norm() /
        (p1 - p2).norm();
    return (out.growth_dist_ub <= Real(1.0)) &&
           (prim_infeas_err <= max_prim_infeas_err);
  } else {
    Vecr<dim> sp;
    const Real sv1 =
        set1->SupportFunction(Linear(tf1).transpose() * out.normal, sp);
    const Real sv2 =
        set2->SupportFunction(-Linear(tf2).transpose() * out.normal, sp);
    return (out.growth_dist_lb > Real(1.0)) &&
           (p2.dot(out.normal) - sv2 > p1.dot(out.normal) + sv1);
  }
}

}  // namespace dgd

#endif  // DGD_ERROR_METRICS_H_
