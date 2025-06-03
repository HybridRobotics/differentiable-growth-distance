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
 * @file error_metrics.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-02-18
 * @brief Error metrics for the growth distance and collision detection
 * functions.
 */

#ifndef DGD_ERROR_METRICS_H_
#define DGD_ERROR_METRICS_H_

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/output.h"

namespace dgd {

/**
 * @brief Solution error metrics.
 */
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
   * specified value of rel_tol.
   */
  double prim_dual_gap;

  /**
   * @brief Primal feasibility error.
   *
   * The primal feasibility error is given by
   * \f[
   * \text{prim_feas_err}
   * = | p_1 - p_2 + \text{growth_dist_ub} \cdot (z_1 - p_1 - (z_2 - p_2))|_2,
   * \f]
   * where \f$p_1\f$ and \f$p_2\f$ are the center positions of the
   * convex sets.
   */
  double prim_feas_err;

  /**
   * @brief Dual feasibility error.
   *
   * The growth distance algorithm always ensures dual feasibility, so this
   * error is zero.
   */
  double dual_feas_err{0.0};
};

/**
 * @brief Gets the primal-dual relative gap and the primal feasibility error.
 *
 * @param  set1,set2 Convex Sets.
 * @param  tf1,tf2   Rigid body transformations for the convex sets.
 * @param  out       Solver output.
 * @return Solution error.
 */
template <int dim>
SolutionError GetSolutionError(const ConvexSet<dim>* set1,
                               const Transformr<dim>& tf1,
                               const ConvexSet<dim>* set2,
                               const Transformr<dim>& tf2,
                               const Output<dim>& out) {
  SolutionError err{};
  if (out.status == SolutionStatus::CoincidentCenters) {
    err.prim_feas_err = err.prim_dual_gap = 0.0;
    return err;
  } else if (out.status == SolutionStatus::MaxIterReached) {
    err.prim_feas_err = err.prim_dual_gap = kInf;
    return err;
  }

  const Vecr<dim> p1{tf1.template block<dim, 1>(0, dim)};
  const Vecr<dim> p2{tf2.template block<dim, 1>(0, dim)};
  const Rotationr<dim> rot1{tf1.template block<dim, dim>(0, 0)};
  const Rotationr<dim> rot2{tf2.template block<dim, dim>(0, 0)};

  Vecr<dim> sp;
  const Real sv1{set1->SupportFunction(rot1.transpose() * out.normal, sp)};
  const Real sv2{set2->SupportFunction(-rot2.transpose() * out.normal, sp)};
  const Real lb{(p2 - p1).dot(out.normal) / (sv1 + sv2)};

  err.prim_dual_gap = std::abs(out.growth_dist_ub / lb - 1.0);
  err.prim_feas_err =
      (p1 - p2 + out.growth_dist_ub * (out.z1 - p1 - out.z2 + p2)).norm();
  return err;
}

/**
 * @brief Asserts the collision status of the two convex sets.
 *
 * @param  set1,set2 Convex Sets.
 * @param  tf1,tf2   Rigid body transformations for the convex sets.
 * @param  out       Solver output.
 * @param  collision Output of the collision detection function.
 * @return true, if the collision status is correct; false otherwise.
 */
template <int dim>
bool AssertCollisionStatus(const ConvexSet<dim>* set1,
                           const Transformr<dim>& tf1,
                           const ConvexSet<dim>* set2,
                           const Transformr<dim>& tf2, const Output<dim>& out,
                           bool collision) {
  if (out.status == SolutionStatus::CoincidentCenters) {
    return collision;
  } else if (out.status == SolutionStatus::MaxIterReached) {
    return !collision;
  }

  const Vecr<dim> p1{tf1.template block<dim, 1>(0, dim)};
  const Vecr<dim> p2{tf2.template block<dim, 1>(0, dim)};
  const Rotationr<dim> rot1{tf1.template block<dim, dim>(0, 0)};
  const Rotationr<dim> rot2{tf2.template block<dim, dim>(0, 0)};

  if (collision) {
#ifdef DGD_COMPUTE_COLLISION_INTERSECTION
    const Real prim_feas_err{
        (p1 - p2 + out.growth_dist_ub * (out.z1 - p1 - out.z2 + p2)).norm()};
    return (out.growth_dist_ub <= 1.0) && (prim_feas_err <= kSqrtEps);
#else
    return (out.growth_dist_ub <= 1.0);
#endif
  } else {
    Vecr<dim> sp;
    const Real sv1{set1->SupportFunction(rot1.transpose() * out.normal, sp)};
    const Real sv2{set2->SupportFunction(-rot2.transpose() * out.normal, sp)};
    return (out.growth_dist_lb > 1.0) &&
           (p2.dot(out.normal) - sv2 > p1.dot(out.normal) + sv1);
  }
}

}  // namespace dgd

#endif  // DGD_ERROR_METRICS_H_
