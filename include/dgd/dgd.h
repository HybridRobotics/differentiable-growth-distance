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
 * @brief Differentiable growth distance algorithm for two and three-dimensional
 * convex sets.
 */

#ifndef DGD_DGD_H_
#define DGD_DGD_H_

#include <type_traits>

#include "dgd/data_types.h"
#include "dgd/dgd_halfspace.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/solvers/bundle_scheme_2d.h"
#include "dgd/solvers/bundle_scheme_3d.h"
#include "dgd/solvers/solver_types.h"

namespace dgd {

/*
 * Growth distance algorithm.
 */

/**
 * @brief Growth distance algorithm for compact convex sets using the cutting
 * plane method.
 *
 * @see GrowthDistance
 */
template <int dim, class C1, class C2, BcSolverType BST = BcSolverType::kCramer>
inline Real GrowthDistanceCp(const C1* set1, const Transformr<dim>& tf1,
                             const C2* set2, const Transformr<dim>& tf2,
                             const Settings& settings, Output<dim>& out,
                             bool warm_start = false) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible set C1");
  static_assert(detail::ConvexSetValidator<dim, C2>::valid,
                "Incompatible set C2");
  return detail::BundleScheme<C1, C2, SolverType::CuttingPlane, BST, false>(
      set1, tf1, set2, tf2, settings, out, warm_start);
}

/**
 * @brief Growth distance algorithm for compact convex sets using the trust
 * region Newton method.
 *
 * @note Cramer's rule results in larger numerical errors for barycentric
 * coordinate computation in 3D.
 *
 * @see GrowthDistance
 */
template <int dim, class C1, class C2>
inline Real GrowthDistanceTrn(const C1* set1, const Transformr<dim>& tf1,
                              const C2* set2, const Transformr<dim>& tf2,
                              const Settings& settings, Output<dim>& out,
                              bool warm_start = false) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible set C1");
  static_assert(detail::ConvexSetValidator<dim, C2>::valid,
                "Incompatible set C2");
  return detail::BundleScheme<C1, C2, SolverType::TrustRegionNewton,
                              BcSolverType::kLU, false>(
      set1, tf1, set2, tf2, settings, out, warm_start);
}

/**
 * @brief Growth distance algorithm for 2D and 3D compact convex sets.
 *
 * @attention When using warm-start, the following properties must be ensured:
 * The same output must be reused from the previous function call;
 * The output must not be used for other pairs of sets in between function
 * calls;
 * The order of the sets must be the same.
 *
 * @note Output from a previous collision detection call can be used to warm
 * start the growth distance algorithm.
 *
 * @param[in]     set1,set2  Compact convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Settings.
 * @param[in,out] out        Output.
 * @param         warm_start Whether to use previous output for warm start.
 * @return        Growth distance lower bound.
 */
template <int dim, class C1, class C2>
inline Real GrowthDistance(const C1* set1, const Transformr<dim>& tf1,
                           const C2* set2, const Transformr<dim>& tf2,
                           const Settings& settings, Output<dim>& out,
                           bool warm_start = false) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible set C1");
  static_assert(detail::ConvexSetValidator<dim, C2>::valid,
                "Incompatible set C2");
  if (set1->IsPolytopic() && set2->IsPolytopic()) {
    return GrowthDistanceCp(set1, tf1, set2, tf2, settings, out, warm_start);
  } else {
    // Trust region Newton method is not used currently.
    return GrowthDistanceCp(set1, tf1, set2, tf2, settings, out, warm_start);
  }
}

/*
 * Collision detection algorithm.
 */

/**
 * @brief Collision detection algorithm for 2D and 3D compact convex sets.
 *
 * Returns true if the centers coincide or if the sets intersect;
 * false if a separating plane has been found or if the maximum number of
 * iterations have been reached.
 *
 * @attention When using warm-start, the following properties must be ensured:
 * The same output must be reused from the previous function call;
 * The output must not be used for other pairs of sets in between function
 * calls;
 * The order of the sets must be the same.
 *
 * @note Output from a previous growth distance call can be used to warm
 * start the collision detection function.
 *
 * @param[in]     set1,set2  Compact convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Settings.
 * @param[in,out] out        Output.
 * @param         warm_start Whether to use previous output for warm start.
 * @return        true, if the sets are colliding; false, otherwise.
 */
template <int dim, class C1, class C2, BcSolverType BST = BcSolverType::kCramer>
bool DetectCollision(const C1* set1, const Transformr<dim>& tf1, const C2* set2,
                     const Transformr<dim>& tf2, const Settings& settings,
                     Output<dim>& out, bool warm_start = false) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible set C1");
  static_assert(detail::ConvexSetValidator<dim, C2>::valid,
                "Incompatible set C2");

  // Collision detection often only takes a few iterations. Using the cutting
  // plane method can be beneficial.
  const Real gd =
      detail::BundleScheme<C1, C2, SolverType::CuttingPlane, BST, true>(
          set1, tf1, set2, tf2, settings, out, warm_start);
  return ((out.status == SolutionStatus::CoincidentCenters) ||
          ((out.status == SolutionStatus::Optimal) && (gd > Real(0.0))));
}

}  // namespace dgd

#endif  // DGD_DGD_H_
