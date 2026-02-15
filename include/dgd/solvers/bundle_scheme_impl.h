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
 * @brief Growth distance algorithm implementations for two compact convex sets.
 * @warning This is a heavy header. It may significantly increase the
 * compilation time. Use only when template implementations are needed.
 */

#ifndef DGD_SOLVERS_BUNDLE_SCHEME_IMPL_H_
#define DGD_SOLVERS_BUNDLE_SCHEME_IMPL_H_

#include "dgd/data_types.h"
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
 */
template <BcSolverType BST = BcSolverType::Cramer, int dim, class C1, class C2>
inline Real GrowthDistanceCpTpl(const C1* set1, const Transformr<dim>& tf1,
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
 */
template <int dim, class C1, class C2>
inline Real GrowthDistanceTrnTpl(const C1* set1, const Transformr<dim>& tf1,
                                 const C2* set2, const Transformr<dim>& tf2,
                                 const Settings& settings, Output<dim>& out,
                                 bool warm_start = false) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible set C1");
  static_assert(detail::ConvexSetValidator<dim, C2>::valid,
                "Incompatible set C2");
  return detail::BundleScheme<C1, C2, SolverType::TrustRegionNewton,
                              BcSolverType::LU, false>(
      set1, tf1, set2, tf2, settings, out, warm_start);
}

/*
 * Collision detection algorithm.
 */

/// @brief Collision detection algorithm for 2D and 3D compact convex sets.
template <BcSolverType BST = BcSolverType::Cramer, int dim, class C1, class C2>
inline bool DetectCollisionTpl(const C1* set1, const Transformr<dim>& tf1,
                               const C2* set2, const Transformr<dim>& tf2,
                               const Settings& settings, Output<dim>& out,
                               bool warm_start = false) {
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

#endif  // DGD_SOLVERS_BUNDLE_SCHEME_IMPL_H_
