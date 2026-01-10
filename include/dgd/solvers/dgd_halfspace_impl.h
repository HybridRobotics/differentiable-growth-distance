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
 * @brief Differentiable growth distance algorithm implementations for a compact
 * convex set and a half-space.
 */

#ifndef DGD_SOLVERS_DGD_HALFSPACE_IMPL_H_
#define DGD_SOLVERS_DGD_HALFSPACE_IMPL_H_

#include <type_traits>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/geometry/halfspace.h"
#include "dgd/output.h"
#include "dgd/settings.h"

namespace dgd {

/*
 * Growth distance algorithm.
 */

/// @brief Growth distance algorithm for a compact convex set and a half-space.
template <int dim, class C1>
inline Real GrowthDistanceHalfspaceTpl(
    const C1* set1, const Transformr<dim>& tf1, const Halfspace<dim>* set2,
    const Transformr<dim>& tf2, const Settings& settings, Output<dim>& out,
    bool warm_start = false) {
  static_assert(detail::ConvexSetValidator<dim, C1, false>::valid,
                "Incompatible compact set C1");

  if (!warm_start) out.hint1_.n_prev = Vecr<dim>::Zero();

  // Check center distance.
  const Vecr<dim> p21 = Affine(tf2) - Affine(tf1);
  const Real cdist = -p21.dot(Linear(tf2).col(dim - 1));
  if (cdist < settings.min_center_dist) {
    out.normal = Vecr<dim>::Zero();
    out.growth_dist_ub = out.growth_dist_lb = Real(0.0);
    out.z1 = Affine(tf1);
    out.status = SolutionStatus::CoincidentCenters;
    return Real(0.0);
  }

  // Check (lower bound of) the Minkowski difference set inradius.
  if (set1->inradius() + set2->margin <= kSqrtEps) {
    out.growth_dist_ub = kInf;
    out.growth_dist_lb = Real(0.0);
    out.status = SolutionStatus::IllConditionedInputs;
    return Real(0.0);
  }

  out.normal = -Linear(tf2).col(dim - 1);
  // Evaluate the support function.
  Vecr<dim> sp1;
  const Real sv1 = set1->SupportFunction(Linear(tf1).transpose() * out.normal,
                                         sp1, &out.hint1_);

  // Compute the optimal solution.
  out.z1.noalias() = Affine(tf1) + Linear(tf1) * sp1;
  out.growth_dist_ub = out.growth_dist_lb = cdist / (sv1 + set2->margin);
  out.iter = 1;
  out.status = SolutionStatus::Optimal;
  return out.growth_dist_lb;
}

/*
 * Collision detection algorithm.
 */

/**
 * @brief Collision detection algorithm for a compact convex set and a
 * half-space.
 */
template <int dim, class C1>
inline bool DetectCollisionHalfspaceTpl(
    const C1* set1, const Transformr<dim>& tf1, const Halfspace<dim>* set2,
    const Transformr<dim>& tf2, const Settings& settings, Output<dim>& out,
    bool warm_start = false) {
  static_assert(detail::ConvexSetValidator<dim, C1, false>::valid,
                "Incompatible compact set C1");
  const Real gd = GrowthDistanceHalfspaceTpl<dim, C1>(
      set1, tf1, set2, tf2, settings, out, warm_start);
  return ((out.status == SolutionStatus::CoincidentCenters) ||
          (gd <= Real(1.0)));
}

}  // namespace dgd

#endif  // DGD_SOLVERS_DGD_HALFSPACE_IMPL_H_
