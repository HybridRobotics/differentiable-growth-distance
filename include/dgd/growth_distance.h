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
 * @brief Growth distance algorithm for two and three-dimensional convex sets.
 */

#ifndef DGD_GROWTH_DISTANCE_H_
#define DGD_GROWTH_DISTANCE_H_

#include <type_traits>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/geometry/halfspace.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/solvers/bundle_scheme_2d.h"
#include "dgd/solvers/bundle_scheme_3d.h"
#include "dgd/solvers/solver_options.h"

namespace dgd {

/*
 * Growth distance algorithm.
 */

// Growth distance algorithm implementation for compact convex sets.
template <int dim, class C1, class C2, detail::SolverType S>
inline Real GrowthDistanceImpl(const C1* set1, const Transformr<dim>& tf1,
                               const C2* set2, const Transformr<dim>& tf2,
                               const Settings& settings, Output<dim>& out,
                               bool warm_start) {
  static_assert((dim == 2) || (dim == 3), "dim must be 2 or 3");
  static_assert(std::is_base_of<ConvexSet<dim>, C1>::value &&
                    std::is_base_of<ConvexSet<dim>, C2>::value,
                "The convex sets must inherit from ConvexSet");
  static_assert((C1::dimension() == dim) && (C2::dimension() == dim),
                "Convex sets are not two or three-dimensional");
  return detail::BundleScheme<C1, C2, S, false>(set1, tf1, set2, tf2, settings,
                                                out, warm_start);
}

// Growth distance algorithm implementation for a compact convex set and a
// half-space.
template <int dim, class C1>
inline Real GrowthDistanceImpl(const C1* set1, const Transformr<dim>& tf1,
                               const Halfspace<dim>* set2,
                               const Transformr<dim>& tf2,
                               const Settings& settings, Output<dim>& out,
                               bool warm_start) {
  static_assert((dim == 2) || (dim == 3), "dim must be 2 or 3");
  static_assert(std::is_base_of<ConvexSet<dim>, C1>::value,
                "The convex set must inherit from ConvexSet");
  static_assert(C1::dimension() == dim,
                "Convex set is not two or three-dimensional");

  if (!warm_start) out.hint1_.n_prev = Vecr<dim>::Zero();

  // Check center distance.
  const Vecr<dim> p21 = Affine(tf2) - Affine(tf1);
  const Real cdist = -p21.dot(Linear(tf2).col(dim - 1));
  if (cdist < settings.min_center_dist) {
    out.normal = Vecr<dim>::Zero();
    out.growth_dist_ub = out.growth_dist_lb = 0.0;
    out.z1 = Affine(tf1);
    out.status = SolutionStatus::CoincidentCenters;
    return 0.0;
  }

  out.normal = -Linear(tf2).col(dim - 1);
  // Evaluate the support function.
  Vecr<dim> sp1;
  const Real sv1 = set1->SupportFunction(Linear(tf1).transpose() * out.normal,
                                         sp1, &out.hint1_);

  // Compute the optimal solution.
  out.z1 = Affine(tf1) + Linear(tf1) * sp1;
  out.growth_dist_ub = out.growth_dist_lb = cdist / (sv1 + set2->margin);
  out.iter = 1;
  out.status = SolutionStatus::Optimal;
  return out.growth_dist_lb;
}

template <int dim, class C2>
inline Real GrowthDistanceImpl(const Halfspace<dim>* set1,
                               const Transformr<dim>& tf1, const C2* set2,
                               const Transformr<dim>& tf2,
                               const Settings& settings, Output<dim>& out,
                               bool warm_start) {
  if (warm_start) out.hint1_ = out.hint2_;
  const Real gd = GrowthDistanceImpl<dim, C2>(set2, tf2, set1, tf1, settings,
                                              out, warm_start);
  out.normal = -out.normal;
  if (warm_start) out.hint2_ = out.hint1_;
  out.z2 = out.z1;
  return gd;
}

/**
 * @name Growth distance algorithm for convex sets
 * @brief Growth distance algorithm for two and three-dimensional convex sets.
 *
 * @attention When using warm-start with two compact convex sets, the following
 * properties must be ensured:
 * The same output must be reused from the previous function call;
 * The output must not be used for other pairs of sets in between function
 * calls;
 * The order of the sets must be the same.
 *
 * @attention When one of the sets is a half-space, out.s1, out.s2, out.bc, and
 * out.z2 are not set.
 *
 * @note When one of the sets is a half-space, warm start is only used to
 * accelerate the support function computation.
 *
 * @note Output from a previous collision detection call can be used to warm
 * start the growth distance algorithm.
 *
 * @param[in]     set1,set2  Convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Settings.
 * @param[in,out] out        Output.
 * @param         warm_start Whether to use previous output for warm start.
 * @return        Growth distance lower bound.
 */
///@{
template <class C1, class C2>
inline Real GrowthDistance(const C1* set1, const Transform2r& tf1,
                           const C2* set2, const Transform2r& tf2,
                           const Settings& settings, Output2& out,
                           bool warm_start = false) {
  if constexpr (std::is_same<Halfspace<2>, C1>::value) {
    return GrowthDistanceImpl<2, C2>(set1, tf1, set2, tf2, settings, out,
                                     warm_start);
  } else if constexpr (std::is_same<Halfspace<2>, C2>::value) {
    return GrowthDistanceImpl<2, C1>(set1, tf1, set2, tf2, settings, out,
                                     warm_start);
  } else {
    return GrowthDistanceImpl<2, C1, C2, detail::SolverType::TrustRegionNewton>(
        set1, tf1, set2, tf2, settings, out, warm_start);
  }
}

template <class C1, class C2>
inline Real GrowthDistance(const C1* set1, const Transform3r& tf1,
                           const C2* set2, const Transform3r& tf2,
                           const Settings& settings, Output3& out,
                           bool warm_start = false) {
  if constexpr (std::is_same<Halfspace<3>, C1>::value) {
    return GrowthDistanceImpl<3, C2>(set1, tf1, set2, tf2, settings, out,
                                     warm_start);
  } else if constexpr (std::is_same<Halfspace<3>, C2>::value) {
    return GrowthDistanceImpl<3, C1>(set1, tf1, set2, tf2, settings, out,
                                     warm_start);
  } else {
    if (set1->IsPolytopic() || set2->IsPolytopic()) {
      return GrowthDistanceImpl<3, C1, C2, detail::SolverType::CuttingPlane>(
          set1, tf1, set2, tf2, settings, out, warm_start);
    } else {
      return GrowthDistanceImpl<3, C1, C2,
                                detail::SolverType::TrustRegionNewton>(
          set1, tf1, set2, tf2, settings, out, warm_start);
    }
  }
}
///@}

/*
 * Collision detection algorithm.
 */

// Collision detection algorithm implementation for compact convex sets.
template <int dim, class C1, class C2, detail::SolverType S>
inline bool DetectCollisionImpl(const C1* set1, const Transformr<dim>& tf1,
                                const C2* set2, const Transformr<dim>& tf2,
                                const Settings& settings, Output<dim>& out,
                                bool warm_start) {
  static_assert((dim == 2) || (dim == 3), "dim must be 2 or 3");
  static_assert((C1::dimension() == dim) && (C2::dimension() == dim),
                "Convex sets are not two or three-dimensional");
  const Real gd = detail::BundleScheme<C1, C2, S, true>(
      set1, tf1, set2, tf2, settings, out, warm_start);
  return ((out.status == SolutionStatus::CoincidentCenters) ||
          ((out.status == SolutionStatus::Optimal) && (gd > Real(0.0))));
}

/**
 * @brief Collision detection algorithm for convex sets.
 *
 * Returns true if the centers coincide or if the sets intersect;
 * false if a separating plane has been found or if the maximum number of
 * iterations have been reached.
 *
 * @attention When using warm-start with two compact convex sets, the following
 * properties must be ensured:
 * The same output must be reused from the previous function call;
 * The output must not be used for other pairs of sets in between function
 * calls;
 * The order of the sets must be the same.
 *
 * @attention When one of the sets is a half-space, out.s1, out.s2, out.bc, and
 * out.z2 are not set.
 *
 * @note When one of the sets is a half-space, warm start is only used to
 * accelerate the support function computation.
 *
 * @note Output from a previous growth distance call can be used to warm
 * start the collision detection function.
 *
 * @param[in]     set1,set2  Convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Settings.
 * @param[in,out] out        Output.
 * @param         warm_start Whether to use previous output for warm start.
 * @return        true, if the sets are colliding; false, otherwise.
 */
template <int dim, class C1, class C2>
bool DetectCollision(const C1* set1, const Transformr<dim>& tf1, const C2* set2,
                     const Transformr<dim>& tf2, const Settings& settings,
                     Output<dim>& out, bool warm_start = false) {
  if constexpr (std::is_same<Halfspace<3>, C1>::value) {
    const Real gd = GrowthDistanceImpl<3, C2>(set1, tf1, set2, tf2, settings,
                                              out, warm_start);
    return ((out.status == SolutionStatus::CoincidentCenters) ||
            (gd <= Real(1.0)));
  } else if constexpr (std::is_same<Halfspace<3>, C2>::value) {
    const Real gd = GrowthDistanceImpl<3, C1>(set1, tf1, set2, tf2, settings,
                                              out, warm_start);
    return ((out.status == SolutionStatus::CoincidentCenters) ||
            (gd <= Real(1.0)));
  } else {
    // Collision detection often only takes a few iterations. Using Trust region
    // Newton can be detrimental.
    return DetectCollisionImpl<dim, C1, C2, detail::SolverType::CuttingPlane>(
        set1, tf1, set2, tf2, settings, out, warm_start);
  }
}

}  // namespace dgd

#endif  // DGD_GROWTH_DISTANCE_H_
