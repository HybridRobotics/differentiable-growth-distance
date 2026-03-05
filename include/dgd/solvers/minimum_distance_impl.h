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
 * @brief Brute force minimum distance algorithm for convex sets.
 */

#ifndef DGD_SOLVERS_MINIMUM_DISTANCE_IMPL_H_
#define DGD_SOLVERS_MINIMUM_DISTANCE_IMPL_H_

#include <utility>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/geometry/halfspace.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/solvers/bundle_scheme_impl.h"
#include "dgd/solvers/solver_types.h"
#include "dgd/utils/transformations.h"

namespace dgd {

namespace detail {

/// @brief Convex set with a mutable margin.
template <int dim, class C>
class PaddedConvexSet : public ConvexSet<dim> {
 public:
  explicit PaddedConvexSet(const C* set, Real margin = Real(0.0));

  Real SupportFunction(
      const Vecr<dim>& n, Vecr<dim>& sp,
      SupportFunctionHint<dim>* hint = nullptr) const final override;

  bool RequireUnitNormal() const final override;

  Real margin() const;

  void set_margin(Real margin);

 private:
  const C* set_;
  Real margin_;
};

template <int dim, class C>
inline PaddedConvexSet<dim, C>::PaddedConvexSet(const C* set, Real margin)
    : ConvexSet<dim>(), set_(set), margin_(margin) {}

template <int dim, class C>
inline Real PaddedConvexSet<dim, C>::SupportFunction(
    const Vecr<dim>& n, Vecr<dim>& sp, SupportFunctionHint<dim>* hint) const {
  const Real sv = set_->SupportFunction(n, sp, hint);
  sp += margin_ * n;
  return sv + margin_;
}

template <int dim, class C>
inline bool PaddedConvexSet<dim, C>::RequireUnitNormal() const {
  return true;
}

template <int dim, class C>
inline Real PaddedConvexSet<dim, C>::margin() const {
  return margin_;
}

template <int dim, class C>
inline void PaddedConvexSet<dim, C>::set_margin(Real margin) {
  this->set_inradius(set_->inradius() + margin);
  margin_ = margin;
}

}  // namespace detail

/// @brief Output for the minimum distance algorithm.
template <int dim>
struct DistanceOutput {
  /// @brief Witness point on set 1.
  Vecr<dim> z1 = Vecr<dim>::Zero();

  /// @brief Witness point on set 2.
  Vecr<dim> z2 = Vecr<dim>::Zero();

  /// @brief Separating vector (pointed towards the second set).
  Vecr<dim> normal = Vecr<dim>::Zero();

  /// @brief Minimum distance between the two sets.
  Real min_dist = Real(0.0);

  /// @brief Number of outer iterations.
  int iter = 0;

  /// @brief Solution status.
  SolutionStatus status = SolutionStatus::MaxIterReached;
};

namespace detail {

/// @brief Sets zero distance output.
template <int dim>
inline void SetZeroDistanceOutput(DistanceOutput<dim>& out) {
  out.z1.setZero();
  out.z2.setZero();
  out.normal.setZero();
  out.min_dist = Real(0.0);
  out.iter = 0;
}

}  // namespace detail

/**
 * @brief Brute force minimum distance algorithm for 2D and 3D compact convex
 * sets.
 */
template <int dim, class C1, class C2>
inline Real MinimumDistanceTpl(const C1* set1, const Transformr<dim>& tf1,
                               const C2* set2, const Transformr<dim>& tf2,
                               const Settings& settings,
                               DistanceOutput<dim>& out, Real tol = kSqrtEps,
                               int max_iter = 100) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible compact set C1");
  static_assert(detail::ConvexSetValidator<dim, C2>::valid,
                "Incompatible compact set C2");

  detail::PaddedConvexSet<dim, C1> set1p(set1);
  Output<dim> o;

  auto gd = [&](Real m) -> std::pair<Real, Real> {
    set1p.set_margin(m);
    GrowthDistanceCpTpl<BcSolverType::LU>(&set1p, tf1, set2, tf2, settings, o);
    o.z1 -= m * o.normal;
    return {o.growth_dist_lb, o.growth_dist_ub};
  };

  // Minimum distance bounds.
  Real dl = Real(0.0), du = (Affine(tf1) - Affine(tf2)).norm();

  [[maybe_unused]] const auto [gd0_lb, gd0_ub] = gd(dl);

  if (o.status == SolutionStatus::IllConditionedInputs) {
    detail::SetZeroDistanceOutput(out);
    out.status = o.status;
    return Real(0.0);
  } else if (gd0_ub <= Real(1.0)) {
    detail::SetZeroDistanceOutput(out);
    out.status = SolutionStatus::Optimal;
    return Real(0.0);
  }

  // Tighten the upper bound.
  du -= du / gd0_ub;

  out.status = SolutionStatus::Optimal;
  int iter = 0;
  while ((du - dl) > tol * std::max(Real(1.0), du)) {
    if (iter >= max_iter) {
      out.status = SolutionStatus::MaxIterReached;
      break;
    }
    ++iter;

    const Real dm = Real(0.5) * (dl + du);
    const auto [gdm_lb, gdm_ub] = gd(dm);
    if (gdm_lb <= Real(1.0)) du = dm;
    if (gdm_ub >= Real(1.0)) dl = dm;
  }

  // Evaluate to obtain accurate witness points.
  gd(Real(0.5) * (dl + du));

  out.z1 = o.z1;
  out.z2 = o.z2;
  out.normal = o.normal;
  out.min_dist = Real(0.5) * (dl + du);
  out.iter = iter;

  return out.min_dist;
}

/**
 * @brief Minimum distance algorithm for for a compact convex set and a
 * half-space.
 */
template <int dim, class C1>
inline Real MinimumDistanceHalfspaceTpl(const C1* set1,
                                        const Transformr<dim>& tf1,
                                        const Halfspace<dim>* set2,
                                        const Transformr<dim>& tf2,
                                        const Settings& /*settings*/,
                                        DistanceOutput<dim>& out) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible compact set C1");

  out.normal = -Linear(tf2).col(dim - 1);
  Vecr<dim> sp1;
  const Real sv1 =
      set1->SupportFunction(Linear(tf1).transpose() * out.normal, sp1);
  const Real d = out.normal.dot(Affine(tf2) - Affine(tf1)) - sv1 - set2->margin;

  if (d <= Real(0.0)) {
    detail::SetZeroDistanceOutput(out);
  } else {
    out.z1 = TransformPoint(tf1, sp1);
    out.z2 = out.z1 + d * out.normal;
    out.min_dist = d;
    out.iter = 0;
  }

  out.status = SolutionStatus::Optimal;
  return out.min_dist;
}

}  // namespace dgd

#endif  // DGD_SOLVERS_MINIMUM_DISTANCE_IMPL_H_
