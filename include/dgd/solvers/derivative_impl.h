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
 * @brief Differentiable growth distance algorithm implementations for two
 * compact convex sets.
 */

#ifndef DGD_SOLVERS_DERIVATIVE_IMPL_H_
#define DGD_SOLVERS_DERIVATIVE_IMPL_H_

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/solvers/solver_utils.h"

namespace dgd {

/**
 * @brief KKT solution set null space algorithm for 2D and 3D compact convex
 * sets.
 */
template <int dim, class C1, class C2>
inline int ComputeKktNullspaceTpl(const C1* set1, const Transformr<dim>& tf1,
                                  const C2* set2, const Transformr<dim>& tf2,
                                  const Settings& settings,
                                  const OutputBundle<dim>& bundle) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible set C1");
  static_assert(detail::ConvexSetValidator<dim, C2>::valid,
                "Incompatible set C2");

  const auto& out = bundle.output;
  const auto& dd = bundle.dir_derivative;

  if ((!out) || (!dd)) return 0;

  if (out->status != SolutionStatus::Optimal) {
    return detail::SetZeroKktNullspace(*dd);
  }

  const Vecr<dim> n = out->normal.normalized();

  NormalPair<dim> zn1, zn2;
  zn1.z.noalias() = out->s1 * out->bc;
  zn1.n.noalias() = Linear(tf1).transpose() * n;
  zn2.z.noalias() = out->s2 * out->bc;
  zn2.n.noalias() = -Linear(tf2).transpose() * n;

  BasePointHint<dim> hint1, hint2;
  hint1.s = &out->s1;
  hint1.bc = &out->bc;
  hint1.sfh = &out->hint1_;
  hint2.s = &out->s2;
  hint2.bc = &out->bc;
  hint2.sfh = &out->hint2_;

  SupportPatchHull<dim> sph1, sph2;
  NormalConeSpan<dim> ncs1, ncs2;

  set1->ComputeLocalGeometry(zn1, sph1, ncs1, &hint1);
  set2->ComputeLocalGeometry(zn2, sph2, ncs2, &hint2);

  if constexpr (dim == 2) {
    // Compute primal solution set null space.
    if ((sph1.aff_dim == 0) || (sph2.aff_dim == 0)) {
      dd->z_nullspace.setZero();
      dd->z_nullity = 0;
    } else {
      dd->z_nullspace.col(0) = Vec2r(n(1), -n(0));
      dd->z_nullity = 1;
    }

    // Compute dual solution set null space.
    dd->n_nullspace.col(0) = n;
    if ((ncs1.span_dim == 1) || (ncs2.span_dim == 1)) {
      dd->n_nullspace.col(1).setZero();
      dd->n_nullity = 1;
    } else {
      dd->n_nullspace.col(1) = Vec2r(n(1), -n(0));
      dd->n_nullity = 2;
    }
  } else {  // dim = 3
    const Vec3r m =
        (std::abs(n(0)) > Real(0.9)) ? Vec3r::UnitY() : Vec3r::UnitX();

    // Transform the basis vectors to the world frame.
    if (sph1.aff_dim == 1) sph1.basis = Linear(tf1) * sph1.basis;
    if (sph2.aff_dim == 1) sph2.basis = Linear(tf2) * sph2.basis;
    if (ncs1.span_dim == 2) ncs1.basis = Linear(tf1) * ncs1.basis;
    if (ncs2.span_dim == 2) ncs2.basis = Linear(tf2) * ncs2.basis;

    using detail::Projection;
    using detail::Volume;

    // Compute primal solution set null space.
    if ((sph1.aff_dim == 0) || (sph2.aff_dim == 0)) {
      dd->z_nullspace.setZero();
      dd->z_nullity = 0;
    } else if (sph1.aff_dim == 2) {
      if (sph2.aff_dim == 2) {
        dd->z_nullspace.col(0) = m.cross(n).normalized();
        dd->z_nullspace.col(1) = n.cross(dd->z_nullspace.col(0));
        dd->z_nullity = 2;
      } else {
        dd->z_nullspace.col(0) = Projection(sph2.basis.col(0), n).normalized();
        dd->z_nullspace.col(1).setZero();
        dd->z_nullity = 1;
      }
    } else if ((sph2.aff_dim == 2) ||
               (Volume(sph1.basis.col(0), sph2.basis.col(0), n) <
                settings.nullspace_tol)) {
      dd->z_nullspace.col(0) = Projection(sph1.basis.col(0), n).normalized();
      dd->z_nullspace.col(1).setZero();
      dd->z_nullity = 1;
    } else {
      dd->z_nullspace.setZero();
      dd->z_nullity = 0;
    }

    // Compute dual solution set null space.
    dd->n_nullspace.col(0) = n;
    if ((ncs1.span_dim == 1) || (ncs2.span_dim == 1)) {
      dd->n_nullspace.template rightCols<2>().setZero();
      dd->n_nullity = 1;
    } else if (ncs1.span_dim == 3) {
      if (ncs2.span_dim == 3) {
        dd->n_nullspace.col(1) = m.cross(n).normalized();
        dd->n_nullspace.col(2) = n.cross(dd->n_nullspace.col(1));
        dd->n_nullity = 3;
      } else {
        dd->n_nullspace.col(1) = Projection(ncs2.basis.col(0), n);
        dd->n_nullspace.col(2).setZero();
        dd->n_nullity = 2;
      }
    } else if ((ncs2.span_dim == 3) ||
               (Volume(ncs1.basis.col(0), ncs2.basis.col(0), n) <
                settings.nullspace_tol)) {
      dd->n_nullspace.col(1) = Projection(ncs1.basis.col(0), n);
      dd->n_nullspace.col(2).setZero();
      dd->n_nullity = 2;
    } else {
      dd->n_nullspace.template rightCols<2>().setZero();
      dd->n_nullity = 1;
    }
  }

  const int nullity = dd->z_nullity + dd->n_nullity;
  dd->value_differentiable = (nullity == 1);

  return nullity;
}

/**
 * @brief Derivative of the growth distance function for 2D and 3D convex sets
 * (including half-spaces).
 *
 * @note This function does not check for optimal value differentiability.
 */
template <TwistFrame twist_frame, int dim>
inline Real GdDerivativeImpl(const KinematicState<dim>& state1,
                             const KinematicState<dim>& state2,
                             const OutputBundle<dim>& bundle) {
  using detail::ComputeVelocity;

  const auto& out = bundle.output;
  const auto& dd = bundle.dir_derivative;

  if ((!out) || (!dd)) return Real(0.0);

  if (out->status != SolutionStatus::Optimal) return (dd->d_gd = Real(0.0));

  const auto& tf1 = state1.tf;
  const auto& tf2 = state2.tf;

  const Real gd = out->growth_dist_ub;  // Corresponding to the primal solution.
  // Note: out->z2 is not set when the second convex set is a half-space.
  const Vecr<dim> z = gd * (out->z1 - Affine(tf1)) + Affine(tf1);

  dd->d_gd = gd *
             out->normal.dot(ComputeVelocity<twist_frame>(state2, z) -
                             ComputeVelocity<twist_frame>(state1, z)) /
             out->normal.dot(Affine(tf2) - Affine(tf1));

  return dd->d_gd;
}

/// @brief Dispatcher for GdDerivativeImpl based on twist frame setting.
template <int dim>
inline Real GdDerivativeTpl(const KinematicState<dim>& state1,
                            const KinematicState<dim>& state2,
                            const Settings& settings,
                            const OutputBundle<dim>& bundle) {
  switch (settings.twist_frame) {
    case TwistFrame::Spatial:
      return GdDerivativeImpl<TwistFrame::Spatial>(state1, state2, bundle);
    case TwistFrame::Hybrid:
      return GdDerivativeImpl<TwistFrame::Hybrid>(state1, state2, bundle);
    case TwistFrame::Body:
      return GdDerivativeImpl<TwistFrame::Body>(state1, state2, bundle);
  }
  return Real(0.0);
}

/**
 * @brief Gradient of the growth distance function for 2D and 3D convex sets
 * (including half-spaces).
 *
 * @note This function does not check for optimal value differentiability.
 */
template <TwistFrame twist_frame, int dim>
inline void GdGradientImpl(const Transformr<dim>& tf1,
                           const Transformr<dim>& tf2,
                           const OutputBundle<dim>& bundle) {
  using detail::ComputeDualTwist;

  const auto& out = bundle.output;
  const auto& td = bundle.total_derivative;

  if ((!out) || (!td)) return;

  if (out->status != SolutionStatus::Optimal) {
    td->d_gd_tf1 = Twistr<dim>::Zero();
    td->d_gd_tf2 = Twistr<dim>::Zero();
    return;
  }

  const Real gd = out->growth_dist_ub;  // Corresponding to the primal solution.
  // Note: out->z2 is not set when the second convex set is a half-space.
  const Vecr<dim> z = gd * (out->z1 - Affine(tf1)) + Affine(tf1);

  const Real c = gd / out->normal.dot(Affine(tf2) - Affine(tf1));
  td->d_gd_tf1 = -c * ComputeDualTwist<twist_frame>(tf1, out->normal, z);
  td->d_gd_tf2 = c * ComputeDualTwist<twist_frame>(tf2, out->normal, z);
}

/// @brief Dispatcher for GdGradientImpl based on twist frame setting.
template <int dim>
inline void GdGradientTpl(const Transformr<dim>& tf1,
                          const Transformr<dim>& tf2, const Settings& settings,
                          const OutputBundle<dim>& bundle) {
  switch (settings.twist_frame) {
    case TwistFrame::Spatial:
      GdGradientImpl<TwistFrame::Spatial>(tf1, tf2, bundle);
      break;
    case TwistFrame::Hybrid:
      GdGradientImpl<TwistFrame::Hybrid>(tf1, tf2, bundle);
      break;
    case TwistFrame::Body:
      GdGradientImpl<TwistFrame::Body>(tf1, tf2, bundle);
      break;
  }
}

}  // namespace dgd

#endif  // DGD_SOLVERS_DERIVATIVE_IMPL_H_
