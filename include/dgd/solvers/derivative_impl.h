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
#include "dgd/geometry/halfspace.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/solvers/solver_utils.h"

namespace dgd {

/*
 * KKT solution set null space algorithm.
 */

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

  const Vecr<dim>& n = out->normal;

  NormalPair<dim> zn1, zn2;
  zn1.z.noalias() = out->s1 * out->bc;
  zn1.n.noalias() = Linear(tf1).transpose() * n;
  zn2.z.noalias() = out->s2 * out->bc;
  zn2.n.noalias() = -Linear(tf2).transpose() * n;

  BasePointHint<dim> hint1{out->bc, out->idx_s1}, hint2{out->bc, out->idx_s2};

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
 * @brief KKT solution set null space algorithm for a compact convex set and a
 * half-space.
 */
template <int dim, class C1>
inline int ComputeKktNullspaceHalfspaceTpl(const C1* set1,
                                           const Transformr<dim>& tf1,
                                           const Halfspace<dim>* /*set2*/,
                                           const Transformr<dim>& tf2,
                                           const Settings& /*settings*/,
                                           const OutputBundle<dim>& bundle) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible compact set C1");

  const auto& out = bundle.output;
  const auto& dd = bundle.dir_derivative;

  if ((!out) || (!dd)) return 0;

  if (out->status != SolutionStatus::Optimal) {
    return detail::SetZeroKktNullspace(*dd);
  }

  NormalPair<dim> zn;
  zn.z.noalias() = InverseTransformPoint(tf1, out->z1);
  zn.n.noalias() = -Linear(tf1).transpose() * Linear(tf2).col(dim - 1);

  BasePointHint<dim> hint{out->bc, out->idx_s1};

  SupportPatchHull<dim> sph;
  NormalConeSpan<dim> ncs;
  set1->ComputeLocalGeometry(zn, sph, ncs, &hint);

  // Compute primal solution set null space.
  if constexpr (dim == 2) {
    dd->z_nullspace = sph.aff_dim * Linear(tf2).col(0);
  } else {  // dim = 3
    if (sph.aff_dim == 0) {
      dd->z_nullspace.setZero();
    } else if (sph.aff_dim == 1) {
      dd->z_nullspace.col(0) =
          Linear(tf1) * detail::Projection(sph.basis.col(0), zn.n).normalized();
      dd->z_nullspace.col(1).setZero();
    } else {
      dd->z_nullspace = Linear(tf2).template leftCols<2>();
    }
  }
  dd->z_nullity = sph.aff_dim;

  // Compute dual solution set null space.
  dd->n_nullspace.col(0) = -Linear(tf2).col(dim - 1);
  dd->n_nullspace.template rightCols<dim - 1>().setZero();
  dd->n_nullity = 1;

  const int nullity = dd->z_nullity + dd->n_nullity;
  dd->value_differentiable = (nullity == 1);

  return nullity;
}

/*
 * Growth distance derivative algorithm.
 */

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
  using detail::VelocityAtPoint;
  constexpr TwistFrame f = twist_frame;

  const auto& out = bundle.output;
  const auto& dd = bundle.dir_derivative;

  if (!out) return Real(0.0);

  if (out->status != SolutionStatus::Optimal) {
    if (dd) dd->d_gd = Real(0.0);
    return Real(0.0);
  }

  const auto& tf1 = state1.tf;
  const auto& tf2 = state2.tf;

  const Real gd = out->growth_dist_ub;  // Corresponding to the primal solution.
  const Vecr<dim> z = gd * (out->z1 - Affine(tf1)) + Affine(tf1);

  const Real d_gd = gd *
                    out->normal.dot(VelocityAtPoint<f>(state2, z) -
                                    VelocityAtPoint<f>(state1, z)) /
                    out->normal.dot(Affine(tf2) - Affine(tf1));

  if (dd) dd->d_gd = d_gd;
  return d_gd;
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
    default:  // Body
      return GdDerivativeImpl<TwistFrame::Body>(state1, state2, bundle);
  }
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
  using detail::DualTwistAtPoint;
  constexpr TwistFrame f = twist_frame;

  const auto& out = bundle.output;
  const auto& td = bundle.total_derivative;

  if ((!out) || (!td)) return;

  if (out->status != SolutionStatus::Optimal) {
    td->d_gd_tf1.setZero();
    td->d_gd_tf2.setZero();
    return;
  }

  const Real gd = out->growth_dist_ub;  // Corresponding to the primal solution.
  // Note: out->z2 is not set when the second convex set is a half-space.
  const Vecr<dim> z = gd * (out->z1 - Affine(tf1)) + Affine(tf1);

  const Real c = gd / out->normal.dot(Affine(tf2) - Affine(tf1));
  td->d_gd_tf1 = -c * DualTwistAtPoint<f>(tf1, out->normal, z);
  td->d_gd_tf2 = c * DualTwistAtPoint<f>(tf2, out->normal, z);
}

/// @brief Dispatcher for GdGradientImpl based on twist frame setting.
template <int dim>
inline void GdGradientTpl(const Transformr<dim>& tf1,
                          const Transformr<dim>& tf2, const Settings& settings,
                          const OutputBundle<dim>& bundle) {
  switch (settings.twist_frame) {
    case TwistFrame::Spatial:
      return GdGradientImpl<TwistFrame::Spatial>(tf1, tf2, bundle);
    case TwistFrame::Hybrid:
      return GdGradientImpl<TwistFrame::Hybrid>(tf1, tf2, bundle);
    default:  // Body
      return GdGradientImpl<TwistFrame::Body>(tf1, tf2, bundle);
  }
}

/*
 * Growth distance optimal solution derivative algorithm.
 */

/**
 * @brief Factorizes the KKT system for the growth distance optimal solution
 * derivatives for 2D and 3D compact convex sets.
 *
 * @note This function also determines the differentiability of the growth
 * distance optimal solution.
 */
template <int dim, class C1, class C2>
inline bool FactorizeKktSystemTpl(const C1* set1, const Transformr<dim>& tf1,
                                  const C2* set2, const Transformr<dim>& tf2,
                                  const Settings& settings,
                                  const OutputBundle<dim>& bundle) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible set C1");
  static_assert(detail::ConvexSetValidator<dim, C2>::valid,
                "Incompatible set C2");

  const auto& out = bundle.output;
  const auto& dd = bundle.dir_derivative;

  if ((!out) || (!dd)) return false;

  if (out->status != SolutionStatus::Optimal) {
    return (dd->differentiable = false);
  }

  const Real gd = out->growth_dist_ub;  // Corresponding to the primal solution.
  const Vecr<dim> z1l = out->s1 * out->bc, z2l = out->s2 * out->bc;
  const Vecr<dim> p = Affine(tf2) - Affine(tf1);
  const Vecr<dim> lmbd = out->normal / out->normal.dot(p);

  BasePointHint<dim> hint1{out->bc, out->idx_s1}, hint2{out->bc, out->idx_s2};

  const bool diff1 = set1->ProjectionDerivative(
      z1l + Linear(tf1).transpose() * lmbd, z1l, dd->d_pi1_, &hint1);
  const bool diff2 = set2->ProjectionDerivative(
      z2l - Linear(tf2).transpose() * lmbd, z2l, dd->d_pi2_, &hint2);
  if ((!diff1) || (!diff2)) return (dd->differentiable = false);

  dd->d_pi1_ = Linear(tf1) * dd->d_pi1_ * Linear(tf1).transpose();
  dd->d_pi2_ = Linear(tf2) * dd->d_pi2_ * Linear(tf2).transpose();
  // Numerically force the projection derivative matrices to contain the normal
  // vector in their null space.
  detail::AlignSymmetricMatrix(out->normal, dd->d_pi1_);
  detail::AlignSymmetricMatrix(out->normal, dd->d_pi2_);

  // Reduced KKT system.
  Matr<dim, dim> A = dd->d_pi1_ + dd->d_pi2_;
  A.noalias() -= Real(2.0) * dd->d_pi1_ * dd->d_pi2_;
  A.noalias() += (dd->d_pi1_ * p - p) * (dd->d_pi2_ * p - p).transpose() / gd;

  if (A.determinant() < settings.jac_tol) return (dd->differentiable = false);
  dd->lu_.compute(A);
  dd->is_halfspace_ = false;
  return true;
}

/**
 * @brief Factorizes the KKT system for the growth distance optimal solution
 * derivatives for a compact convex set and a half-space.
 *
 * @note This function also determines the differentiability of the growth
 * distance optimal solution.
 */
template <int dim, class C1>
inline bool FactorizeKktSystemHalfspaceTpl(const C1* set1,
                                           const Transformr<dim>& tf1,
                                           const Halfspace<dim>* /*set2*/,
                                           const Transformr<dim>& tf2,
                                           const Settings& settings,
                                           const OutputBundle<dim>& bundle) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible set C1");

  const auto& out = bundle.output;
  const auto& dd = bundle.dir_derivative;

  if ((!out) || (!dd)) return false;

  if (out->status != SolutionStatus::Optimal) {
    return (dd->differentiable = false);
  }

  const Vecr<dim> z1l = out->s1.col(0);
  const Vecr<dim>& n = out->normal;
  const Vecr<dim> lmbd = n / n.dot(Affine(tf2) - Affine(tf1));

  BasePointHint<dim> hint{out->bc, out->idx_s1};

  const bool diff = set1->ProjectionDerivative(
      z1l + Linear(tf1).transpose() * lmbd, z1l, dd->d_pi1_, &hint);
  if (!diff) return (dd->differentiable = false);

  dd->d_pi1_ = Linear(tf1) * dd->d_pi1_ * Linear(tf1).transpose();
  // Numerically force the projection derivative matrices to contain the normal
  // vector in their null space.
  detail::AlignSymmetricMatrix(n, dd->d_pi1_);

  // Reduced KKT system.
  Matr<dim, dim> A = -dd->d_pi1_;
  A.diagonal().array() += Real(1.0);

  if (A.determinant() < settings.jac_tol) return (dd->differentiable = false);
  dd->lu_.compute(A);
  dd->is_halfspace_ = true;
  return true;
}

namespace detail {

/**
 * @brief Solver for the growth distance optimal solution derivatives for 2D and
 * 3D convex sets (including half-spaces).
 */
template <int dim>
struct SolutionDerivativeSolver {
  /// @brief Relative center position.
  Vecr<dim> p;

  /// @brief Dual optimal solution.
  Vecr<dim> lmbd;

  /// @brief Relative center velocity.
  Vecr<dim> d_p;

  /**
   * @name Primal optimal solution velocities
   * @brief Velocities of the (local frame fixed) primal optimal solutions.
   */
  ///@{
  Vecr<dim> d_z1_tf1, d_z2_tf2;
  ///@}

  /**
   * @name Dual optimal solution velocities
   * @brief Velocities of the (local frame fixed) dual optimal solution.
   */
  ///@{
  Vecr<dim> d_lmbd_r1, d_lmbd_r2;
  ///@}

  /**
   * @name Optimal solution derivatives
   * @brief Derivatives of the primal and dual optimal solutions.
   */
  ///@{
  Vecr<dim> d_normal, d_z1, d_z2;
  ///@}

  /// @brief Growth distance.
  Real gd;

  /// @brief Growth distance derivative.
  Real d_gd;

  SolutionDerivativeSolver(const Output<dim>& out,
                           const DirectionalDerivative<dim>& dd)
      : out_(out), dd_(dd) {}

  /// @brief Initializes the solver.
  void Initialize(const Transformr<dim>& tf1, const Transformr<dim>& tf2) {
    gd = out_.growth_dist_ub;  // Corresponding to the primal solution.
    p = Affine(tf2) - Affine(tf1);
    lmbd = out_.normal / out_.normal.dot(p);
  }

  /// @brief Computes the fixed frame solution velocities.
  template <TwistFrame twist_frame>
  void ComputeVelocitiesImpl(const KinematicState<dim>& state1,
                             const KinematicState<dim>& state2) {
    using detail::VelocityAtPoint;
    using detail::VelocityOfVector;
    constexpr TwistFrame f = twist_frame;

    d_p = VelocityAtPoint<f, dim>(state2, Affine(state2.tf)) -
          VelocityAtPoint<f, dim>(state1, Affine(state1.tf));
    d_z1_tf1 = VelocityAtPoint<f, dim>(state1, out_.z1);
    d_z2_tf2 = VelocityAtPoint<f, dim>(state2, out_.z2);
    d_lmbd_r1 = VelocityOfVector<f, dim>(state1, lmbd);
    d_lmbd_r2 = VelocityOfVector<f, dim>(state2, lmbd);
  }

  /// @brief Dispatcher for ComputeVelocitiesImpl based on twist frame setting.
  void ComputeVelocities(const KinematicState<dim>& state1,
                         const KinematicState<dim>& state2,
                         TwistFrame twist_frame) {
    switch (twist_frame) {
      case TwistFrame::Spatial:
        ComputeVelocitiesImpl<TwistFrame::Spatial>(state1, state2);
        break;
      case TwistFrame::Hybrid:
        ComputeVelocitiesImpl<TwistFrame::Hybrid>(state1, state2);
        break;
      default:  // Body
        ComputeVelocitiesImpl<TwistFrame::Body>(state1, state2);
    }
  }

  /**
   * @brief Computes the primal and dual optimal solution derivatives for 2D and
   * 3D compact convex sets.
   */
  void ComputeDerivatives() {
    const Vecr<dim> b1 = (Real(1.0) - gd) * d_p;
    const Vecr<dim> b2 = d_z1_tf1 - dd_.d_pi1_ * (d_lmbd_r1 + d_z1_tf1);
    const Vecr<dim> b3 = d_z2_tf2 + dd_.d_pi2_ * (d_lmbd_r2 - d_z2_tf2);
    const Real s = lmbd.dot(d_p) + p.dot(b3);

    const Vecr<dim> u = (s * p - b1) / gd - b3;
    const Vecr<dim> y1 = dd_.lu_.solve(b2 + u + dd_.d_pi1_ * (b3 - u));
    const Real y2 = y1.dot(dd_.d_pi2_ * p - p) + s;

    d_z2 = dd_.d_pi2_ * y1 + b3;
    d_z1 = d_z2 + (b1 - p * y2) / gd;
    d_normal =
        (d_z2 - y1 - out_.normal.dot(d_z2 - y1) * out_.normal) / lmbd.norm();
    // Note that the equation in GdDerivativeImpl gives the same result (within
    // machine precision).
    d_gd = gd * y2;
  }

  /**
   * @brief Computes the primal and dual optimal solution derivatives for a
   * compact convex set and a half-space.
   */
  void ComputeDerivativesHalfspace() {
    const Vecr<dim> u = d_z2_tf2 - d_z1_tf1 - d_p;
    const Real gd_inv = Real(1.0) / gd;

    d_normal = d_lmbd_r2 * out_.normal.dot(p);
    d_z1 = d_z1_tf1 + dd_.lu_.solve(dd_.d_pi1_ *
                                    (d_lmbd_r2 - d_lmbd_r1 +
                                     (lmbd.dot(u) - p.dot(d_lmbd_r2)) * lmbd));
    d_gd = gd * lmbd.dot(gd * u + d_p);
    d_z2 = d_z1 + (Real(1.0) - gd_inv) * d_p + (d_gd * gd_inv * gd_inv) * p;
  }

 private:
  const Output<dim>& out_;
  const DirectionalDerivative<dim>& dd_;
};

}  // namespace detail

/**
 * @brief Derivative of the growth distance optimal solution for 2D and 3D
 * convex sets (including half-spaces).
 */
template <int dim>
inline void GdSolutionDerivativeTpl(const KinematicState<dim>& state1,
                                    const KinematicState<dim>& state2,
                                    const Settings& settings,
                                    const OutputBundle<dim>& bundle) {
  const auto& out = bundle.output;
  const auto& dd = bundle.dir_derivative;

  if ((!out) || (!dd) || (!dd->differentiable)) {
    detail::SetZeroSolutionDerivative(*dd);
  }

  // Initialize the solver.
  detail::SolutionDerivativeSolver<dim> sds(*out, *dd);
  sds.Initialize(state1.tf, state2.tf);

  // Compute the fixed frame solution velocities.
  sds.ComputeVelocities(state1, state2, settings.twist_frame);

  // Compute the primal and dual optimal solution derivatives.
  if (dd->is_halfspace_) {
    sds.ComputeDerivativesHalfspace();
  } else {
    sds.ComputeDerivatives();
  }

  dd->d_normal = sds.d_normal;
  dd->d_z1 = sds.d_z1;
  dd->d_z2 = sds.d_z2;
  dd->d_gd = sds.d_gd;

  /*
  // Full KKT system: (for compact convex sets)
  // Variables:
  //  [gd R1 z1l', gd R2 z2l', (gd lmbd)', gd' / gd].
  // To convert the full system to the reduced KKT system:
  // Reduced variables:
  //  [x - (gd lmbd)' / gd],
  // where x = R2' z2l + R2 * z2l' + p2'.
  const Vecr<dim> z = gd * (out->z1 - Affine(tf1)) + Affine(tf1);
  const Vecr<dim> d_z_tf1 = VelocityAtPoint<f>(state1, z);
  const Vecr<dim> d_z_tf2 = VelocityAtPoint<f>(state2, z);
  const Vecr<dim> sep_vel = d_z_tf2 - d_z_tf1;
  // d_gd is solved together with other variables for robustness.
  // dd->d_gd = gd * lmbd.dot(sep_vel);

  // Full KKT system.
  Matr<3 * dim + 1, 3 * dim + 1> A;
  A.setZero();
  A.template block<dim, dim>(0, 0).setIdentity();
  A.template block<dim, dim>(0, dim) = -Matr<dim, dim>::Identity();
  A.template block<dim, 1>(0, 3 * dim) = p;
  A.template block<dim, dim>(dim, 0) = -dd->d_pi1_;
  A.template block<dim, dim>(dim, 0).diagonal().array() += Real(1.0);
  A.template block<dim, dim>(dim, 2 * dim) = -dd->d_pi1_;
  A.template block<dim, dim>(2 * dim, dim) = -dd->d_pi2_;
  A.template block<dim, dim>(2 * dim, dim).diagonal().array() += Real(1.0);
  A.template block<dim, dim>(2 * dim, 2 * dim) = dd->d_pi2_;
  A.template block<1, dim>(3 * dim, 2 * dim) = p.transpose();
  A(3 * dim, 3 * dim) = -gd;

  Vecr<3 * dim + 1> b;
  b.template head<dim>() = sep_vel;
  b.template segment<dim>(dim) = gd * dd->d_pi1_ * d_lmbd_r1;
  b.template segment<dim>(2 * dim) = -(gd * dd->d_pi2_ * d_lmbd_r2);
  b(3 * dim) =
      -gd * lmbd.dot(VelocityAtPoint<f, dim>(state2, Affine(tf2)) -
                     VelocityAtPoint<f, dim>(state1, Affine(tf1)));

  Eigen::PartialPivLU<Matr<3 * dim + 1, 3 * dim + 1>> lu(A);
  if (std::abs(lu.determinant()) < settings.jac_tol) return false;

  // Extract solution derivatives.
  const Vecr<3 * dim + 1> y = lu.solve(b);
  const Vecr<dim> y1 = y.template head<dim>();
  const Vecr<dim> y2 = y.template segment<dim>(dim);
  const Vecr<dim> y3 = y.template segment<dim>(2 * dim);

  dd->d_normal = (y3 - n * n.dot(y3)) / (gd * lmbd.norm());
  dd->d_z1 = VelocityAtPoint<f>(state1, out->z1) + y1 / gd;
  dd->d_z2 = VelocityAtPoint<f>(state2, out->z2) + y2 / gd;
  dd->d_gd = gd * y(3 * dim);
  */
}

/**
 * @brief Jacobian of the growth distance optimal solution for 2D and 3D convex
 * sets (including half-spaces).
 */
template <int dim>
inline void GdJacobianTpl(const Transformr<dim>& tf1,
                          const Transformr<dim>& tf2, const Settings& settings,
                          const OutputBundle<dim>& bundle) {
  using Gradr = typename TotalDerivative<dim>::Gradr;
  using Jacr = typename TotalDerivative<dim>::Jacr;
  constexpr int tw_dim = SeDim<dim>();

  const auto& out = bundle.output;
  const auto& dd = bundle.dir_derivative;
  const auto& td = bundle.total_derivative;

  if ((!out) || (!dd) || (!td) || (!dd->differentiable)) {
    detail::SetZeroSolutionDerivative(*dd);
  }

  detail::SolutionDerivativeSolver<dim> sds(*out, *dd);
  sds.Initialize(tf1, tf2);

  KinematicState<dim> state1{}, state2{};
  state1.tf = tf1;
  state2.tf = tf2;

  auto set_derivatives = [&](int i, Jacr& d_normal_tf, Jacr& d_z1_tf,
                             Jacr& d_z2_tf, Gradr& d_gd_tf) {
    sds.ComputeVelocities(state1, state2, settings.twist_frame);
    if (dd->is_halfspace_) {
      sds.ComputeDerivativesHalfspace();
    } else {
      sds.ComputeDerivatives();
    }
    d_normal_tf.col(i) = sds.d_normal;
    d_z1_tf.col(i) = sds.d_z1;
    d_z2_tf.col(i) = sds.d_z2;
    d_gd_tf(i) = sds.d_gd;
  };

  for (int i = 0; i < tw_dim; ++i) {
    state1.tw(i) = Real(1.0);
    set_derivatives(i, td->d_normal_tf1, td->d_z1_tf1, td->d_z2_tf1,
                    td->d_gd_tf1);
    state1.tw(i) = Real(0.0);
  }

  for (int i = 0; i < tw_dim; ++i) {
    state2.tw(i) = Real(1.0);
    set_derivatives(i, td->d_normal_tf2, td->d_z1_tf2, td->d_z2_tf2,
                    td->d_gd_tf2);
    state2.tw(i) = Real(0.0);
  }
}

}  // namespace dgd

#endif  // DGD_SOLVERS_DERIVATIVE_IMPL_H_
