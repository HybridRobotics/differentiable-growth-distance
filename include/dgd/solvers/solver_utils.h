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
 * @brief Common utility functions for the solvers.
 */

#ifndef DGD_SOLVERS_SOLVER_UTILS_H_
#define DGD_SOLVERS_SOLVER_UTILS_H_

#include <Eigen/Geometry>
#include <cmath>
#include <type_traits>

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/output.h"
#include "dgd/solvers/solver_settings.h"
#include "dgd/solvers/solver_types.h"
#include "dgd/utils/transformations.h"

namespace dgd {

namespace detail {

/// @brief Helper struct that is false for any solver type.
template <SolverType S>
struct always_false : std::false_type {};

/// @brief Increments i modulo n.
template <int n>
inline constexpr int Inc(int i) {
  return (i == n - 1) ? 0 : i + 1;
}

/// @brief Decrements i modulo n.
template <int n>
inline constexpr int Dec(int i) {
  return (i == 0) ? n - 1 : i - 1;
}

/// @brief Inverts 0 and 1.
inline constexpr int Inv(int i) { return 1 - i; }

/// @brief Rectified Linear Unit (ReLU) function.
inline constexpr Real Relu(Real x) { return std::max(x, Real(0.0)); }

/// @brief Projects a vector onto the orthogonal plane of a unit-norm vector.
inline Vec3r Projection(const Vec3r& v, const Vec3r& n) {
  return v - n.dot(v) * n;
}

/// @brief Computes the volume of a parallelepiped defined by three vectors.
inline Real Volume(const Vec3r& v1, const Vec3r& v2, const Vec3r& v3) {
  return std::abs(v1.cross(v2).dot(v3));
}

/// @brief Alignment rotation function.
template <int dim>
Rotationr<dim> RotationToZAxis(const Vecr<dim>& n) {
  static_assert((dim == 2) || (dim == 3), "Dimension must be 2 or 3");
}

/// @brief Returns a rotation matrix \f$R\f$ such that \f$R n = (0, 1)\f$.
template <>
inline Rotation2r RotationToZAxis(const Vec2r& n) {
  return (Rotation2r() << n(1), -n(0), n(0), n(1)).finished();
}

/// @brief Returns a rotation matrix \f$R\f$ such that \f$R n = (0, 0, 1)\f$.
template <>
inline Rotation3r RotationToZAxis(const Vec3r& n) {
  Vec3r axis = n + Vec3r::UnitZ();
  const Real norm2 = axis.squaredNorm();
  if (norm2 > kEps * kEps) {
    Rotation3r m;
    m.noalias() = (Real(2.0) / norm2) * axis * axis.transpose();
    m.diagonal() -= Vec3r::Ones();
    return m;
  } else {
    return Vec3r(Real(1.0), Real(-1.0), Real(-1.0)).asDiagonal();
  }
}

/// @brief Initializes the output for cold start.
template <int dim, class C1, class C2>
inline void InitializeOutput(const C1* set1, const C2* set2, Output<dim>& out) {
  out.hint2_.n_prev = out.hint1_.n_prev = Vecr<dim>::Zero();
  out.r1_ = set1->inradius();
  out.r2_ = set2->inradius();
  out.normalize_2norm_ = set1->RequireUnitNormal() || set2->RequireUnitNormal();
}

/// @brief Sets zero output when the centers of the convex sets coincide.
template <int dim>
inline Real SetZeroOutput(const Transformr<dim>& tf1,
                          const Transformr<dim>& tf2, Output<dim>& out) {
  out.normal = Vecr<dim>::Zero();
  out.growth_dist_ub = out.growth_dist_lb = Real(0.0);
  out.z1 = Affine(tf1);
  out.z2 = Affine(tf2);
  out.status = SolutionStatus::CoincidentCenters;
  return Real(0.0);
}

/// @brief Sets the output when the input convex sets are ill-conditioned.
template <int dim>
inline Real SetInfOutput(Output<dim>& out) {
  out.growth_dist_ub = kInf;
  out.growth_dist_lb = Real(0.0);
  out.status = SolutionStatus::IllConditionedInputs;
  return Real(0.0);
}

/// @brief Normalizes the normal vector.
template <int dim>
inline void NormalizeNormal(Vecr<dim>& n, bool normalize_2norm) {
  if (normalize_2norm) {
    n.normalize();
  } else {
    n /= n.template lpNorm<Eigen::Infinity>();
  }
}

/// @brief Computes the primal optimal solution in the world frame of reference.
template <int dim>
inline Real ComputePrimalSolution(const Transformr<dim>& tf1,
                                  const Transformr<dim>& tf2, Real cdist,
                                  Real lb, Output<dim>& out) {
  out.z1.noalias() = TransformPoint<dim>(tf1, out.s1 * out.bc);
  out.z2.noalias() = TransformPoint<dim>(tf2, out.s2 * out.bc);
  return out.growth_dist_ub = cdist / lb;
}

/// @brief Computes the dual optimal solution in the world frame of reference.
template <int dim>
inline Real ComputeDualSolution(const Rotationr<dim>& rot, Real cdist, Real ub,
                                Output<dim>& out) {
  out.normal = rot.transpose() * out.normal;
  return out.growth_dist_lb = cdist / ub;
}

/// @brief Minkowski difference set properties.
template <int dim>
struct MinkowskiDiffProp {
  /// @brief Alignment rotation matrices.
  Matr<dim, dim> rot, rot1, rot2;

  /// @brief Relative center position.
  Vecr<dim> p21;

  /// @brief Center distance.
  Real cdist;

  /// @brief Minkowski difference set inradius.
  Real r;

  /// @brief Sets p21 and cdist.
  void SetCenterDistance(const Transformr<dim>& tf1,
                         const Transformr<dim>& tf2) {
    p21 = Affine(tf2) - Affine(tf1);
    cdist = p21.norm();
  }

  /// @brief Sets rot, rot1, and rot2.
  void SetRotationMatrices(const Transformr<dim>& tf1,
                           const Transformr<dim>& tf2) {
    rot = RotationToZAxis<dim>(p21 / cdist);
    rot1.noalias() = rot * Linear(tf1);
    rot2.noalias() = rot * Linear(tf2);
  }
};

/// @brief nth-order support function output.
template <int dim, int order>
struct SupportFunctionOutput;

template <int dim>
struct SupportFunctionOutput<dim, 1> {
  /// @brief Support points.
  Vecr<dim> sp, sp1_, sp2_;

  /// @brief Support values.
  Real sv1, sv2;

  /// @brief Evaluates the support function at the given normal vector.
  template <class C1, class C2>
  void Evaluate(const C1* set1, const C2* set2,
                const MinkowskiDiffProp<dim>& mdp, const Vecr<dim>& n,
                Output<dim>& out) {
    sv1 = set1->SupportFunction(mdp.rot1.transpose() * n, sp1_, &out.hint1_);
    sv2 = set2->SupportFunction(-mdp.rot2.transpose() * n, sp2_, &out.hint2_);
    sp.noalias() = mdp.rot1 * sp1_ - mdp.rot2 * sp2_;
  }

  /// @brief Gets the support points.
  const Vecr<dim>& sp1() const { return sp1_; }
  const Vecr<dim>& sp2() const { return sp2_; }
};

template <int dim>
struct SupportFunctionOutput<dim, 2> {
  /// @brief Support function derivatives.
  SupportFunctionDerivatives<dim> deriv1, deriv2;

  /// @brief Support function Hessian.
  Matr<dim, dim> Dsp;

  /// @brief Support point.
  Vecr<dim> sp;

  /// @brief Support values.
  Real sv1, sv2;

  /// @brief Differentiability.
  bool differentiable;

  /// @brief Evaluates the support function at the given normal vector.
  template <class C1, class C2>
  void Evaluate(const C1* set1, const C2* set2,
                const MinkowskiDiffProp<dim>& mdp, const Vecr<dim>& n,
                Output<dim>& out) {
    sv1 = set1->SupportFunction(mdp.rot1.transpose() * n, deriv1, &out.hint1_);
    sv2 = set2->SupportFunction(-mdp.rot2.transpose() * n, deriv2, &out.hint2_);
    sp.noalias() = mdp.rot1 * deriv1.sp - mdp.rot2 * deriv2.sp;
    differentiable = deriv1.differentiable & deriv2.differentiable;
    if (differentiable) {
      Dsp.noalias() = mdp.rot1 * deriv1.Dsp * mdp.rot1.transpose() +
                      mdp.rot2 * deriv2.Dsp * mdp.rot2.transpose();
    }
  }

  /// @brief Evaluates the first-order support function at the given normal
  /// vector.
  template <class C1, class C2>
  void EvaluateFirstOrder(const C1* set1, const C2* set2,
                          const MinkowskiDiffProp<dim>& mdp, const Vecr<dim>& n,
                          Output<dim>& out) {
    sv1 =
        set1->SupportFunction(mdp.rot1.transpose() * n, deriv1.sp, &out.hint1_);
    sv2 = set2->SupportFunction(-mdp.rot2.transpose() * n, deriv2.sp,
                                &out.hint2_);
    sp.noalias() = mdp.rot1 * deriv1.sp - mdp.rot2 * deriv2.sp;
  }

  /// @brief Gets the support points.
  const Vecr<dim>& sp1() const { return deriv1.sp; }
  const Vecr<dim>& sp2() const { return deriv2.sp; }
};

/// @brief Sets zero KKT solution set null space.
template <int dim>
inline int SetZeroKktNullspace(DirectionalDerivative<dim>& dd) {
  dd.z_nullspace = Matr<dim, dim - 1>::Zero();
  dd.n_nullspace = Matr<dim, dim>::Zero();
  dd.z_nullity = 0;
  dd.n_nullity = 0;
  return 0;
}

/**
 * @brief Computes the velocity of a point on a rigid body given its twist.
 *
 * @note The twist frame of reference is given by twist_frame.
 *
 * @param  state Kinematic state of the rigid body.
 * @param  pt    Point on the rigid body in the world frame.
 * @return Velocity of the point in the world frame.
 */
template <TwistFrame twist_frame, int dim>
inline Vecr<dim> ComputeVelocity(const KinematicState<dim>& state,
                                 const Vecr<dim>& pt) {
  if constexpr (twist_frame == TwistFrame::Spatial) {
    return VelocityAtPoint(state.tw, pt);
  } else if constexpr (twist_frame == TwistFrame::Hybrid) {
    return VelocityAtPoint(state.tw, pt - Affine(state.tf));
  } else {  // Body
    return Linear(state.tf) *
           VelocityAtPoint(state.tw, InverseTransformPoint(state.tf, pt));
  }
}

/**
 * @brief Computes the dual twist on a rigid body given a dual velocity at a
 * point.
 *
 * @param  f  Dual velocity at the point in the world frame.
 * @param  pt Point on the rigid body in the world frame.
 * @return Dual twist on the rigid body in the twist_frame frame.
 */
template <TwistFrame twist_frame, int dim>
inline Twistr<dim> ComputeDualTwist([[maybe_unused]] const Transformr<dim>& tf,
                                    const Vecr<dim>& f, const Vecr<dim>& pt) {
  if constexpr (twist_frame == TwistFrame::Spatial) {
    return DualTwistAtPoint(f, pt);
  } else if constexpr (twist_frame == TwistFrame::Hybrid) {
    return DualTwistAtPoint(f, pt - Affine(tf));
  } else {  // Body
    return DualTwistAtPoint(Linear(tf).transpose() * f,
                            InverseTransformPoint(tf, pt));
  }
}

}  // namespace detail

}  // namespace dgd

#endif  // DGD_SOLVERS_SOLVER_UTILS_H_
