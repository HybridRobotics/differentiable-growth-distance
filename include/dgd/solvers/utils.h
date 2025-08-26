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

#ifndef DGD_SOLVERS_UTILS_H_
#define DGD_SOLVERS_UTILS_H_

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/output.h"
#include "dgd/solvers/solver_options.h"

namespace dgd {

namespace detail {

// Sets rot such that rot * n = (0, 1).
inline void RotationToZAxis(const Vec2r& n, Rotation2r& rot) {
  rot << n(1), -n(0), n(0), n(1);
}

// Sets rot such that rot * n = (0, 0, 1).
inline void RotationToZAxis(const Vec3r& n, Rotation3r& rot) {
  Vec3r axis = n + Vec3r::UnitZ();
  const Real norm = axis.norm();
  if (norm > kEps) {
    axis /= norm;
    rot = Real(2.0) * axis * axis.transpose() - Rotation3r::Identity();
  } else {
    rot = Vec3r(1.0, -1.0, -1.0).asDiagonal();
  }
}

// Initializes the output for cold start.
template <int dim, class C1, class C2>
inline void InitializeOutput(const C1* set1, const C2* set2, Output<dim>& out) {
  out.hint2_.n_prev = out.hint1_.n_prev = Vecr<dim>::Zero();
  out.r1_ = set1->inradius();
  out.r2_ = set2->inradius();
  out.normalize_2norm_ = set1->RequireUnitNormal() || set2->RequireUnitNormal();
}

// Sets zero output when the centers of the convex sets coincide.
template <int dim>
inline Real SetZeroOutput(const Transformr<dim>& tf1,
                          const Transformr<dim>& tf2, Output<dim>& out) {
  out.normal = Vecr<dim>::Zero();
  out.growth_dist_ub = out.growth_dist_lb = 0.0;
  out.z1 = Affine(tf1);
  out.z2 = Affine(tf2);
  out.status = SolutionStatus::CoincidentCenters;
  return 0.0;
}

// Normalizes the normal vector.
template <int dim>
inline void NormalizeNormal(Vecr<dim>& n, bool normalize_2norm) {
  if (normalize_2norm) {
    n.normalize();
  } else {
    n /= n.template lpNorm<Eigen::Infinity>();
  }
}

// Computes the primal optimal solution in the world frame of reference.
template <int dim>
inline Real ComputePrimalSolution(const Transformr<dim>& tf1,
                                  const Transformr<dim>& tf2, Real cdist,
                                  Real lb, Output<dim>& out) {
  out.z1.noalias() = Linear(tf1) * out.s1 * out.bc + Affine(tf1);
  out.z2.noalias() = Linear(tf2) * out.s2 * out.bc + Affine(tf2);
  return out.growth_dist_ub = cdist / lb;
}

// Computes the dual optimal solution in the world frame of reference.
template <int dim>
inline Real ComputeDualSolution(const Rotationr<dim>& rot, Real cdist, Real ub,
                                Output<dim>& out) {
  out.normal = rot.transpose() * out.normal;
  return out.growth_dist_lb = cdist / ub;
}

// Computes gamma for the proximal bundle method.
inline Real ComputeGammaProximalBundle(Real ub, Real r, int iter) {
  Real gamma;
  if constexpr (SolverSettings::kProxThresh < kEps) {
    gamma = r / ub;
  } else {
    gamma = Real(1.0) /
            std::sqrt((ub * ub) / (r * r) - SolverSettings::kProxThresh);
  }
  if constexpr (SolverSettings::kProxRegType ==
                ProximalRegularization::kConstant) {
    gamma *= SolverSettings::kProxKc;
  } else {
    gamma *= Real(iter) * SolverSettings::kProxKa;
  }
  return gamma;
}

// Helper struct that is false for any solver type.
template <SolverType S>
struct always_false : std::false_type {};

// Minkowski difference set properties.
template <int dim>
struct MinkowskiDiffProp {
  // Alignment rotation matrices.
  Matr<dim, dim> rot, rot1, rot2;

  // Relative center position.
  Vecr<dim> p21;

  // Center distance.
  Real cdist;

  // Minkowski difference set inradius.
  Real r;

  // Sets p21 and cdist.
  void SetCenterDistance(const Transformr<dim>& tf1,
                         const Transformr<dim>& tf2) {
    p21 = Affine(tf2) - Affine(tf1);
    cdist = p21.norm();
  }

  // Sets rot, rot1, and rot2.
  void SetRotationMatrices(const Transformr<dim>& tf1,
                           const Transformr<dim>& tf2) {
    RotationToZAxis(p21 / cdist, rot);
    rot1.noalias() = rot * Linear(tf1);
    rot2.noalias() = rot * Linear(tf2);
  }
};

// nth-order support function output.
template <int dim, int order>
struct SupportFunctionOutput;

template <int dim>
struct SupportFunctionOutput<dim, 1> {
  // Support points.
  Vecr<dim> sp, sp1, sp2;

  // Support values.
  Real sv1, sv2;

  // Evaluates the support function at the given normal vector.
  template <class C1, class C2>
  void Evaluate(const C1* set1, const C2* set2,
                const MinkowskiDiffProp<dim>& mdp, const Vecr<dim>& n,
                Output<dim>& out) {
    sv1 = set1->SupportFunction(mdp.rot1.transpose() * n, sp1, &out.hint1_);
    sv2 = set2->SupportFunction(-mdp.rot2.transpose() * n, sp2, &out.hint2_);
    sp.noalias() = mdp.rot1 * sp1 - mdp.rot2 * sp2;
  }
};

template <int dim>
struct SupportFunctionOutput<dim, 2> {
  // Support function derivatives.
  SupportFunctionDerivatives<dim> deriv1, deriv2;

  // Support function Hessian.
  Matr<dim, dim> Dsp;

  // Support point.
  Vecr<dim> sp;

  // Support values.
  Real sv1, sv2;

  // Differentiability.
  bool differentiable;

  // Evaluates the support function at the given normal vector.
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
};

}  // namespace detail

}  // namespace dgd

#endif  // DGD_SOLVERS_UTILS_H_
