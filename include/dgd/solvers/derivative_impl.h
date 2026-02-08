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
                                  const Settings& settings, Output<dim>& out) {
  static_assert(detail::ConvexSetValidator<dim, C1>::valid,
                "Incompatible set C1");
  static_assert(detail::ConvexSetValidator<dim, C2>::valid,
                "Incompatible set C2");

  if (out.status != SolutionStatus::Optimal) {
    out.z_nullspace = Matr<dim, dim - 1>::Zero();
    out.n_nullspace = Matr<dim, dim>::Zero();
    out.z_nullity = dim - 1;
    out.n_nullity = dim;
    return -1;
  }

  const Vecr<dim> n = out.normal.normalized();

  NormalPair<dim> zn1, zn2;
  zn1.z = out.s1 * out.bc;
  zn1.n = Linear(tf1).transpose() * n;
  zn2.z = out.s2 * out.bc;
  zn2.n = -Linear(tf2).transpose() * n;

  BasePointHint<dim> hint1, hint2;
  hint1.s = &out.s1;
  hint1.bc = &out.bc;
  hint1.sfh = &out.hint1_;
  hint2.s = &out.s2;
  hint2.bc = &out.bc;
  hint2.sfh = &out.hint2_;

  SupportPatchHull<dim> sph1, sph2;
  NormalConeSpan<dim> ncs1, ncs2;

  set1->ComputeLocalGeometry(zn1, sph1, ncs1, &hint1);
  set2->ComputeLocalGeometry(zn2, sph2, ncs2, &hint2);

  if constexpr (dim == 2) {
    // Compute primal solution set null space.
    if ((sph1.aff_dim == 0) || (sph2.aff_dim == 0)) {
      out.z_nullity = 0;
    } else {
      out.z_nullspace.col(0) = Vec2r(n(1), -n(0));
      out.z_nullity = 1;
    }

    // Compute dual solution set null space.
    out.n_nullspace.col(0) = n;
    if ((ncs1.span_dim == 1) || (ncs2.span_dim == 1)) {
      out.n_nullity = 1;
    } else {
      out.n_nullspace.col(1) = Vec2r(n(1), -n(0));
      out.n_nullity = 2;
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
      out.z_nullity = 0;
    } else if (sph1.aff_dim == 2) {
      if (sph2.aff_dim == 2) {
        out.z_nullspace.col(0) = m.cross(n).normalized();
        out.z_nullspace.col(1) = n.cross(out.z_nullspace.col(0));
        out.z_nullity = 2;
      } else {
        out.z_nullspace.col(0) = Projection(sph2.basis.col(0), n).normalized();
        out.z_nullity = 1;
      }
    } else if ((sph2.aff_dim == 2) ||
               (Volume(sph1.basis.col(0), sph2.basis.col(0), n) <
                settings.nullspace_tol)) {
      out.z_nullspace.col(0) = Projection(sph1.basis.col(0), n).normalized();
      out.z_nullity = 1;
    } else {
      out.z_nullity = 0;
    }

    // Compute dual solution set null space.
    out.n_nullspace.col(0) = n;
    if ((ncs1.span_dim == 1) || (ncs2.span_dim == 1)) {
      out.n_nullity = 1;
    } else if (ncs1.span_dim == 3) {
      if (ncs2.span_dim == 3) {
        out.n_nullspace.col(1) = m.cross(n).normalized();
        out.n_nullspace.col(2) = n.cross(out.n_nullspace.col(1));
        out.n_nullity = 3;
      } else {
        out.n_nullspace.col(1) = Projection(ncs2.basis.col(0), n);
        out.n_nullity = 2;
      }
    } else if ((ncs2.span_dim == 3) ||
               (Volume(ncs1.basis.col(0), ncs2.basis.col(0), n) <
                settings.nullspace_tol)) {
      out.n_nullspace.col(1) = Projection(ncs1.basis.col(0), n);
      out.n_nullity = 2;
    } else {
      out.n_nullity = 1;
    }
  }

  return out.z_nullity + out.n_nullity;
}

}  // namespace dgd

#endif  // DGD_SOLVERS_DERIVATIVE_IMPL_H_
