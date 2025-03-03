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
 * @file growth_distance.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-02-18
 * @brief Growth distance algorithm for convex sets.
 */

#ifndef DGD_GROWTH_DISTANCE_H_
#define DGD_GROWTH_DISTANCE_H_

#include "dgd/data_types.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/utils.h"

#ifndef DONOT_PRINT_DIAGNOSTICS
#include "dgd/io.h"
#endif

namespace dgd {

/**********************************************************
 * GROWTH DISTANCE ALGORITHM FOR 2D CONVEX SETS           *
 **********************************************************/

namespace {

/**
 * @brief Sets zero output when the centers of the convex sets coincide.
 *
 * @tparam     dim Dimension of the convex sets.
 * @param[out] out Solver output.
 * @return     Growth distance (\f$= 0\f$).
 */
template <int dim>
inline Real SetZeroOutput(SolverOutput<dim>& out) {
  out.growth_dist_ub = out.growth_dist_lb = Real(0.0);
  out.status = SolutionStatus::kCoincidentCenters;
  return Real(0.0);
}

/**
 * @brief Updates the normal vector using the simplex.
 *
 * @param[in]  simplex Simplex.
 * @param[out] normal  Normal vector.
 */
inline void UpdateNormal(const Matf<2, 2>& simplex, Vec2f& normal) {
  normal(0) = simplex(1, 1) - simplex(1, 0);
  normal(1) = simplex(0, 0) - simplex(0, 1) -
              kEps;  // Small constant added for dual feasibility.
  normal.normalize();
}

/**
 * @brief Updates the simplex and barycentric coordinates and returns the upper
 * bound, given a support point.
 *
 * @param[in]  sp      Support point for the Minkowski difference set.
 * @param[in]  sp1     Support point for convex set 1.
 * @param[in]  sp2     Support point for convex set 2.
 * @param[out] simplex Simplex.
 * @param[out] out     Solver output.
 * @return     The updated value of the upper bound.
 */
template <typename DerivedA, typename DerivedB>
inline Real UpdateSimplex(const Vec2f& sp, const MatrixBase<DerivedA>& sp1,
                          const MatrixBase<DerivedB>& sp2, Matf<2, 2>& simplex,
                          SolverOutput<2>& out) {
  const int idx = (sp(0) >= 0);
  simplex.col(idx) = sp;
  out.s1.col(idx) = sp1;
  out.s2.col(idx) = sp2;

  const Real len{simplex(0, 1) - simplex(0, 0) + kEps};
  out.bc(0) = (simplex(0, 1) + kEps / Real(2.0)) / len;
  out.bc(1) = (-simplex(0, 0) + kEps / Real(2.0)) / len;

  return simplex.row(1) * out.bc;
}

/**
 * @brief Initializes the simplex for the algorithm.
 *
 * @param[out]    normal  Initial normal vector.
 * @param[out]    simplex Initial simplex.
 * @param[in,out] out     Solver output.
 */
inline void InitializeSimplex(Vec2f& normal, Matf<2, 2>& simplex,
                              SolverOutput<2>& out) {
  normal = -Vec2f::UnitY();

  simplex.col(0) = -out.inradius * Vec2f::UnitX();
  simplex.col(1) = out.inradius * Vec2f::UnitX();

  out.bc = Real(0.5) * Vec2f::Ones();
}

}  // namespace

/**
 * @brief Growth distance algorithm for 2D convex sets.
 *
 * @param[in]     set1       Convex set 1.
 * @param[in]     tf1        Rigid body transformation for convex set 1.
 * @param[in]     set2       Convex set 2.
 * @param[in]     tf2        Rigid body transformation for convex set 2.
 * @param[in]     settings   Solver settings.
 * @param[in,out] out        Solver output.
 * @param         warm_start Use previous solver output to warm start current
 *                           solution.
 * @return        (lower bound of) the growth distance.
 */
template <class C1, class C2>
Real GrowthDistance(const C1* set1, const Transform2f& tf1, const C2* set2,
                    const Transform2f& tf2, const SolverSettings& settings,
                    SolverOutput<2>& out, bool warm_start) {
  static_assert((C1::Dimension() == 2) && (C2::Dimension() == 2),
                "Convex sets are not two-dimensional!");

  if (!warm_start) out.inradius = set1->GetInradius() + set2->GetInradius();
  out.iter = 0;

  // Check center distance.
  const Vec2f p12{tf1.block<2, 1>(0, 2) - tf2.block<2, 1>(0, 2)};
  const Real cdist{p12.norm()};
  if (cdist < settings.min_center_dist) return SetZeroOutput(out);

  // Alignment rotation matrix.
  Rot2f rot;
  RotationToZAxis(p12 / cdist, rot);
  // Axis-aligned rotation matrices.
  const Rot2f rot1{rot * tf1.block<2, 2>(0, 0)};
  const Rot2f rot2{rot * tf2.block<2, 2>(0, 0)};

  // Growth distance bounds.
  Real lb{-kInf}, ub{0.0};
  // Normal vector, suppport points, and the simplex matrix.
  Vec2f normal, sp1, sp2, sp;
  Matf<2, 2> simplex;

  InitializeSimplex(normal, simplex, out);
  if (warm_start && (out.status == SolutionStatus::kOptimal)) {
    const Matf<2, 2> s1_{out.s1};
    const Matf<2, 2> s2_{out.s2};
    for (int i = 0; i < 2; ++i)
      if (out.bc(i) > kEps) {
        sp.noalias() = rot1 * s1_.col(i) - rot2 * s2_.col(i);
        if (normal.dot(sp - simplex.col(0)) > Real(0.0)) {
          ub = UpdateSimplex(sp, s1_.col(i), s2_.col(i), simplex, out);
          UpdateNormal(simplex, normal);
        }
      }
  }

#ifndef DONOT_PRINT_DIAGNOSTICS
  io::PrintDiagnosticsHeader(lb, ub, simplex, settings, out);
#endif

  // Loop invariants (at the start of each iteration):
  //  a) normal, ub, and bc correspond to simplex.
  //  b) lb and out.normal are correctly set for iter >= 1.
  // s1 and s2 (corresponding to nonzero bc) are correctly set at the end of the
  // algorithm.
  while (true) {
    // Compute support point for the Minkowski difference set along the normal.
    const Real sv1{set1->SupportFunction(rot1.transpose() * normal, sp1)};
    const Real sv2{set2->SupportFunction(-rot2.transpose() * normal, sp2)};
    sp.noalias() = rot1 * sp1 - rot2 * sp2;
    // Update the lower bound and the current best normal vector.
    const Real lb_{(sv1 + sv2) / normal(1)};
    if (lb_ > lb) {
      lb = lb_;
      out.normal = normal;
    }
    ub = UpdateSimplex(sp, sp1, sp2, simplex, out);
    ++out.iter;

#ifndef DONOT_PRINT_DIAGNOSTICS
    io::PrintSolutionDiagnostics(lb, ub, simplex, settings, out);
#endif

    if (lb >= ub * settings.rel_tol) {
      out.status = SolutionStatus::kOptimal;
      break;
    }
    if (out.iter >= settings.max_iter) {
      out.status = SolutionStatus::kMaxIterReached;
      break;
    }
    UpdateNormal(simplex, normal);
  }

  out.growth_dist_lb = -cdist / lb;
  out.growth_dist_ub = -cdist / ub;
  // Transform normal vector.
  out.normal = rot.transpose() * out.normal;

#ifndef DONOT_PRINT_DIAGNOSTICS
  io::PrintDiagnosticsFooter();
#endif

  return out.growth_dist_lb;
}

/**********************************************************
 * GROWTH DISTANCE ALGORITHM FOR 3D CONVEX SETS           *
 **********************************************************/

namespace {

/**
 * @brief Struct containing all the information about the simplex.
 */
struct Simplex {
  /**
   * @brief Simplex points in the aligned coordinates.
   *
   * The simplex points are in CCW order when projected to the x-y plane.
   */
  Matf<3, 3> s;

  /**
   * @brief Normal vector.
   */
  Vec3f n;

  /**
   * @name Convex set support points
   * @brief Support points for the convex sets in the untransformed coordinates.
   */
  ///@{
  Vec3f sp1;
  Vec3f sp2;
  ///@}

  /**
   * @brief Support point for the Minkowski difference set at the current
   * normal, in the aligned coordinates.
   */
  Vec3f sp;

  /**
   * @brief Barycentric coordinates of sp.
   */
  Vec3f bc;

  /**
   * @brief (\f$2 \times\f$) area of the projected simplex (\f$> 0\f$).
   */
  Real area;
};

/**
 * @brief Updates the normal vector using the simplex.
 *
 * @param[in,out] sx Simplex struct.
 */
inline void UpdateNormal(Simplex& sx) {
  // The triangle edges are used to compute the normal vector because the origin
  // may not lie in the triangle interior.
  sx.n(0) = (sx.s(1, 2) - sx.s(1, 0)) * (sx.s(2, 1) - sx.s(2, 0)) -
            (sx.s(2, 2) - sx.s(2, 0)) * (sx.s(1, 1) - sx.s(1, 0));
  sx.n(1) = (sx.s(2, 2) - sx.s(2, 0)) * (sx.s(0, 1) - sx.s(0, 0)) -
            (sx.s(0, 2) - sx.s(0, 0)) * (sx.s(2, 1) - sx.s(2, 0));
  sx.n(2) = -sx.area;
  sx.n.normalize();
}

/**
 * @brief Updates the barycentric coordinates of the origin.
 *
 * @param[in,out] sx  Simplex struct.
 * @param[out]    out Solver output.
 * @return        z-coordinate of the intersection point.
 */
inline Real UpdateOriginCoordinates(Simplex& sx, SolverOutput<3>& out) {
  out.bc(0) = sx.s(0, 1) * sx.s(1, 2) - sx.s(1, 1) * sx.s(0, 2);
  out.bc(1) = sx.s(0, 2) * sx.s(1, 0) - sx.s(1, 2) * sx.s(0, 0);
  out.bc(2) = sx.s(0, 0) * sx.s(1, 1) - sx.s(1, 0) * sx.s(0, 1);
  // The projected (signed) simplex area is guaranteed to be positive. The
  // implementation below is for robustness.
  out.bc.array() = out.bc.array().abs() + kEps * kEps / Real(3.0);
  sx.area = out.bc.sum();
  out.bc = out.bc / sx.area;
  return sx.s.row(2) * out.bc;
}

/**
 * @brief Computes the barycentric coordinates of the support point.
 *
 * @param[in,out] sx Simplex struct.
 */
inline void ComputeSupportCoordinates(Simplex& sx) {
  sx.bc(0) = (sx.s(0, 1) - sx.sp(0)) * (sx.s(1, 2) - sx.sp(1)) -
             (sx.s(1, 1) - sx.sp(1)) * (sx.s(0, 2) - sx.sp(0));
  sx.bc(1) = (sx.s(0, 2) - sx.sp(0)) * (sx.s(1, 0) - sx.sp(1)) -
             (sx.s(1, 2) - sx.sp(1)) * (sx.s(0, 0) - sx.sp(0));
  sx.bc(2) = (sx.s(0, 0) - sx.sp(0)) * (sx.s(1, 1) - sx.sp(1)) -
             (sx.s(1, 0) - sx.sp(1)) * (sx.s(0, 1) - sx.sp(0));
  sx.bc.array() = sx.bc.array() + kEps * kEps / Real(3.0);
  sx.bc = sx.bc / sx.area;
}

/**
 * @brief Updates the simplex and barycentric coordinates and returns the upper
 * bound, given a support point.
 *
 * @param[in,out] sx  Simplex struct.
 * @param[out]    out Solver output.
 * @return        The updated value of the upper bound.
 */
inline Real UpdateSimplex(Simplex& sx, SolverOutput<3>& out) {
  ComputeSupportCoordinates(sx);
  // Replace the exiting simplex point with the support point.
  int exiting_idx{0};
  Real value{1.0};
  for (int i = 0; i < 3; ++i)
    if ((sx.bc(i) > kEps) && (out.bc(i) < sx.bc(i) * value)) {
      value = out.bc(i) / sx.bc(i);
      exiting_idx = i;
    }
  sx.s.col(exiting_idx) = sx.sp;
  out.s1.col(exiting_idx) = sx.sp1;
  out.s2.col(exiting_idx) = sx.sp2;
  return UpdateOriginCoordinates(sx, out);
}

/**
 * @brief Initializes the simplex for the algorithm.
 *
 * @param[in,out] sx  Simplex struct.
 * @param[in,out] out Solver output.
 */
inline void InitializeSimplex(Simplex& sx, SolverOutput<3>& out) {
  sx.n = -Vec3f::UnitZ();

  sx.s.col(0) = out.inradius * Vec3f{0.5, 0.5, 0.0};
  sx.s.col(1) = out.inradius * Vec3f{-0.5, 0.5, 0.0};
  sx.s.col(2) = -out.inradius * Vec3f::UnitY();
  sx.area = Real(1.5) * out.inradius * out.inradius;

  out.bc = Real(1.0 / 3.0) * Vec3f::Ones();
}

}  // namespace

/**
 * @brief Growth distance algorithm for 3D convex sets.
 *
 * @param[in]     set1       Convex set 1.
 * @param[in]     tf1        Rigid body transformation for convex set 1.
 * @param[in]     set2       Convex set 2.
 * @param[in]     tf2        Rigid body transformation for convex set 2.
 * @param[in]     settings   Solver settings.
 * @param[in,out] out        Solver output.
 * @param         warm_start Use previous solver output to warm start current
 *                           solution.
 * @return        (lower bound of) the growth distance.
 */
template <class C1, class C2>
Real GrowthDistance(const C1* set1, const Transform3f& tf1, const C2* set2,
                    const Transform3f& tf2, const SolverSettings& settings,
                    SolverOutput<3>& out, bool warm_start) {
  static_assert((C1::Dimension() == 3) && (C2::Dimension() == 3),
                "Convex sets are not three-dimensional!");

  if (!warm_start) out.inradius = set1->GetInradius() + set2->GetInradius();
  out.iter = 0;

  // Check center distance.
  const Vec3f p12{tf1.block<3, 1>(0, 3) - tf2.block<3, 1>(0, 3)};
  const Real cdist{p12.norm()};
  if (cdist < settings.min_center_dist) return SetZeroOutput(out);

  // Alignment rotation matrix.
  Rot3f rot;
  RotationToZAxis(p12 / cdist, rot);
  // Axis-aligned rotation matrices.
  const Rot3f rot1{rot * tf1.block<3, 3>(0, 0)};
  const Rot3f rot2{rot * tf2.block<3, 3>(0, 0)};

  // Growth distance bounds.
  Real lb{-kInf}, ub{0.0};
  Simplex sx;

  InitializeSimplex(sx, out);
  if (warm_start && (out.status == SolutionStatus::kOptimal)) {
    const Matf<3, 3> s1_{out.s1};
    const Matf<3, 3> s2_{out.s2};
    for (int i = 0; i < 3; ++i)
      if (out.bc(i) > kEps) {
        sx.sp.noalias() = rot1 * s1_.col(i) - rot2 * s2_.col(i);
        if (sx.n.dot(sx.sp - sx.s.col(0)) > Real(0.0)) {
          sx.sp1 = s1_.col(i);
          sx.sp2 = s2_.col(i);
          ub = UpdateSimplex(sx, out);
          UpdateNormal(sx);
        }
      }
  }

#ifndef DONOT_PRINT_DIAGNOSTICS
  io::PrintDiagnosticsHeader(lb, ub, sx.s, settings, out);
#endif

  // Loop invariants (at the start of each iteration):
  //  a) n, ub, and out.bc correspond to s.
  //  b) lb and normal are correctly set for iter >= 1.
  // s1 and s2 (corresponding to nonzero out.bc) are correctly set at the end of
  // the algorithm.
  while (true) {
    // Compute support point for the Minkowski difference set along the normal.
    const Real sv1{set1->SupportFunction(rot1.transpose() * sx.n, sx.sp1)};
    const Real sv2{set2->SupportFunction(-rot2.transpose() * sx.n, sx.sp2)};
    sx.sp.noalias() = rot1 * sx.sp1 - rot2 * sx.sp2;
    // Update the lower bound and the current best normal vector.
    const Real lb_{(sv1 + sv2) / sx.n(2)};
    if (lb_ > lb) {
      lb = lb_;
      out.normal = sx.n;
    }
    ub = UpdateSimplex(sx, out);
    ++out.iter;

#ifndef DONOT_PRINT_DIAGNOSTICS
    io::PrintSolutionDiagnostics(lb, ub, sx.s, settings, out);
#endif

    if (lb >= ub * settings.rel_tol) {
      out.status = SolutionStatus::kOptimal;
      break;
    }
    if (out.iter >= settings.max_iter) {
      out.status = SolutionStatus::kMaxIterReached;
      break;
    }
    UpdateNormal(sx);
  }

  out.growth_dist_lb = -cdist / lb;
  out.growth_dist_ub = -cdist / ub;
  // Transform the normal vector.
  out.normal = rot.transpose() * out.normal;

#ifndef DONOT_PRINT_DIAGNOSTICS
  io::PrintDiagnosticsFooter();
#endif

  return out.growth_dist_lb;
}

/**********************************************************
 * ADDITIONAL UTILITY FUNCTIONS                           *
 **********************************************************/

template <int dim>
SolutionError GetSolutionError(const Transformf<dim>& tf1,
                               const Transformf<dim>& tf2,
                               SolverOutput<dim>& out) {
  SolutionError err;

  const Vecf<dim> p12{tf1.template block<dim, 1>(0, dim) -
                      tf2.template block<dim, 1>(0, dim)};
  const Rotf<dim> rot1{tf1.template block<dim, dim>(0, 0)};
  const Rotf<dim> rot2{tf2.template block<dim, dim>(0, 0)};
  const Vecf<dim> cp12{(rot1 * out.s1 - rot2 * out.s2) * out.bc};

  err.primal_dual_rel_gap =
      std::abs(out.growth_dist_ub / out.growth_dist_lb - Real(1.0));
  err.primal_feas_err = (p12 + cp12 * out.growth_dist_ub).norm();
  return err;
}

}  // namespace dgd

#endif  // DGD_GROWTH_DISTANCE_H_
