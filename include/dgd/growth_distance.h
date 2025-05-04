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
#include "dgd/geometry/convex_set.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/utils.h"

#ifdef DGD_PRINT_DIAGNOSTICS
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
inline Real UpdateSimplex(const Vec2f& sp, const Vec2f& sp1, const Vec2f& sp2,
                          Matf<2, 2>& simplex, SolverOutput<2>& out) {
  const int idx = (sp(0) >= 0);
  simplex.col(idx) = sp;
  out.s1.col(idx) = sp1;
  out.s2.col(idx) = sp2;

  out.bc(0) = (simplex(0, 1) + kEps / Real(2.0));
  out.bc(1) = (-simplex(0, 0) + kEps / Real(2.0));
  out.bc = out.bc / out.bc.sum();

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

  simplex.col(0) = Vec2f(-out.inradius, 0.0);
  simplex.col(1) = Vec2f(out.inradius, 0.0);

  out.bc = Vec2f::Constant(0.5);
}

}  // namespace

/**
 * @brief Growth distance algorithm for 2D convex sets.
 *
 * @attention When using warm-start, the following properties must be ensured:
 * The same out struct must be reused from the previous function call;
 * The out struct must not be used for other pairs of sets in between function
 * calls;
 * The order of set1 and set2 must not be changed.
 *
 * @tparam        collide    If true, performs a boolean collision check.
 * @param[in]     set1,set2  Convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the convex sets.
 * @param[in]     settings   Solver settings.
 * @param[in,out] out        Solver output.
 * @param         warm_start Use previous solver output to warm start current
 *                           solution (default = false).
 * @return        (lower bound of) the growth distance.
 */
template <class C1, class C2, bool collide = false>
Real GrowthDistance(const C1* set1, const Transform2f& tf1, const C2* set2,
                    const Transform2f& tf2, const SolverSettings& settings,
                    SolverOutput<2>& out, bool warm_start = false) {
  static_assert((C1::Dimension() == 2) && (C2::Dimension() == 2),
                "Convex sets are not two-dimensional");

  if (!warm_start) {
    out.inradius = set1->Inradius() + set2->Inradius();
    out.hint2_.n_prev = out.hint1_.n_prev = Vec2f::Zero();
  }
  out.iter = 0;
  const bool normalize{set1->RequireUnitNormal() || set2->RequireUnitNormal()};

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
  // Warm-start.
  if (warm_start && (out.status == SolutionStatus::kOptimal)) {
    const Matf<2, 2> s1_{out.s1};
    const Matf<2, 2> s2_{out.s2};
    for (int i = 0; i < 2; ++i)
      if (out.bc(i) > kEps) {
        sp.noalias() = rot1 * s1_.col(i) - rot2 * s2_.col(i);
        if (normal.dot(sp - simplex.col(0)) > Real(0.0)) {
          ub = UpdateSimplex(sp, s1_.col(i), s2_.col(i), simplex, out);
          UpdateNormal(simplex, normal);
          if (normalize)
            normal.normalize();
          else
            normal = normal / normal.lpNorm<Eigen::Infinity>();
        }
      }
  }

#ifdef DGD_PRINT_DIAGNOSTICS
  io::PrintDiagnosticsHeader(lb, ub, simplex, settings, out);
#endif

  // Loop invariants (at the start of each iteration):
  //  a) normal, ub, and bc correspond to simplex.
  //  b) lb and out.normal are correctly set for iter >= 1.
  // s1 and s2 (corresponding to nonzero bc) are correctly set at the end of the
  // algorithm.
  while (true) {
    // Compute support point for the Minkowski difference set along the normal.
    const Real sv1{
        set1->SupportFunction(rot1.transpose() * normal, sp1, &out.hint1_)};
    const Real sv2{
        set2->SupportFunction(-rot2.transpose() * normal, sp2, &out.hint2_)};
    sp.noalias() = rot1 * sp1 - rot2 * sp2;
    // Update the lower bound and the current best normal vector.
    const Real lb_{(sv1 + sv2) / normal(1)};
    if (lb_ > lb) {
      lb = lb_;
      out.normal = normal;
    }
    ub = UpdateSimplex(sp, sp1, sp2, simplex, out);
    ++out.iter;

#ifdef DGD_PRINT_DIAGNOSTICS
    io::PrintSolutionDiagnostics(lb, ub, simplex, settings, out);
#endif

    if constexpr (collide) {
      // Perform collision check.
      if (lb > -cdist) {
        // No collision.
        out.growth_dist_lb = -cdist / lb;
        out.normal = rot.transpose() * out.normal;
        out.status = SolutionStatus::kOptimal;
        return -1.0;
      } else if (ub <= -cdist) {
        // Collision.
        out.growth_dist_ub = -cdist / ub;
        out.status = SolutionStatus::kOptimal;
        return 1.0;
      }
    } else {
      // Check primal-dual gap.
      if (lb >= ub * settings.rel_tol) {
        out.status = SolutionStatus::kOptimal;
        break;
      }
    }
    if (out.iter >= settings.max_iter) {
      out.status = SolutionStatus::kMaxIterReached;
      break;
    }
    UpdateNormal(simplex, normal);
    if (normalize)
      normal.normalize();
    else
      normal = normal / normal.lpNorm<Eigen::Infinity>();
  }

  out.growth_dist_lb = -cdist / lb;
  out.growth_dist_ub = -cdist / ub;
  // Transform normal vector to world frame.
  out.normal = rot.transpose() * out.normal;

#ifdef DGD_PRINT_DIAGNOSTICS
  io::PrintDiagnosticsFooter();
#endif

  return out.growth_dist_lb;
}

/**********************************************************
 * GROWTH DISTANCE ALGORITHM FOR 3D CONVEX SETS           *
 **********************************************************/

namespace {

/**
 * @brief Struct containing all the information about the inner and outer
 * polyhedral approximations.
 */
struct Approximation {
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
   * @brief Support points for the convex sets in the local coordinates.
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
 * @param[in,out] a Approximation struct.
 */
inline void UpdateNormal(Approximation& a) {
  // The triangle edges are used to compute the normal vector because the origin
  // may not lie in the triangle interior.
  a.n(0) = (a.s(1, 2) - a.s(1, 0)) * (a.s(2, 1) - a.s(2, 0)) -
           (a.s(2, 2) - a.s(2, 0)) * (a.s(1, 1) - a.s(1, 0));
  a.n(1) = (a.s(2, 2) - a.s(2, 0)) * (a.s(0, 1) - a.s(0, 0)) -
           (a.s(0, 2) - a.s(0, 0)) * (a.s(2, 1) - a.s(2, 0));
  a.n(2) = -a.area;
}

/**
 * @brief Updates the barycentric coordinates of the origin.
 *
 * @param[in,out] a   Approximation struct.
 * @param[out]    out Solver output.
 * @return        z-coordinate of the intersection point.
 */
inline Real UpdateOriginCoordinates(Approximation& a, SolverOutput<3>& out) {
  out.bc(0) = a.s(0, 1) * a.s(1, 2) - a.s(1, 1) * a.s(0, 2);
  out.bc(1) = a.s(0, 2) * a.s(1, 0) - a.s(1, 2) * a.s(0, 0);
  out.bc(2) = a.s(0, 0) * a.s(1, 1) - a.s(1, 0) * a.s(0, 1);
  // The projected (signed) simplex area is guaranteed to be positive. The
  // implementation below is for robustness.
  out.bc.array() = out.bc.array().abs() + kEps * kEps / Real(3.0);
  a.area = out.bc.sum();
  out.bc = out.bc / a.area;
  return a.s.row(2) * out.bc;
}

/**
 * @brief Computes the barycentric coordinates of the support point.
 *
 * @param[in,out] a Approximation struct.
 */
inline void ComputeSupportCoordinates(Approximation& a) {
  a.bc(0) = (a.s(0, 1) - a.sp(0)) * (a.s(1, 2) - a.sp(1)) -
            (a.s(1, 1) - a.sp(1)) * (a.s(0, 2) - a.sp(0));
  a.bc(1) = (a.s(0, 2) - a.sp(0)) * (a.s(1, 0) - a.sp(1)) -
            (a.s(1, 2) - a.sp(1)) * (a.s(0, 0) - a.sp(0));
  a.bc(2) = (a.s(0, 0) - a.sp(0)) * (a.s(1, 1) - a.sp(1)) -
            (a.s(1, 0) - a.sp(1)) * (a.s(0, 1) - a.sp(0));
  a.bc.array() = a.bc.array() + kEps * kEps / Real(3.0);
  a.bc = a.bc / a.area;
}

/**
 * @brief Updates the simplex and barycentric coordinates and returns the upper
 * bound, given a support point.
 *
 * @param[in,out] a   Approximation struct.
 * @param[out]    out Solver output.
 * @return        The updated value of the upper bound.
 */
inline Real UpdateSimplex(Approximation& a, SolverOutput<3>& out) {
  ComputeSupportCoordinates(a);
  // Compute one iteration of the simplex algorithm.
  int exiting_idx{0};
  Real value{1.0};
  for (int i = 0; i < 3; ++i)
    if ((a.bc(i) > kEps) && (out.bc(i) < a.bc(i) * value)) {
      value = out.bc(i) / a.bc(i);
      exiting_idx = i;
    }
  // Replace the exiting simplex point with the support point.
  a.s.col(exiting_idx) = a.sp;
  out.s1.col(exiting_idx) = a.sp1;
  out.s2.col(exiting_idx) = a.sp2;
  return UpdateOriginCoordinates(a, out);
}

/**
 * @brief Initializes the polyhedral approximation for the algorithm.
 *
 * @param[in,out] a   Approximation struct.
 * @param[in,out] out Solver output.
 */
inline void InitializeApproximation(Approximation& a, SolverOutput<3>& out) {
  a.n = -Vec3f::UnitZ();

  a.s.col(0) = out.inradius * Vec3f(0.5, 0.5, 0.0);
  a.s.col(1) = out.inradius * Vec3f(-0.5, 0.5, 0.0);
  a.s.col(2) = Vec3f(0.0, -out.inradius, 0.0);
  a.area = Real(1.5) * out.inradius * out.inradius;

  out.bc = Vec3f::Constant(Real(1.0 / 3.0));
}

}  // namespace

/**
 * @brief Growth distance algorithm for 3D convex sets.
 *
 * @tparam        collide    If true, performs a boolean collision check.
 * @param[in]     set1,set2  Convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the convex sets.
 * @param[in]     settings   Solver settings.
 * @param[in,out] out        Solver output.
 * @param         warm_start Use previous solver output to warm start current
 *                           solution (default = false).
 * @return        (lower bound of) the growth distance.
 */
template <class C1, class C2, bool collide = false>
Real GrowthDistance(const C1* set1, const Transform3f& tf1, const C2* set2,
                    const Transform3f& tf2, const SolverSettings& settings,
                    SolverOutput<3>& out, bool warm_start = false) {
  static_assert((C1::Dimension() == 3) && (C2::Dimension() == 3),
                "Convex sets are not three-dimensional");

  if (!warm_start) {
    out.inradius = set1->Inradius() + set2->Inradius();
    out.hint2_.n_prev = out.hint1_.n_prev = Vec3f::Zero();
  }
  out.iter = 0;
  const bool normalize{set1->RequireUnitNormal() || set2->RequireUnitNormal()};

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
  Approximation a;

  InitializeApproximation(a, out);
  // Warm-start.
  if (warm_start && (out.status == SolutionStatus::kOptimal)) {
    const Matf<3, 3> s1_{out.s1};
    const Matf<3, 3> s2_{out.s2};
    for (int i = 0; i < 3; ++i)
      if (out.bc(i) > kEps) {
        a.sp.noalias() = rot1 * s1_.col(i) - rot2 * s2_.col(i);
        if (a.n.dot(a.sp - a.s.col(0)) > Real(0.0)) {
          a.sp1 = s1_.col(i);
          a.sp2 = s2_.col(i);
          ub = UpdateSimplex(a, out);
          UpdateNormal(a);
          if (normalize)
            a.n.normalize();
          else
            a.n = a.n / a.n.lpNorm<Eigen::Infinity>();
        }
      }
  }

#ifdef DGD_PRINT_DIAGNOSTICS
  io::PrintDiagnosticsHeader(lb, ub, a.s, settings, out);
#endif

  // Loop invariants (at the start of each iteration):
  //  a) n, ub, and out.bc correspond to s.
  //  b) lb and normal are correctly set for iter >= 1.
  // s1 and s2 (corresponding to nonzero out.bc) are correctly set at the end of
  // the algorithm.
  while (true) {
    // Compute support point for the Minkowski difference set along the normal.
    const Real sv1{
        set1->SupportFunction(rot1.transpose() * a.n, a.sp1, &out.hint1_)};
    const Real sv2{
        set2->SupportFunction(-rot2.transpose() * a.n, a.sp2, &out.hint2_)};
    a.sp.noalias() = rot1 * a.sp1 - rot2 * a.sp2;
    // Update the lower bound and the current best normal vector.
    const Real lb_{(sv1 + sv2) / a.n(2)};
    if (lb_ > lb) {
      lb = lb_;
      out.normal = a.n;
    }
    ub = UpdateSimplex(a, out);
    ++out.iter;

#ifdef DGD_PRINT_DIAGNOSTICS
    io::PrintSolutionDiagnostics(lb, ub, a.s, settings, out);
#endif

    if constexpr (collide) {
      // Perform collision check.
      if (lb > -cdist) {
        // No collision.
        out.growth_dist_lb = -cdist / lb;
        out.normal = rot.transpose() * out.normal;
        out.status = SolutionStatus::kOptimal;
        return -1.0;
      } else if (ub <= -cdist) {
        // Collision.
        out.growth_dist_ub = -cdist / ub;
        out.status = SolutionStatus::kOptimal;
        return 1.0;
      }
    } else {
      // Check primal-dual gap.
      if (lb >= ub * settings.rel_tol) {
        out.status = SolutionStatus::kOptimal;
        break;
      }
    }
    if (out.iter >= settings.max_iter) {
      out.status = SolutionStatus::kMaxIterReached;
      break;
    }
    UpdateNormal(a);
    if (normalize)
      a.n.normalize();
    else
      a.n = a.n / a.n.lpNorm<Eigen::Infinity>();
  }

  out.growth_dist_lb = -cdist / lb;
  out.growth_dist_ub = -cdist / ub;
  // Transform the normal vector to world frame.
  out.normal = rot.transpose() * out.normal;

#ifdef DGD_PRINT_DIAGNOSTICS
  io::PrintDiagnosticsFooter();
#endif

  return out.growth_dist_lb;
}

/**********************************************************
 * BOOLEAN COLLISION DETECTION ALGORITHM                  *
 **********************************************************/

/**
 * @brief Collision detection algorithm for 2D and 3D convex sets.
 *
 * The function returns true if the centers coincide or if the sets intersect;
 * false if a separating plane has been found or if the maximum number of
 * iterations have been reached.
 *
 * @param[in]     set1,set2  Convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the convex sets.
 * @param[in]     settings   Solver settings.
 * @param[in,out] out        Solver output.
 * @param         warm_start Use previous solver output to warm start current
 *                           solution (default = false).
 * @return        true, if the convex sets are colliding.
 */
template <class C1, class C2, int dim>
inline bool CollisionCheck(const C1* set1, const Transformf<dim>& tf1,
                           const C2* set2, const Transformf<dim>& tf2,
                           const SolverSettings& settings,
                           SolverOutput<dim>& out, bool warm_start = false) {
  const Real gd{GrowthDistance<C1, C2, true>(set1, tf1, set2, tf2, settings,
                                             out, warm_start)};
  return ((out.status == SolutionStatus::kCoincidentCenters) ||
          (out.status == SolutionStatus::kOptimal && gd > 0.0));
}

/**********************************************************
 * ADDITIONAL UTILITY FUNCTIONS                           *
 **********************************************************/

/**
 * @brief Computes an optimal solution for the growth distance problem.
 *
 * @attention Solver status must be kOptimal or kCoincidentCenters.
 *
 * @attention For the collision detection problem, out.s1 and out.s2 might not
 * correspond to the simplex in the Minkowski difference set. So, this function
 * might not return the intersection point.
 *
 * @tparam dim  Dimension of the convex sets.
 * @param  tf1  Rigid body transformations for convex set 1.
 * @param  tf2  (Unused) Rigid body transformations for convex set 2.
 * @param  out  Solver output.
 * @param  zopt Optimal solution.
 */
template <int dim>
void GetOptimalSolution(const Transformf<dim>& tf1,
                        const Transformf<dim>& /*tf2*/,
                        const SolverOutput<dim>& out, Vecf<dim>& zopt) {
  const Vecf<dim> p1{tf1.template block<dim, 1>(0, dim)};
  const Rotf<dim> rot1{tf1.template block<dim, dim>(0, 0)};
  zopt = p1 + out.growth_dist_ub * rot1 * out.s1 * out.bc;
}

/**
 * @brief Gets the primal-dual relative gap and the primal feasibility error.
 *
 * @tparam dim       Dimension of the convex sets.
 * @param  set1,set2 Convex Sets.
 * @param  tf1,tf2   Rigid body transformations for the convex sets.
 * @param  out       Solver output.
 * @return Solution error.
 */
template <int dim>
SolutionError GetSolutionError(const ConvexSet<dim>* set1,
                               const Transformf<dim>& tf1,
                               const ConvexSet<dim>* set2,
                               const Transformf<dim>& tf2,
                               const SolverOutput<dim>& out) {
  SolutionError err;
  if (out.status != SolutionStatus::kOptimal &&
      out.status != SolutionStatus::kCoincidentCenters) {
    err.prim_dual_gap = -1.0;
    err.prim_feas_err = -1.0;
    return err;
  }

  const Vecf<dim> p1{tf1.template block<dim, 1>(0, dim)};
  const Vecf<dim> p2{tf2.template block<dim, 1>(0, dim)};
  const Rotf<dim> rot1{tf1.template block<dim, dim>(0, 0)};
  const Rotf<dim> rot2{tf2.template block<dim, dim>(0, 0)};
  const Vecf<dim> cp1{p1 + out.growth_dist_ub * rot1 * out.s1 * out.bc};
  const Vecf<dim> cp2{p2 + out.growth_dist_ub * rot2 * out.s2 * out.bc};

  Vecf<dim> sp;
  const Real sv1{set1->SupportFunction(rot1.transpose() * out.normal, sp)};
  const Real sv2{set2->SupportFunction(-rot2.transpose() * out.normal, sp)};
  const Real lb{(p2 - p1).dot(out.normal) / (sv1 + sv2)};

  err.prim_dual_gap = std::abs(out.growth_dist_ub / lb - 1.0);
  err.prim_feas_err = (cp1 - cp2).norm();
  return err;
}

/**
 * @brief Asserts the collision status of the two convex sets.
 *
 * @tparam dim       Dimension of the convex sets.
 * @param  set1,set2 Convex Sets.
 * @param  tf1,tf2   Rigid body transformations for the convex sets.
 * @param  out       Solver output.
 * @param  collision Output of the CollisionCheck function.
 * @return true, if the collision status is correct; false otherwise.
 */
template <int dim>
bool AssertCollision(const ConvexSet<dim>* set1, const Transformf<dim>& tf1,
                     const ConvexSet<dim>* set2, const Transformf<dim>& tf2,
                     const SolverOutput<dim>& out, bool collision) {
  if (out.status == SolutionStatus::kCoincidentCenters) {
    return collision;
  } else if (out.status == SolutionStatus::kMaxIterReached) {
    return !collision;
  }

  const Vecf<dim> p1{tf1.template block<dim, 1>(0, dim)};
  const Vecf<dim> p2{tf2.template block<dim, 1>(0, dim)};
  const Rotf<dim> rot1{tf1.template block<dim, dim>(0, 0)};
  const Rotf<dim> rot2{tf2.template block<dim, dim>(0, 0)};

  if (collision) {
    // const Vecf<dim> cp1{p1 + out.growth_dist_ub * rot1 * out.s1 * out.bc};
    // const Vecf<dim> cp2{p2 + out.growth_dist_ub * rot2 * out.s2 * out.bc};
    // return ((cp1 - cp2).norm() < kEpsSqrt) && (out.growth_dist_ub <= 1.0);
    return out.growth_dist_ub <= 1.0;
  } else {
    Vecf<dim> sp;
    const Real sv1{set1->SupportFunction(rot1.transpose() * out.normal, sp)};
    const Real sv2{set2->SupportFunction(-rot2.transpose() * out.normal, sp)};
    return (out.growth_dist_lb > 1.0) &&
           (p2.dot(out.normal) - sv2 > p1.dot(out.normal) + sv1);
  }
}

}  // namespace dgd

#endif  // DGD_GROWTH_DISTANCE_H_
