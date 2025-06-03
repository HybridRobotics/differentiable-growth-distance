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
 * @brief Growth distance algorithm for 2D and 3D convex sets.
 */

#ifndef DGD_GROWTH_DISTANCE_H_
#define DGD_GROWTH_DISTANCE_H_

#include "dgd/data_types.h"
#include "dgd/debug.h"
#include "dgd/geometry/convex_set.h"
#include "dgd/output.h"
#include "dgd/settings.h"

#ifdef DGD_COMPUTE_COLLISION_INTERSECTION

#define DGD_INITIALIZE_SET_SIMPLICES(...) \
  dgd::InitializeSetSimplices(__VA_ARGS__)
#define DGD_COMPUTE_PRIMAL_SOLUTION(...) dgd::ComputePrimalSolution(__VA_ARGS__)

#else

#define DGD_INITIALIZE_SET_SIMPLICES(...) (void)0
#define DGD_COMPUTE_PRIMAL_SOLUTION(...) (void)0

#endif

namespace dgd {

/**
 * Common utility functions.
 */

namespace {

// Sets zero output when the centers of the convex sets coincide.
template <int dim>
inline Real SetZeroOutput(Output<dim>& out) {
  out.normal = Vecr<dim>::Zero();
  out.growth_dist_ub = out.growth_dist_lb = Real(0.0);
  out.z2 = out.z1 = Vecr<dim>::Zero();
  out.status = SolutionStatus::CoincidentCenters;
  return Real(0.0);
}

// Computes the primal solution in the world frame of reference.
template <int dim>
inline void ComputePrimalSolution(const Transformr<dim>& tf1,
                                  const Transformr<dim>& tf2,
                                  Output<dim>& out) {
  out.z1.noalias() = tf1.template block<dim, dim>(0, 0) * out.s1 * out.bc +
                     tf1.template block<dim, 1>(0, dim);
  out.z2.noalias() = tf2.template block<dim, dim>(0, 0) * out.s2 * out.bc +
                     tf2.template block<dim, 1>(0, dim);
}

}  // namespace

/**
 * 2D utility functions.
 */

namespace {

// Sets rot such that rot * n = Vec2r::UnitY().
inline void RotationToZAxis(const Vec2r& n, Rotation2r& rot) {
  rot(0, 0) = n(1);
  rot(1, 0) = n(0);
  rot(0, 1) = -n(0);
  rot(1, 1) = n(1);
}

// Updates the normal vector using the simplex.
inline void UpdateNormal(const Matr<2, 2>& simplex, Vec2r& normal,
                         bool normalize) {
  normal(0) = simplex(1, 1) - simplex(1, 0);
  normal(1) = simplex(0, 0) - simplex(0, 1) -
              kEps;  // Small constant added for dual feasibility.
  if (normalize) {
    normal.normalize();
  } else {
    normal /= normal.lpNorm<Eigen::Infinity>();
  }
}

// Updates the simplex and the barycentric coordinates and returns the upper
// bound, given a support point.
inline Real UpdateSimplex(const Vec2r& sp, const Vec2r& sp1, const Vec2r& sp2,
                          Matr<2, 2>& simplex, Output<2>& out) {
  const int idx = (sp(0) >= 0);
  simplex.col(idx) = sp;
  out.s1.col(idx) = sp1;
  out.s2.col(idx) = sp2;

  out.bc(0) = (simplex(0, 1) + kEps / Real(2.0));
  out.bc(1) = (-simplex(0, 0) + kEps / Real(2.0));
  out.bc /= out.bc.sum();

  return simplex.row(1) * out.bc;
}

// Initializes the simplex for the algorithm.
inline void InitializeSimplex(Vec2r& normal, Matr<2, 2>& simplex,
                              Output<2>& out) {
  normal = -Vec2r::UnitY();

  simplex.col(0) = Vec2r(-out.inradius, 0.0);
  simplex.col(1) = Vec2r(out.inradius, 0.0);

  out.bc = Vec2r::Constant(0.5);
}

// Initializes the convex set simplices.
template <class C1, class C2>
inline void InitializeSetSimplices(const C1* set1, const Rotation2r& rot1,
                                   const C2* set2, const Rotation2r& rot2,
                                   Output<2>& out) {
  const Real r1{set1->inradius()}, r2{set2->inradius()};

  out.s1.col(1).noalias() = r1 * rot1.transpose().col(0);
  out.s1.col(0) = -out.s1.col(1);

  out.s2.col(0).noalias() = r2 * rot2.transpose().col(0);
  out.s2.col(1) = -out.s2.col(0);
}

}  // namespace

/**
 * Growth distance algorithm for 2D convex sets.
 */

/**
 * @brief Growth distance algorithm for 2D convex sets.
 *
 * @attention When using warm-start, the following properties must be ensured:
 * The same out struct must be reused from the previous function call;
 * The out struct must not be used for other pairs of sets in between function
 * calls;
 * The order of set1 and set2 must not be changed.
 *
 * @tparam        detect_collision If true, performs a boolean collision check.
 * @param[in]     set1,set2        Convex sets.
 * @param[in]     tf1,tf2          Rigid body transformations for the sets.
 * @param[in]     settings         Solver settings.
 * @param[in,out] out              Solver output.
 * @param         warm_start       Use previous solver output to warm start
 * current current solution (default = false).
 * @return        (lower bound of) the growth distance.
 */
template <class C1, class C2, bool detect_collision = false>
Real GrowthDistance(const C1* set1, const Transform2r& tf1, const C2* set2,
                    const Transform2r& tf2, const Settings& settings,
                    Output<2>& out, bool warm_start = false) {
  static_assert((C1::dimension() == 2) && (C2::dimension() == 2),
                "Convex sets are not two-dimensional");

  if (!warm_start) {
    out.hint2_.n_prev = out.hint1_.n_prev = Vec2r::Zero();
    out.inradius = set1->inradius() + set2->inradius();
  }
  int iter{0};
  const bool normalize{set1->RequireUnitNormal() || set2->RequireUnitNormal()};

  // Check center distance.
  const Vec2r p12{tf1.block<2, 1>(0, 2) - tf2.block<2, 1>(0, 2)};
  const Real cdist{p12.norm()};
  if (cdist < settings.min_center_dist) return SetZeroOutput(out);

  // Alignment rotation matrix.
  Rotation2r rot;
  RotationToZAxis(p12 / cdist, rot);
  const Rotation2r rot1{rot * tf1.block<2, 2>(0, 0)};
  const Rotation2r rot2{rot * tf2.block<2, 2>(0, 0)};

  // Growth distance bounds.
  Real lb{-kInf}, ub{0.0};
  // Normal vector, suppport points, and the simplex matrix.
  Vec2r normal, sp1, sp2, sp;
  Matr<2, 2> simplex;

  InitializeSimplex(normal, simplex, out);
  if (warm_start && (out.status == SolutionStatus::Optimal)) {
    // Warm-start.
    const Matr<2, 2> s1_c{out.s1};
    const Matr<2, 2> s2_c{out.s2};
    DGD_INITIALIZE_SET_SIMPLICES(set1, rot1, set2, rot2, out);
    for (int i = 0; i < 2; ++i) {
      if (out.bc(i) > kEps) {
        sp.noalias() = rot1 * s1_c.col(i) - rot2 * s2_c.col(i);
        if (normal.dot(sp - simplex.col(0)) > Real(0.0)) {
          ub = UpdateSimplex(sp, s1_c.col(i), s2_c.col(i), simplex, out);
          UpdateNormal(simplex, normal, normalize);
        }
      }
    }
  } else {
    DGD_INITIALIZE_SET_SIMPLICES(set1, rot1, set2, rot2, out);
  }

  DGD_PRINT_DEBUG_HEADER(iter, lb, ub, simplex, settings, out);

  Real gd;
  while (true) {
    // Compute support point for the Minkowski difference set along the normal.
    const Real sv1{
        set1->SupportFunction(rot1.transpose() * normal, sp1, &out.hint1_)};
    const Real sv2{
        set2->SupportFunction(-rot2.transpose() * normal, sp2, &out.hint2_)};
    sp.noalias() = rot1 * sp1 - rot2 * sp2;
    // Update the lower bound and the current best normal vector.
    const Real lb_n{(sv1 + sv2) / normal(1)};
    if (lb_n > lb) {
      lb = lb_n;
      out.normal = normal;
    }
    ub = UpdateSimplex(sp, sp1, sp2, simplex, out);
    ++iter;

    DGD_PRINT_DEBUG_ITERATION(iter, lb, ub, simplex, settings, out);

    // Termination criteria.
    if constexpr (detect_collision) {
      // Perform collision check.
      if (lb > -cdist) {
        // No collision.
        out.normal = rot.transpose() * out.normal;
        out.growth_dist_lb = -cdist / lb;
        out.status = SolutionStatus::Optimal;
        gd = -1.0;
        break;
      } else if (ub <= -cdist) {
        // Collision.
        out.growth_dist_ub = -cdist / ub;
        DGD_COMPUTE_PRIMAL_SOLUTION(tf1, tf2, out);
        out.status = SolutionStatus::Optimal;
        gd = 1.0;
        break;
      }
    } else {
      // Check primal-dual gap.
      if (lb >= ub * settings.rel_tol) {
        // Transform normal vector to world frame.
        out.normal = rot.transpose() * out.normal;
        out.growth_dist_lb = -cdist / lb;
        out.growth_dist_ub = -cdist / ub;
        ComputePrimalSolution(tf1, tf2, out);
        out.status = SolutionStatus::Optimal;
        gd = out.growth_dist_lb;
        break;
      }
    }
    if (out.iter >= settings.max_iter) {
      out.status = SolutionStatus::MaxIterReached;
      gd = -cdist / lb;
      break;
    }

    UpdateNormal(simplex, normal, normalize);
  }
  out.iter = iter;

  DGD_PRINT_DEBUG_FOOTER();

  return gd;
}

/**
 * 3D utility functions.
 */

namespace {

// Sets rot such that rot * n = Vec3r::UnitZ().
inline void RotationToZAxis(const Vec3r& n, Rotation3r& rot) {
  Vec3r axis{n + Vec3r::UnitZ()};
  const Real norm{axis.norm()};
  if (norm > kEps) {
    axis /= norm;
    rot.noalias() =
        Real(2.0) * axis * axis.transpose() - Rotation3r::Identity();
  } else {
    rot = Vec3r(1.0, -1.0, -1.0).asDiagonal();
  }
}

// Temporary variables for the growth distance solver.
struct SolverContext {
  // Simplex points in the aligned coordinates.
  // The simplex points are in CCW order when projected to the x-y plane.
  Matr<3, 3> s;

  // Set support points in local coordinates.
  Vec3r sp1, sp2;

  // Support point for the Minkowski difference set at the current normal,
  // in the aligned coordinates.
  Vec3r sp;

  // Barycentric coordinates of sp.
  Vec3r bc;

  // Normal vector.
  Vec3r n;

  // Twice the area of the projected simplex (> 0).
  Real area;
};

// Updates the normal vector using the simplex.
inline void UpdateNormal(SolverContext& c, bool normalize) {
  // The triangle edges are used to compute the normal vector because the origin
  // may not lie in the triangle interior.
  // c.n = (c.s.col(2) - c.s.col(0)).cross(c.s.col(1) - c.s.col(0));
  c.n(0) = (c.s(1, 2) - c.s(1, 0)) * (c.s(2, 1) - c.s(2, 0)) -
           (c.s(2, 2) - c.s(2, 0)) * (c.s(1, 1) - c.s(1, 0));
  c.n(1) = (c.s(2, 2) - c.s(2, 0)) * (c.s(0, 1) - c.s(0, 0)) -
           (c.s(0, 2) - c.s(0, 0)) * (c.s(2, 1) - c.s(2, 0));
  c.n(2) = -c.area;
  if (normalize) {
    c.n.normalize();
  } else {
    c.n /= c.n.lpNorm<Eigen::Infinity>();
  }
}

// Updates the barycentric coordinates of the origin, and returns the
// z-coordinate of the intersection point.
inline Real UpdateOriginCoordinates(SolverContext& c, Output<3>& out) {
  out.bc(0) = c.s(0, 1) * c.s(1, 2) - c.s(1, 1) * c.s(0, 2);
  out.bc(1) = c.s(0, 2) * c.s(1, 0) - c.s(1, 2) * c.s(0, 0);
  out.bc(2) = c.s(0, 0) * c.s(1, 1) - c.s(1, 0) * c.s(0, 1);
  // The projected (signed) simplex area is guaranteed to be positive. The
  // implementation below is for robustness.
  out.bc.array() = out.bc.array() + kEps * kEps / Real(3.0);
  c.area = out.bc.sum();
  out.bc /= c.area;
  return c.s.row(2) * out.bc;
}

// Computes the barycentric coordinates of the support point.
inline void ComputeSupportCoordinates(SolverContext& c) {
  c.bc(0) = (c.s(0, 1) - c.sp(0)) * (c.s(1, 2) - c.sp(1)) -
            (c.s(1, 1) - c.sp(1)) * (c.s(0, 2) - c.sp(0));
  c.bc(1) = (c.s(0, 2) - c.sp(0)) * (c.s(1, 0) - c.sp(1)) -
            (c.s(1, 2) - c.sp(1)) * (c.s(0, 0) - c.sp(0));
  c.bc(2) = (c.s(0, 0) - c.sp(0)) * (c.s(1, 1) - c.sp(1)) -
            (c.s(1, 0) - c.sp(1)) * (c.s(0, 1) - c.sp(0));
  c.bc.array() += kEps * kEps / Real(3.0);
  c.bc /= c.area;
}

// Updates the simplex and barycentric coordinates and returns the upper bound,
// given a support point.
inline Real UpdateSimplex(SolverContext& c, Output<3>& out) {
  ComputeSupportCoordinates(c);
  // Compute one iteration of the simplex algorithm.
  int exiting_idx{0};
  Real value{1.0};
  for (int i = 0; i < 3; ++i) {
    if ((c.bc(i) > kEps) && (out.bc(i) < c.bc(i) * value)) {
      value = out.bc(i) / c.bc(i);
      exiting_idx = i;
    }
  }
  // Replace the exiting simplex point with the support point.
  c.s.col(exiting_idx) = c.sp;
  out.s1.col(exiting_idx) = c.sp1;
  out.s2.col(exiting_idx) = c.sp2;
  return UpdateOriginCoordinates(c, out);
}

// Initializes the simplex for the algorithm.
inline void InitializeSimplex(SolverContext& c, Output<3>& out) {
  c.n = -Vec3r::UnitZ();

  c.s.col(0) = out.inradius * Vec3r(0.5, 0.5, 0.0);
  c.s.col(1) = out.inradius * Vec3r(-0.5, 0.5, 0.0);
  c.s.col(2) = Vec3r(0.0, -out.inradius, 0.0);
  c.area = Real(1.5) * out.inradius * out.inradius;

  out.bc = Vec3r::Constant(Real(1.0 / 3.0));
}

// Initializes the convex set simplices.
template <class C1, class C2>
inline void InitializeSetSimplices(const C1* set1, const Rotation3r& rot1,
                                   const C2* set2, const Rotation3r& rot2,
                                   Output<3>& out) {
  const Real r1{set1->inradius()}, r2{set2->inradius()};

  out.s1.col(1).noalias() = r1 * rot1.transpose() * Vec3r(-0.5, 0.5, 0.0);
  out.s1.col(2).noalias() = rot1.transpose() * Vec3r(0.0, -r1, 0.0);
  out.s1.col(0) = -(out.s1.col(1) + out.s1.col(2));

  out.s2.col(1).noalias() = r2 * rot2.transpose() * Vec3r(0.5, -0.5, 0.0);
  out.s2.col(2).noalias() = rot2.transpose() * Vec3r(0.0, r2, 0.0);
  out.s2.col(0) = -(out.s2.col(1) + out.s2.col(2));
}

}  // namespace

/**
 * Growth distance algorithm for 3D convex sets.
 */

/**
 * @brief Growth distance algorithm for 3D convex sets.
 *
 * @attention When using warm-start, the following properties must be ensured:
 * The same out struct must be reused from the previous function call;
 * The out struct must not be used for other pairs of sets in between function
 * calls;
 * The order of set1 and set2 must not be changed.
 *
 * @tparam        detect_collision If true, performs a boolean collision check.
 * @param[in]     set1,set2        Convex sets.
 * @param[in]     tf1,tf2          Rigid body transformations for the sets.
 * @param[in]     settings         Solver settings.
 * @param[in,out] out              Solver output.
 * @param         warm_start       Use previous solver output to warm start
 *                                 current solution (default = false).
 * @return        (lower bound of) the growth distance.
 */
template <class C1, class C2, bool detect_collision = false>
Real GrowthDistance(const C1* set1, const Transform3r& tf1, const C2* set2,
                    const Transform3r& tf2, const Settings& settings,
                    Output<3>& out, bool warm_start = false) {
  static_assert((C1::dimension() == 3) && (C2::dimension() == 3),
                "Convex sets are not three-dimensional");

  if (!warm_start) {
    out.hint2_.n_prev = out.hint1_.n_prev = Vec3r::Zero();
    out.inradius = set1->inradius() + set2->inradius();
  }
  int iter{0};
  const bool normalize{set1->RequireUnitNormal() || set2->RequireUnitNormal()};

  // Check center distance.
  const Vec3r p12{tf1.block<3, 1>(0, 3) - tf2.block<3, 1>(0, 3)};
  const Real cdist{p12.norm()};
  if (cdist < settings.min_center_dist) return SetZeroOutput(out);

  // Alignment rotation matrix.
  Rotation3r rot;
  RotationToZAxis(p12 / cdist, rot);
  const Rotation3r rot1{rot * tf1.block<3, 3>(0, 0)};
  const Rotation3r rot2{rot * tf2.block<3, 3>(0, 0)};

  // Growth distance bounds.
  Real lb{-kInf}, ub{0.0};
  SolverContext c;

  InitializeSimplex(c, out);
  if (warm_start && (out.status == SolutionStatus::Optimal)) {
    // Warm-start.
    const Matr<3, 3> s1_c{out.s1};
    const Matr<3, 3> s2_c{out.s2};
    DGD_INITIALIZE_SET_SIMPLICES(set1, rot1, set2, rot2, out);
    for (int i = 0; i < 3; ++i) {
      if (out.bc(i) > kEps) {
        c.sp.noalias() = rot1 * s1_c.col(i) - rot2 * s2_c.col(i);
        if (c.n.dot(c.sp - c.s.col(0)) > Real(0.0)) {
          c.sp1 = s1_c.col(i);
          c.sp2 = s2_c.col(i);
          ub = UpdateSimplex(c, out);
          UpdateNormal(c, normalize);
        }
      }
    }
  } else {
    DGD_INITIALIZE_SET_SIMPLICES(set1, rot1, set2, rot2, out);
  }

  DGD_PRINT_DEBUG_HEADER(iter, lb, ub, c.s, settings, out);

  // Loop invariants (at the start of each iteration):
  //  a) n, ub, and out.bc correspond to s.
  //  b) lb and normal are correctly set for iter >= 1.
  // s1 and s2 (corresponding to nonzero out.bc) are correctly set at the end of
  // the algorithm.
  Real gd;
  while (true) {
    // Compute support point for the Minkowski difference set along the normal.
    const Real sv1{
        set1->SupportFunction(rot1.transpose() * c.n, c.sp1, &out.hint1_)};
    const Real sv2{
        set2->SupportFunction(-rot2.transpose() * c.n, c.sp2, &out.hint2_)};
    c.sp.noalias() = rot1 * c.sp1 - rot2 * c.sp2;
    // Update the lower bound and the current best normal vector.
    const Real lb_n{(sv1 + sv2) / c.n(2)};
    if (lb_n > lb) {
      lb = lb_n;
      out.normal = c.n;
    }
    ub = UpdateSimplex(c, out);
    ++iter;

    DGD_PRINT_DEBUG_ITERATION(iter, lb, ub, c.s, settings, out);

    // Termination criteria.
    if constexpr (detect_collision) {
      // Perform collision check.
      if (lb > -cdist) {
        // No collision.
        out.normal = rot.transpose() * out.normal;
        out.growth_dist_lb = -cdist / lb;
        out.status = SolutionStatus::Optimal;
        gd = -1.0;
        break;
      } else if (ub <= -cdist) {
        // Collision.
        out.growth_dist_ub = -cdist / ub;
        DGD_COMPUTE_PRIMAL_SOLUTION(tf1, tf2, out);
        out.status = SolutionStatus::Optimal;
        gd = 1.0;
        break;
      }
    } else {
      // Check primal-dual gap.
      if (lb >= ub * settings.rel_tol) {
        // Transform the normal vector to world frame.
        out.normal = rot.transpose() * out.normal;
        out.growth_dist_lb = -cdist / lb;
        out.growth_dist_ub = -cdist / ub;
        ComputePrimalSolution(tf1, tf2, out);
        out.status = SolutionStatus::Optimal;
        gd = out.growth_dist_lb;
        break;
      }
    }
    if (out.iter >= settings.max_iter) {
      out.status = SolutionStatus::MaxIterReached;
      gd = -cdist / lb;
      break;
    }

    UpdateNormal(c, normalize);
  }
  out.iter = iter;

  DGD_PRINT_DEBUG_FOOTER();

  return gd;
}

/**
 * Boolean collision detection function.
 */

/**
 * @brief Collision detection algorithm for 2D and 3D convex sets.
 *
 * Returns true if the centers coincide or if the sets intersect;
 * false if a separating plane has been found or if the maximum number of
 * iterations have been reached.
 *
 * @param[in]     set1,set2  Convex sets.
 * @param[in]     tf1,tf2    Rigid body transformations for the sets.
 * @param[in]     settings   Solver settings.
 * @param[in,out] out        Solver output.
 * @param         warm_start Use previous solver output to warm start current
 *                           solution (default = false).
 * @return        true, if the sets are colliding; false, otherwise.
 */
template <class C1, class C2, int dim>
inline bool DetectCollision(const C1* set1, const Transformr<dim>& tf1,
                            const C2* set2, const Transformr<dim>& tf2,
                            const Settings& settings, Output<dim>& out,
                            bool warm_start = false) {
  const Real gd{GrowthDistance<C1, C2, true>(set1, tf1, set2, tf2, settings,
                                             out, warm_start)};
  return ((out.status == SolutionStatus::CoincidentCenters) ||
          (out.status == SolutionStatus::Optimal && gd > 0.0));
}

}  // namespace dgd

#endif  // DGD_GROWTH_DISTANCE_H_
