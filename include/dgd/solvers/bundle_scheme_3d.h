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
 * @brief Bundle schemes for the three-dimensional growth distance problem.
 */

#ifndef DGD_SOLVERS_BUNDLE_SCHEME_3D_H_
#define DGD_SOLVERS_BUNDLE_SCHEME_3D_H_

#include <cassert>
#include <cmath>
#include <type_traits>

// #include "eiquadprog/eiquadprog-rt.hpp"

#include "dgd/data_types.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/solvers/debug.h"
#include "dgd/solvers/solver_options.h"
#include "dgd/solvers/utils.h"

namespace dgd {

namespace detail {

/*
 * Local context.
 */

// Local context for the bundle scheme.
struct BundleScheme3Context {
  // Simplex points in the aligned coordinates.
  // The simplex points are in CCW order when projected to the x-y plane.
  Matr<3, 3> s;

  // Barycentric coordinates.
  Vec3r bc;

  // Normal vector.
  Vec3r n;

  // Twice the area of the projected simplex (> 0).
  Real area;
};

/*
 * Initialization.
 */

// Initializes the simplex, normal vector, and barycentric coordinates.
inline void InitializeSimplex(BundleScheme3Context& bsc, Real r,
                              Output<3>& out) {
  bsc.s.col(0) = r * Vec3r(0.5, 0.5, 0.0);
  bsc.s.col(1) = r * Vec3r(-0.5, 0.5, 0.0);
  bsc.s.col(2) = Vec3r(0.0, -r, 0.0);
  bsc.n = Vec3r::UnitZ();
  bsc.area = Real(1.5) * r * r;
  out.bc = Vec3r::Constant(Real(1.0 / 3.0));
}

// Initializes the convex set simplices corresponding to the Minkowski
// difference set simplex.
inline void InitializeSetSimplices(const MinkowskiDiffProp<3>& mdp,
                                   Output<3>& out) {
  out.s1.col(1).noalias() =
      out.r1_ * mdp.rot1.transpose() * Vec3r(-0.5, 0.5, 0.0);
  out.s1.col(2) = -out.r1_ * mdp.rot1.transpose().col(1);
  out.s1.col(0) = -(out.s1.col(1) + out.s1.col(2));
  out.s2.col(1).noalias() =
      out.r2_ * mdp.rot2.transpose() * Vec3r(0.5, -0.5, 0.0);
  out.s2.col(2) = out.r2_ * mdp.rot2.transpose().col(1);
  out.s2.col(0) = -(out.s2.col(1) + out.s2.col(2));
}

/*
 * Cutting plane method functions.
 */

// Cycles the index in a CCW/CW order (for the projected simplex).
inline constexpr int Ccw3(int idx) { return (idx == 2) ? 0 : idx + 1; }

inline constexpr int Cw3(int idx) { return (idx == 0) ? 2 : idx - 1; }

// Swaps two columns of the simplex matrix.
inline void SwapSimplexColumns(Matr<3, 3>& s, int i, int j) {
  const Vec3r temp = s.col(i);
  s.col(i) = s.col(j);
  s.col(j) = temp;
}

// Computes the barycentric coordinates of a point with respect to the
// projected simplex, assuming nondegeneracy.
inline void ComputePointCoordinates(const Eigen::Ref<const Vec2r>& p,
                                    BundleScheme3Context& bsc) {
  bsc.bc(0) = ((bsc.s(0, 1) - p(0)) * (bsc.s(1, 2) - p(1)) -
               (bsc.s(1, 1) - p(1)) * (bsc.s(0, 2) - p(0))) /
              bsc.area;
  bsc.bc(1) = ((bsc.s(0, 2) - p(0)) * (bsc.s(1, 0) - p(1)) -
               (bsc.s(1, 2) - p(1)) * (bsc.s(0, 0) - p(0))) /
              bsc.area;
  bsc.bc(2) = Real(1.0) - (bsc.bc(0) + bsc.bc(1));
}

// When the projected simplex is degenerate, checks whether an edge intersects
// with the z-axis; if so, updates the z-intercept and barycentric coordinates.
template <int ax, int i>
inline void UpdateZIntersection1D(const Matr<3, 3>& s, Vec3r& bc, Real& max) {
  constexpr int j = Ccw3(i), k = Cw3(i);
  if (std::signbit(s(ax, i)) != std::signbit(s(ax, j))) {
    const Real len = s(ax, i) - s(ax, j);
    const Real bcj = (std::abs(len) > SolverSettings::kEpsArea3)
                         ? s(ax, i) / len
                         : Real(0.5);
    const Real v = s(2, i) + bcj * (s(2, j) - s(2, i));
    if (v > max) {
      bc(i) = Real(1.0) - bcj;
      bc(j) = bcj;
      bc(k) = 0.0;
      max = v;
    }
  }
}

// Updates the barycentric coordinates of the origin when the projected simplex
// is degenerate, and returns the z-coordinate of the intersection point.
template <int ax>
inline Real UpdateOriginCoordinates1D(const Matr<3, 3>& s, Vec3r& bc) {
  Real max = -1.0;
  UpdateZIntersection1D<ax, 0>(s, bc, max);
  UpdateZIntersection1D<ax, 1>(s, bc, max);
  UpdateZIntersection1D<ax, 2>(s, bc, max);
  if (max < Real(0.0)) {
    int i;
    s.row(ax).cwiseAbs().minCoeff(&i);
    bc(i) = 1.0;
    bc(Ccw3(i)) = bc(Cw3(i)) = 0.0;
    max = s(2, i);
  }
  return max;
}

// Updates the barycentric coordinates of the origin, and returns the
// z-coordinate of the intersection point.
inline Real UpdateOriginCoordinates(BundleScheme3Context& bsc, Vec3r& bc) {
  // The projected (signed) simplex area is guaranteed to be positive. The
  // implementation below is for robustness.
  bc(0) = std::abs(bsc.s(0, 1) * bsc.s(1, 2) - bsc.s(1, 1) * bsc.s(0, 2));
  bc(1) = std::abs(bsc.s(0, 2) * bsc.s(1, 0) - bsc.s(1, 2) * bsc.s(0, 0));
  bc(2) = std::abs(bsc.s(0, 0) * bsc.s(1, 1) - bsc.s(1, 0) * bsc.s(0, 1));
  bsc.area = bc.sum();
  if (bsc.area > SolverSettings::kEpsArea3) {
    bc /= bsc.area;
    return bsc.s.row(2) * bc;
  } else {  // The projected simplex is degenerate.
    if (bsc.s.row(0).lpNorm<Eigen::Infinity>() >
        bsc.s.row(1).lpNorm<Eigen::Infinity>()) {
      return UpdateOriginCoordinates1D<0>(bsc.s, bc);
    } else {
      return UpdateOriginCoordinates1D<1>(bsc.s, bc);
    }
  }
}

// Updates the simplices and the barycentric coordinates and returns the lower
// bound, given a support point.
inline Real UpdateSimplex(const Vec3r& sp, const Vec3r& sp1, const Vec3r& sp2,
                          BundleScheme3Context& bsc, Output<3>& out,
                          int* idxn = nullptr) {
  int exiting_idx = 0;
  if (bsc.area > SolverSettings::kEpsArea3) {
    ComputePointCoordinates(sp.head<2>(), bsc);
    // Perform one iteration of the Simplex algorithm.
    Real value = 1.0;
    for (int i = 0; i < 3; ++i) {
      if ((out.bc(i) < bsc.bc(i)) && (out.bc(i) < bsc.bc(i) * value)) {
        value = out.bc(i) / bsc.bc(i);
        exiting_idx = i;
      }
    }
  } else {
    out.bc.minCoeff(&exiting_idx);
    // Check projected simplex orientation.
    const int j = Ccw3(exiting_idx), k = Cw3(exiting_idx);
    if ((bsc.s.col(j) - sp).cross(bsc.s.col(k) - sp)(2) < Real(0.0)) {
      // Correct the simplex orientation.
      SwapSimplexColumns(bsc.s, j, k);
      SwapSimplexColumns(out.s1, j, k);
      SwapSimplexColumns(out.s2, j, k);
    }
  }

  // Replace the exiting simplex point with the support point.
  bsc.s.col(exiting_idx) = sp;
  out.s1.col(exiting_idx) = sp1;
  out.s2.col(exiting_idx) = sp2;
  if (idxn) *idxn = exiting_idx;
  return UpdateOriginCoordinates(bsc, out.bc);
}

// Updates the normal vector for the cutting plane method.
inline void UpdateNormalCuttingPlane(const BundleScheme3Context& bsc,
                                     Vec3r& n) {
  // The triangle edges are used to compute the normal vector because the origin
  // may not lie in the triangle interior.
  // n = (bsc.s.col(1) - bsc.s.col(0)).cross(bsc.s.col(2) - bsc.s.col(0));
  n(0) = (bsc.s(2, 2) - bsc.s(2, 0)) * (bsc.s(1, 1) - bsc.s(1, 0)) -
         (bsc.s(1, 2) - bsc.s(1, 0)) * (bsc.s(2, 1) - bsc.s(2, 0));
  n(1) = (bsc.s(0, 2) - bsc.s(0, 0)) * (bsc.s(2, 1) - bsc.s(2, 0)) -
         (bsc.s(2, 2) - bsc.s(2, 0)) * (bsc.s(0, 1) - bsc.s(0, 0));
  n(2) = bsc.area + SolverSettings::kEpsNormal3;
}

/*
 * Proximal bundle method functions.
 */

// // Updates the normal vector to the proximal point value.
// // The cost matrix should be nonzero; otherwise, use the cutting plane
// // update.
// inline void UpdateNormalProximalPoint(const Matr<2, 2>& Q, const Vec3r& n_cp,
//                                       BundleScheme3Context& bsc) {
//   const Vec2r lmb_cp = n_cp.head<2>();
//   const Vec2r lmb = bsc.n.head<2>() / bsc.n(2) - lmb_cp;
//
//   // Solve min_x (c'x + 0.5 x'Px), s.t. Aineq x + bineq >= 0.
//   RtMatrixX<3, 3>::d P = kEps * Matr<3, 3>::Identity();
//   P.block<2, 2>(0, 0) = Q;
//   RtVectorX<3>::d c = Vec3r::Zero();
//   c.head<2>().noalias() = -(Q + kEps * Matr<2, 2>::Identity()) * lmb;
//   c(2) = 1.0;
//   const RtMatrixX<0, 3>::d Aeq;
//   const RtVectorX<0>::d beq(0);
//   RtMatrixX<3, 3>::d Aineq;
//   Aineq.leftCols<2>() = -bsc.s.topRows<2>().transpose();
//   Aineq.col(2) = Vec3r::Ones();
//   const RtVectorX<3>::d bineq = Vec3r::Zero();
//   RtVectorX<3>::d x = Vec3r::Zero();
//
//   {
//     using namespace eiquadprog::solvers;
//     RtEiquadprog<3, 0, 3> qp;
//     [[maybe_unused]] RtEiquadprog_status status;
//     status = qp.solve_quadprog(P, c, Aeq, beq, Aineq, bineq, x);
//     assert(status == RT_EIQUADPROG_OPTIMAL);
//   }
//
//   bsc.n(0) = static_cast<Real>(x(0)) + lmb_cp(0);
//   bsc.n(1) = static_cast<Real>(x(1)) + lmb_cp(1);
//   bsc.n(2) = 1.0;
// }

/*
 * Trust region Newton method functions.
 */

// Partial trust region Newton solution.
inline void UpdateNormalPartialNewton(const Matr<2, 2>& hess, const Vec3r& n_cp,
                                      BundleScheme3Context& bsc, int idx) {
  // Compute the pseudoinverse of the cost matrix, assuming it is nonzero.
  Matr<2, 2> hess_inv;
  const Real det = hess.determinant();
  if (det > SolverSettings::kPinvTol3) {
    const Real det_inv = Real(1.0) / det;
    hess_inv(0, 0) = hess(1, 1) * det_inv;
    hess_inv(0, 1) = hess_inv(1, 0) = -hess(1, 0) * det_inv;
    hess_inv(1, 1) = hess(0, 0) * det_inv;
  } else {  // The Hessian is rank 1.
    if constexpr (SolverSettings::kSkipTrnIfSingularHess3) {
      bsc.n = n_cp;
      return;
    }
    const Real trace = hess.trace();
    hess_inv = hess / (trace * trace);
  }

  // Newton step solution.
  const Vec2r grad = bsc.s.col(idx).head<2>(), lmb = bsc.n.head<2>() / bsc.n(2);
  const Vec2r lmb_opt = lmb - hess_inv * grad;
  // Residual error should be on the order of machine epsilon squared.
  Real res_err2 = 0.0;
  if constexpr (!SolverSettings::kSkipTrnIfSingularHess3) {
    res_err2 = (hess * (lmb_opt - lmb) + grad).squaredNorm();
  }
  // Check trust region feasibility and residual error.
  const Vec3r sv =
      bsc.s.topRows<2>().transpose() * lmb_opt + bsc.s.row(2).transpose();
  if ((res_err2 < SolverSettings::kPinvResErr3) && (sv(idx) >= sv(Ccw3(idx))) &&
      (sv(idx) >= sv(Cw3(idx)))) {
    // Newton step solution lies within trust region bounds.
    bsc.n.head<2>() = lmb_opt;
    bsc.n(2) = 1.0;
    // return -Real(0.5) * grad.dot(hess_inv * grad);
  } else {
    // Return the cutting plane solution.
    bsc.n = n_cp;
    // return (n_cp.head<2>() - lmb).dot(grad + Real(0.5) * hess *
    // (n_cp.head<2>() - lmb));
  }
}

// // Full trust region Newton solution.
// inline void UpdateNormalFullNewton(const Matr<2, 2>& hess, const Vec3r& n_cp,
//                                    BundleScheme3Context& bsc, int idx) {
//   const Vec2r lmb = bsc.n.head<2>() / bsc.n(2) - n_cp.head<2>();
//   const Vec2r grad = bsc.s.col(idx).head<2>() - hess * lmb;
//
//   // Setup trust region Newton QP problem.
//   const RtMatrixX<2, 2>::d Q =
//       hess + SolverSettings::kPinvTol3 * Matr<2, 2>::Identity();
//   const RtVectorX<2>::d c = grad - SolverSettings::kPinvTol3 * lmb;
//   const RtMatrixX<0, 2>::d Aeq;
//   const RtVectorX<0>::d beq(0);
//   RtMatrixX<2, 2>::d Aineq;
//   Aineq.row(0) = (bsc.s.col(idx) - bsc.s.col(Ccw3(idx))).head<2>();
//   Aineq.row(1) = (bsc.s.col(idx) - bsc.s.col(Cw3(idx))).head<2>();
//   const RtVectorX<2>::d bineq = Vecr<2>::Zero();
//   RtVectorX<2>::d x = Vecr<2>::Zero();
//
//   // Solve problem.
//   {
//     using namespace eiquadprog::solvers;
//     RtEiquadprog<2, 0, 2> qp;
//     [[maybe_unused]] RtEiquadprog_status status =
//         qp.solve_quadprog(Q, c, Aeq, beq, Aineq, bineq, x);
//     assert(status == RT_EIQUADPROG_OPTIMAL);
//   }
//
//   const Vec2r lmb_opt = Vec2r(static_cast<Real>(x(0)),
//   static_cast<Real>(x(1))); bsc.n.head<2>() = lmb_opt + n_cp.head<2>();
//   bsc.n(2) = 1.0;
//   // return grad.dot(lmb_opt) + Real(0.5) * lmb_opt.dot(hess * lmb_opt);
// }

// Updates the normal vector for the trust region Newton method.
inline void UpdateNormalNewton(const Matr<2, 2>& hess, const Vec3r& n_cp,
                               BundleScheme3Context& bsc, int idx) {
  if constexpr (SolverSettings::kTrnLevel == TrustRegionNewtonLevel::kPartial) {
    UpdateNormalPartialNewton(hess, n_cp, bsc, idx);
  }  // else {
  //   UpdateNormalFullNewton(hess, n_cp, bsc, idx);
  // }
}

/*
 * Warm start.
 */

// Warm starts the simplex and the normal vector and returns the lower bound.
template <bool detect_collision>
inline Real WarmStart(const MinkowskiDiffProp<3>& mdp,
                      BundleScheme3Context& bsc, Output<3>& out) {
  const Matr<3, 3> s1 = out.s1, s2 = out.s2;
  Vec3r sp;
  Real lb = 0.0;

  InitializeSimplex(bsc, mdp.r, out);
  // if constexpr (detect_collision) InitializeSetSimplices(mdp, out);
  InitializeSetSimplices(mdp, out);
  for (int i = 0; i < 3; ++i) {
    // Simplex points having a small contribution are not considered.
    if (out.bc(i) > SolverSettings::kEpsMinBc) {
      sp.noalias() = mdp.rot1 * s1.col(i) - mdp.rot2 * s2.col(i);
      if (bsc.n.dot(sp - bsc.s.col(0)) > Real(0.0)) {
        lb = UpdateSimplex(sp, s1.col(i), s2.col(i), bsc, out);
        UpdateNormalCuttingPlane(bsc, bsc.n);
      }
    }
  }
  NormalizeNormal(bsc.n, out.normalize_2norm_);
  return lb;
}

/*
 * Debug printing.
 */

// Prints debugging information at any iteration of the algorithm.
inline void PrintDebugIteration(int iter, [[maybe_unused]] Real cdist, Real lb,
                                Real ub, Real rel_tol, const Matr<3, 3>& s,
                                const Vec3r& bc) {
  if constexpr (SolverSettings::kPrintGdBounds) {
    PrintDebugIteration(iter, cdist / lb, cdist / ub, rel_tol,
                        (s.topRows<2>() * bc).norm());
  } else {
    PrintDebugIteration(iter, ub, lb, rel_tol, (s.topRows<2>() * bc).norm());
  }
}

/*
 * General bundle scheme in 3D.
 */

// Bundle scheme for the three-dimensional growth distance problem.
// When detecting collision, the output is 1.0 if the sets are colliding,
// and -1.0 otherwise.
template <class C1, class C2, SolverType S, bool detect_collision>
Real BundleScheme(const C1* set1, const Transform3r& tf1, const C2* set2,
                  const Transform3r& tf2, const Settings& settings,
                  Output<3>& out, bool warm_start) {
  if constexpr ((S == SolverType::TrustRegionNewton) &&
                (SolverSettings::kTrnLevel == TrustRegionNewtonLevel::kFull)) {
    static_assert(
        always_false<S>::value,
        "The full trust region Newton method for dim = 3 is disabled");
  }
  if constexpr (S == SolverType::ProximalBundle) {
    static_assert(always_false<S>::value,
                  "The proximal bundle method for dim = 3 is disabled");
  }

  if (!warm_start) InitializeOutput(set1, set2, out);

  MinkowskiDiffProp<3> mdp;
  // Check center distance.
  mdp.SetCenterDistance(tf1, tf2);
  if (mdp.cdist < settings.min_center_dist) return SetZeroOutput(tf1, tf2, out);
  // Set alignment rotation matrices.
  mdp.SetRotationMatrices(tf1, tf2);
  mdp.r = out.r1_ + out.r2_;

  // Support function output.
  SupportFunctionOutput<3, SolverOrder<S>()> sfo;
  // Local bundle scheme context.
  BundleScheme3Context bsc;
  // Growth distance bounds.
  Real lb = 0.0, ub = kInf, gd;
  // Other local variables.
  Matr<2, 2> Q;
  Vec3r n_cp;
  int iter = 0, idxn;
  bool use_qp_update;

  if (warm_start && (out.status == SolutionStatus::Optimal)) {
    // Warm start.
    lb = WarmStart<detect_collision>(mdp, bsc, out);
  } else {
    // Cold start.
    InitializeSimplex(bsc, mdp.r, out);
    // Note: There can be some edge case numerical issues with the primal
    // infeasibility error if the convex set simplices are not initialized.
    // These issues don't occur with the cutting plane method.
    // if constexpr (detect_collision) InitializeSetSimplices(mdp, out);
    InitializeSetSimplices(mdp, out);
  }
  if constexpr (S != SolverType::CuttingPlane) n_cp = bsc.n;

  if constexpr (SolverSettings::kEnableDebugPrinting) {
    PrintDebugHeader(SolverName<S>() + " (dim = 3)");
    PrintDebugIteration(iter, mdp.cdist, lb, ub, settings.rel_tol, bsc.s,
                        out.bc);
  }

  while (true) {
    // Evaluate the support functions at the normal.
    sfo.Evaluate(set1, set2, mdp, bsc.n, out);

    // Update the upper bound and the current best normal vector.
    const Real ub_new = (sfo.sv1 + sfo.sv2) / bsc.n(2);
    if (ub_new < ub) {
      ub = ub_new;
      out.normal = bsc.n;
    }
    // Update the lower bound and the simplex.
    if constexpr (S == SolverType::CuttingPlane) {
      lb = UpdateSimplex(sfo.sp, sfo.sp1, sfo.sp2, bsc, out);
    } else {
      // Check if the lower bound can be improved; if not, skip the simplex
      // update, and update the normal to the cutting plane normal.
      if ((use_qp_update =
               (n_cp.dot(sfo.sp - bsc.s.col(0)) > SolverSettings::kEpsLb3))) {
        if constexpr (S == SolverType::ProximalBundle) {
          lb = UpdateSimplex(sfo.sp, sfo.sp1, sfo.sp2, bsc, out);
        } else {  // TrustRegionNewton.
          lb = UpdateSimplex(sfo.sp, sfo.deriv1.sp, sfo.deriv2.sp, bsc, out,
                             &idxn);
        }
      }
    }
    ++iter;

    if constexpr (SolverSettings::kEnableDebugPrinting) {
      PrintDebugIteration(iter, mdp.cdist, lb, ub, settings.rel_tol, bsc.s,
                          out.bc);
    }

    // Termination criteria.
    if constexpr (detect_collision) {
      // Perform collision check.
      if (ub < mdp.cdist) {
        // No collision.
        ComputeDualSolution(mdp.rot, mdp.cdist, ub, out);
        out.status = SolutionStatus::Optimal;
        gd = -1.0;
        break;
      } else if (lb >= mdp.cdist) {
        // Collision.
        ComputePrimalSolution(tf1, tf2, mdp.cdist, lb, out);
        out.status = SolutionStatus::Optimal;
        gd = 1.0;
        break;
      }
    } else {
      // Check primal-dual gap.
      if (ub <= lb * settings.rel_tol) {
        ComputePrimalSolution(tf1, tf2, mdp.cdist, lb, out);
        gd = ComputeDualSolution(mdp.rot, mdp.cdist, ub, out);
        out.status = SolutionStatus::Optimal;
        break;
      }
    }
    // Check number of iterations.
    if (iter >= settings.max_iter) {
      ComputePrimalSolution(tf1, tf2, mdp.cdist, lb, out);
      gd = ComputeDualSolution(mdp.rot, mdp.cdist, ub, out);
      out.status = SolutionStatus::MaxIterReached;
      break;
    }

    // Update the normal vector.
    if constexpr (S == SolverType::CuttingPlane) {
      UpdateNormalCuttingPlane(bsc, bsc.n);
    } else {
      UpdateNormalCuttingPlane(bsc, n_cp);
      // Note: The normal update functions assume that n_cp(2) = 1.
      n_cp /= n_cp(2);
      if constexpr (S == SolverType::ProximalBundle) {
        if (use_qp_update) {
          Q = ComputeGammaProximalBundle(ub, mdp.r, iter) *
              Matr<2, 2>::Identity();
          // UpdateNormalProximalPoint(Q, n_cp, bsc);
        } else {
          bsc.n = n_cp;
        }
      } else {  // TrustRegionNewton.
        if (use_qp_update && sfo.differentiable &&
            (bsc.n(2) * (sfo.Dsp(0, 0) + sfo.Dsp(1, 1)) >
             SolverSettings::kPinvTol3)) {
          Q = bsc.n(2) * sfo.Dsp.template block<2, 2>(0, 0);
          UpdateNormalNewton(Q, n_cp, bsc, idxn);
        } else {
          bsc.n = n_cp;
        }
      }
    }
    NormalizeNormal(bsc.n, out.normalize_2norm_);
  }
  out.iter = iter;

  if constexpr (SolverSettings::kEnableDebugPrinting) PrintDebugFooter();

  return gd;
}

}  // namespace detail

}  // namespace dgd

#endif  // DGD_SOLVERS_BUNDLE_SCHEME_3D_H_
