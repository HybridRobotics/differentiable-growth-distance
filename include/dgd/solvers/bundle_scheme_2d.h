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
 * @brief Bundle schemes for the two-dimensional growth distance problem.
 */

#ifndef DGD_SOLVERS_BUNDLE_SCHEME_2D_H_
#define DGD_SOLVERS_BUNDLE_SCHEME_2D_H_

#include <cmath>

#include "dgd/data_types.h"
#include "dgd/output.h"
#include "dgd/settings.h"
#include "dgd/solvers/debug.h"
#include "dgd/solvers/solver_settings.h"
#include "dgd/solvers/solver_types.h"
#include "dgd/solvers/solver_utils.h"

namespace dgd {

namespace detail {

/*
 * Initialization.
 */

/// @brief Initializes the simplex, normal vector, and barycentric coordinates.
inline void InitializeSimplex(Matr<2, 2>& s, Vec2r& n, Real r, Output<2>& out) {
  s.col(0) = Vec2r(-r, Real(0.0));
  s.col(1) = Vec2r(r, Real(0.0));
  n = Vec2r::UnitY();
  out.bc = Vec2r::Constant(Real(0.5));
}

/**
 * @brief Initializes the convex set simplices corresponding to the Minkowski
 * difference set simplex.
 */
inline void InitializeSetSimplices(const MinkowskiDiffProp<2>& mdp,
                                   Output<2>& out) {
  out.s1.col(1) = out.r1_ * mdp.rot1.transpose().col(0);
  out.s1.col(0) = -out.s1.col(1);
  out.s2.col(0) = out.r2_ * mdp.rot2.transpose().col(0);
  out.s2.col(1) = -out.s2.col(0);
}

/*
 * Cutting plane method functions.
 */

/**
 * @brief Updates the simplices and the barycentric coordinates and returns the
 * lower bound, given a support point.
 */
inline Real UpdateSimplex(const Vec2r& sp, const Vec2r& sp1, const Vec2r& sp2,
                          Matr<2, 2>& s, Output<2>& out, int* idxn = nullptr) {
  const int idx = (sp(0) >= 0);
  s.col(idx) = sp;
  out.s1.col(idx) = sp1;
  out.s2.col(idx) = sp2;
  if (idxn) *idxn = idx;

  out.bc = Vec2r(s(0, 1), -s(0, 0));
  // Note: out.bc.sum() is always greater than (or on the order of)
  // 2.0 * rel_tol / inradius.
  // Otherwise, the algorithm would have converged in the previous iteration.
  out.bc /= out.bc.sum();
  return s.row(1) * out.bc;
}

/// @brief Updates the normal vector for the cutting plane method.
inline void UpdateNormalCuttingPlane(const Matr<2, 2>& s, Vec2r& n) {
  // Note that dual feasibility is always satisfied if the algorithm has not
  // converged.
  n = Vec2r(s(1, 0) - s(1, 1), s(0, 1) - s(0, 0));
}

/*
 * Trust region Newton method functions.
 */

/**
 * @brief Updates the normal vector to the trust region Newton solution.
 *
 * @note \f$\text{hess} > 0\f$ should be ensured;
 * if \f$\text{hess} = 0\f$, use the cutting plane update.
 */
inline void UpdateNormalTrustRegionNewton(const Matr<2, 2>& s,
                                          const Vec2r& n_cp, Vec2r& n,
                                          Real hess, int idxn) {
  const Real sgn = Real(2 * idxn - 1);
  // const Real grad = hess * (n_cp(0) / n_cp(1) - n(0) / n(1)) + s(0, idxn);
  const Real grad =
      hess * (n_cp(0) * n(1) - n(0) * n_cp(1)) + n(1) * n_cp(1) * s(0, idxn);
  if (sgn * grad < Real(0.0)) {
    n(0) -= n(1) * (s(0, idxn) / hess);
  } else {
    n = n_cp;
  }
}

/*
 * Warm start.
 */

/**
 * @brief Warm starts the simplex and the normal vector and returns the lower
 * bound using the previous primal optimal solution.
 */
template <bool detect_collision>
inline Real PrimalWarmStart(const MinkowskiDiffProp<2>& mdp, Matr<2, 2>& s,
                            Vec2r& n, Output<2>& out) {
  const Matr<2, 2> s1 = out.s1, s2 = out.s2;
  Vec2r sp;
  Real lb = Real(0.0);

  InitializeSimplex(s, n, mdp.r, out);
  // if constexpr (detect_collision) InitializeSetSimplices(mdp, out);
  InitializeSetSimplices(mdp, out);
  for (int i = 0; i < 2; ++i) {
    // Simplex points having a small contribution are not considered.
    if (out.bc(i) > SolverSettings::kEpsMinBc) {
      sp.noalias() = mdp.rot1 * s1.col(i) - mdp.rot2 * s2.col(i);
      if (n.dot(sp - s.col(0)) > Real(0.0)) {
        lb = UpdateSimplex(sp, s1.col(i), s2.col(i), s, out);
        UpdateNormalCuttingPlane(s, n);
      }
    }
  }
  NormalizeNormal(n, out.normalize_2norm_);
  return lb;
}

/**
 * @brief Warm starts the normal vector using the previous dual optimal
 * solution.
 */
inline void DualWarmStart(const MinkowskiDiffProp<2>& mdp, const Output<2>& out,
                          Vec2r& n) {
  if ((n(1) = mdp.rot.row(1).dot(out.normal)) > Real(0.0)) {
    // Warm-started normal is dual feasible.
    n(0) = mdp.rot.row(0).dot(out.normal);
  } else {
    n(1) = Real(1.0);
  }
}

/*
 * Debug printing.
 */

/// @brief Prints the debugging header.
template <SolverType S>
inline void PrintDebugHeader(bool warm_start, WarmStartType ws_type) {
  std::string header =
      SolverName<S>() + " (dim = 2) (" + InitializationName(warm_start);
  if (warm_start) header += ": " + WarmStartName(ws_type);
  header += ")";
  PrintDebugHeader(header);
}

/// @brief Prints debugging information at an iteration of the algorithm.
inline void PrintDebugIteration(int iter, [[maybe_unused]] Real cdist, Real lb,
                                Real ub, Real rel_tol, const Matr<2, 2>& s,
                                const Vec2r& bc) {
  if constexpr (SolverSettings::kPrintGdBounds) {
    PrintDebugIteration(iter, cdist / lb, cdist / ub, rel_tol,
                        std::abs(s.row(0) * bc));
  } else {
    PrintDebugIteration(iter, ub, lb, rel_tol, std::abs(s.row(0) * bc));
  }
}

/*
 * General bundle scheme in 2D.
 */

/**
 * @brief Bundle scheme for the growth distance problem in 2D.
 *
 * @note When detecting collision, the output is 1.0 if the sets are colliding,
 * and -1.0 otherwise.
 */
template <class C1, class C2, SolverType S, BcSolverType /*BST*/,
          bool detect_collision>
Real BundleScheme(const C1* set1, const Transform2r& tf1, const C2* set2,
                  const Transform2r& tf2, const Settings& settings,
                  Output<2>& out, bool warm_start) {
  if (!warm_start) InitializeOutput(set1, set2, out);

  MinkowskiDiffProp<2> mdp;
  // Check center distance.
  mdp.SetCenterDistance(tf1, tf2);
  if (mdp.cdist < settings.min_center_dist) return SetZeroOutput(tf1, tf2, out);
  // Set alignment rotation matrices and inradius.
  mdp.SetRotationMatrices(tf1, tf2);
  mdp.r = out.r1_ + out.r2_;

  // Support function output.
  SupportFunctionOutput<2, SolverOrder<S>()> sfo;
  // Simplex matrix and the normal vector.
  Matr<2, 2> s;
  Vec2r n, n_cp;
  // Growth distance bounds.
  Real lb = Real(0.0), ub = kInf, gd;
  // Other local variables.
  int iter = 0, idxn;
  bool update_lb;

  if (warm_start && (out.status == SolutionStatus::Optimal)) {
    // Warm start.
    if (settings.ws_type == WarmStartType::Primal) {  // Primal warm start.
      lb = PrimalWarmStart<detect_collision>(mdp, s, n, out);
    } else {  // Dual warm start.
      InitializeSimplex(s, n, mdp.r, out);
      // if constexpr (detect_collision) InitializeSetSimplices(mdp, out);
      InitializeSetSimplices(mdp, out);
      DualWarmStart(mdp, out, n);
    }
  } else {
    // Cold start.
    InitializeSimplex(s, n, mdp.r, out);
    // Note: There can be some edge case numerical issues with the primal
    // infeasibility error if the convex set simplices are not initialized.
    // These issues don't occur with the cutting plane method.
    // if constexpr (detect_collision) InitializeSetSimplices(mdp, out);
    InitializeSetSimplices(mdp, out);
  }

  if constexpr (SolverSettings::kVerboseIteration) {
    PrintDebugHeader<S>(warm_start, settings.ws_type);
    PrintDebugIteration(iter, mdp.cdist, lb, ub, settings.rel_tol, s, out.bc);
  }
#ifdef DGD_EXTRACT_METRICS
  InitializeLogs(settings.max_iter, out);
  LogBounds(iter, lb, ub, out);
#endif  // DGD_EXTRACT_METRICS

  while (true) {
    // Evaluate the support functions at the normal.
    sfo.Evaluate(set1, set2, mdp, n, out);

    // Update the upper bound and the current best normal vector.
    const Real ub_new = (sfo.sv1 + sfo.sv2) / n(1);
    if (ub_new < ub) {
      ub = ub_new;
      out.normal = n;
    }

    // Update the lower bound and the simplex.
    if ((update_lb = !((iter == 0) && warm_start &&
                       (settings.ws_type == WarmStartType::Dual) &&
                       (sfo.sp(1) <= Real(0.0))))) {
      if constexpr (S == SolverType::CuttingPlane) {
        lb = UpdateSimplex(sfo.sp, sfo.sp1(), sfo.sp2(), s, out);
      } else {  // TrustRegionNewton
        // In 2D, the lower bound is guaranteed to be nondecreasing.
        lb = UpdateSimplex(sfo.sp, sfo.sp1(), sfo.sp2(), s, out, &idxn);
      }
    }

    ++iter;

    if constexpr (SolverSettings::kVerboseIteration) {
      PrintDebugIteration(iter, mdp.cdist, lb, ub, settings.rel_tol, s, out.bc);
    }
#ifdef DGD_EXTRACT_METRICS
    LogBounds(iter, lb, ub, out);
#endif  // DGD_EXTRACT_METRICS

    // Termination criteria.
    if constexpr (detect_collision) {
      // Perform collision check.
      if (ub < mdp.cdist) {
        // No collision.
        ComputeDualSolution(mdp.rot, mdp.cdist, ub, out);
        out.status = SolutionStatus::Optimal;
        gd = Real(-1.0);
        break;
      } else if (lb >= mdp.cdist) {
        // Collision.
        ComputePrimalSolution(tf1, tf2, mdp.cdist, lb, out);
        out.status = SolutionStatus::Optimal;
        gd = Real(1.0);
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
    if (update_lb) {
      if constexpr (S == SolverType::CuttingPlane) {
        UpdateNormalCuttingPlane(s, n);
      } else {  // TrustRegionNewton
        UpdateNormalCuttingPlane(s, n_cp);
        Real hess;
        if (sfo.differentiable &&
            ((hess = n(1) * sfo.Dsp(0, 0)) > SolverSettings::kHessMin2)) {
          UpdateNormalTrustRegionNewton(s, n_cp, n, hess, idxn);
        } else {
          n = n_cp;
        }
      }
      NormalizeNormal(n, out.normalize_2norm_);
    } else {
      // (Dual warm start) Lower bound was not updated; reset the normal vector.
      n = Vec2r::UnitY();
    }
  }

  out.iter = iter;

  if constexpr (SolverSettings::kVerboseIteration) PrintDebugFooter();

  // (test)
  // out.prim_infeas_err = std::abs(s.row(0) * out.bc);

  return gd;
}

}  // namespace detail

}  // namespace dgd

#endif  // DGD_SOLVERS_BUNDLE_SCHEME_2D_H_
