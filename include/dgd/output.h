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
 * @file output.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-02-18
 * @brief Solver output struct.
 */

#ifndef DGD_OUTPUT_H_
#define DGD_OUTPUT_H_

#include <cstdint>

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief Solution status at the termination of the algorithm.
 */
enum class SolutionStatus : uint8_t {
  /**
   * @brief Optimal solution reached according to the relative tolerance
   * criterion.
   *
   * @see SolverSettings.rel_tol
   */
  kOptimal,

  /**
   * @brief Maximum number of iterations reached.
   *
   * @see SolverSettings.max_iter
   */
  kMaxIterReached,

  /**
   * @brief Coincident center positions of the convex sets.
   *
   * @see SolverSettings.min_center_dist
   */
  kCoincidentCenters,
};

/**
 * @brief Support function hint struct; used internally.
 *
 * @tparam dim Dimension of the convex sets.
 */
template <int dim>
struct SupportFunctionHint {
  Vecf<dim> n_prev{Vecf<dim>::Zero()};
  int idx_ws{-1};
};

/**
 * @brief Solver output struct.
 *
 * @attention When not using warm start, a SolverOutput instance can be shared
 * across different pairs of convex sets. When using warm start, each pair
 * should use a different object.
 *
 * @tparam dim Dimension of the convex sets.
 */
template <int dim>
struct SolverOutput {
  /**
   * @name Convex set simplex vertices
   * @brief Support points for convex set \f$i\f$, \f$(i = 1, 2)\f$,
   * corresponding to the optimal simplex for the Minkowski difference set.
   *
   * @attention When the simplex points from the previous time step are
   * available, they can be used to warm start the growth distance algorithm.
   *
   * @note The simplex points are stored in the local reference frame of the
   * convex set, i.e., the support points returned by the support function are
   * directly stored.
   */
  ///@{
  Matf<dim, dim> s1{Matf<dim, dim>::Zero()};
  Matf<dim, dim> s2{Matf<dim, dim>::Zero()};
  ///@}

  /**
   * @brief Barycentric coordinates corresponding to the optimal simplex.
   */
  Vecf<dim> bc{Vecf<dim>::Zero()};

  /**
   * @brief The normal vector of an optimal hyperplane (dual optimal solution).
   *
   * @attention The normal vector is in the world frame of reference.
   */
  Vecf<dim> normal{Vecf<dim>::Zero()};

  /**
   * @name Support function hints
   * @brief Additional hints for the support functions; used internally.
   */
  ///@{
  SupportFunctionHint<dim> hint1_{};
  SupportFunctionHint<dim> hint2_{};
  ///@}

  /**
   * @brief Lower bound on the growth distance.
   *
   * The lower bound corresponds to the optimal hyperplane.
   */
  Real growth_dist_lb{0.0};

  /**
   * @brief Upper bound on the growth distance.
   *
   * The upper bound corresponds to the ray intersection point on the Minkowski
   * difference set.
   */
  Real growth_dist_ub{0.0};

  /**
   * @brief Inradius of the Minkowski difference set.
   */
  Real inradius{kEps};

  /**
   * @brief Number of solver iterations.
   */
  int iter{0};

  /**
   * @brief Solution status.
   */
  SolutionStatus status{SolutionStatus::kMaxIterReached};
};

/**
 * @brief Solution error metrics.
 */
struct SolutionError {
  /**
   * @brief Relative primal-dual gap.
   *
   * The error is given by
   * \f[
   * \text{prim_dual_gap}
   * = \left|\frac{\text{growth_dist_ub}}{\text{growth_dist_lb}} - 1\right|.
   * \f]
   * When the growth distance algorithm converges, this error is less than the
   * specified value of rel_tol.
   */
  double prim_dual_gap;

  /**
   * @brief Primal feasibility error.
   *
   * The growth distance algorithm returns contact points in the form of convex
   * set support points and barycentric coordinates. Then, the primal
   * feasibility error is given by
   * \f[
   * \text{prim_feas_err}
   * = | p_{12} + cp_{12} \cdot \text{growth_dist_ub}|_2,
   * \f]
   * where \f$p_{12}\f$ and \f$cp_{12}\f$ are the center position and contact
   * point (wrt the center) on the Minkowski difference set.
   */
  double prim_feas_err;

  /**
   * @brief Dual feasibility error.
   *
   * The growth distance algorithm always ensures dual feasibility, so this
   * error is zero.
   */
  double dual_feas_err{0.0};
};

}  // namespace dgd

#endif  // DGD_OUTPUT_H_
