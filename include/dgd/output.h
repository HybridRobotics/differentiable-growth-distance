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
 * @brief Solver output struct.
 *
 * @note A simplex tableau of size \f$= 2\f$ is sufficient to guarantee
 * exponential convergence for two-dimensional sets.
 *
 * @tparam dim          Dimension of the convex sets.
 * @tparam simplex_size Maximum number of points in the simplex tableau.
 */
template <int dim, int simplex_size>
struct SolverOutput {
  /**
   * @name Convex set simplex vertices
   * @brief Support points for convex set \f$i\f$, \f$(i = 1, 2)\f$,
   * corresponding to the optimal simplex tableau for the Minkowski difference
   * set.
   *
   * @attention When the simplex points from the previous time step are
   * available, they can be used to warm start the growth distance algorithm.
   *
   * @note The simplex points are stored in the untransformed reference frame
   * of the convex set, i.e., the support points returned by the support
   * function are directly stored.
   */
  ///@{
  Matf<dim, simplex_size> s1;
  Matf<dim, simplex_size> s2;
  ///@}

  /**
   * @brief Barycentric coordinates corresponding to the optimal simplex
   * tableau.
   */
  Vecf<dim> bc;

  /**
   * @brief The normal vector of an optimal hyperplane.
   *
   * @attention The normal vector is in the transformed frame of reference.
   */
  Vecf<dim> normal;

  /**
   * @brief Lower bound on the growth distance.
   *
   * The lower bound corresponds to the optimal hyperplane.
   */
  Real growth_dist_lb;

  /**
   * @brief Upper bound on the growth distance.
   *
   * The upper bound corresponds to the ray intersection point on the Minkowski
   * difference set.
   */
  Real growth_dist_ub;

  /**
   * @brief Inradius of the Minkowski difference set.
   */
  Real inradius;

  /**
   * @brief Number of solver iterations.
   */
  int iter;

  /**
   * @brief Solution status.
   */
  SolutionStatus status;
};

template <int dim>
using SolutionOutput = SolutionOutput<dim, dim>;

}  // namespace dgd

#endif  // DGD_OUTPUT_H_
