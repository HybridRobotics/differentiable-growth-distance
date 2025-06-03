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
#include "dgd/geometry/convex_set.h"

namespace dgd {

/**
 * @brief Solution status at the termination of the algorithm.
 */
enum class SolutionStatus : uint8_t {
  /**
   * @brief Optimal solution reached according to the relative tolerance
   * criterion.
   *
   * @see Settings.rel_tol
   */
  Optimal,

  /**
   * @brief Maximum number of iterations reached.
   *
   * @see Settings.max_iter
   */
  MaxIterReached,

  /**
   * @brief Coincident center positions of the convex sets.
   *
   * @see Settings.min_center_dist
   */
  CoincidentCenters,
};

/**
 * @brief Solver output struct.
 *
 * @attention When not using warm start, an Output object can be shared across
 * different pairs of convex sets. When using warm start, each pair should use
 * a different object.
 *
 * @tparam dim Dimension of the convex sets.
 */
template <int dim>
struct Output {
  /**
   * @name Convex set simplex vertices
   * @brief Support points for convex set \f$i\f$, \f$(i = 1, 2)\f$,
   * corresponding to the optimal inner polyhedral approximation for the
   * Minkowski difference set.
   *
   * @attention When the simplex points from the previous time step are
   * available, they can be used to warm start the growth distance algorithm.
   *
   * @note The simplex points are stored in the local reference frame of the
   * convex set, i.e., the support points returned by the support function are
   * directly stored.
   */
  ///@{
  Matr<dim, dim> s1{Matr<dim, dim>::Zero()};
  Matr<dim, dim> s2{Matr<dim, dim>::Zero()};
  ///@}

  /**
   * @brief Barycentric coordinates corresponding to the optimal simplex.
   */
  Vecr<dim> bc{Vecr<dim>::Zero()};

  /**
   * @brief The normal vector of an optimal hyperplane (dual optimal solution).
   *
   * @attention The normal vector is in the world frame of reference and is
   * oriented along the vector p2 - p1, i.e., normal.dot(p2 - p1) > 0.
   *
   * @note The normal vector need not have unit 2-norm.
   */
  Vecr<dim> normal{Vecr<dim>::Zero()};

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
   * The upper bound corresponds to the ray intersection point on the inner
   * polyhedral approximation of the Minkowski difference set.
   */
  Real growth_dist_ub{0.0};

  /**
   * @name Primal optimal solutions
   * @brief Primal optimal solutions for each convex set.
   *
   * @attention The primal solutions are in the world frame of reference.
   */
  ///@{}
  Vecr<dim> z1{Vecr<dim>::Zero()};
  Vecr<dim> z2{Vecr<dim>::Zero()};
  ///@}

  /**
   * @brief (Lower bound of the) inradius of the Minkowski difference set.
   */
  Real inradius{kEps};

  /**
   * @brief Number of solver iterations.
   */
  int iter{0};

  /**
   * @brief Solution status.
   */
  SolutionStatus status{SolutionStatus::MaxIterReached};
};

}  // namespace dgd

#endif  // DGD_OUTPUT_H_
