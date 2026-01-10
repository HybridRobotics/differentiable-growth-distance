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
 * @brief Growth distance algorithm output.
 */

#ifndef DGD_OUTPUT_H_
#define DGD_OUTPUT_H_

#include <string>
#ifdef DGD_EXTRACT_METRICS
#include <vector>
#endif  // DGD_EXTRACT_METRICS

#include "dgd/data_types.h"
#include "dgd/geometry/convex_set.h"

namespace dgd {

/// @brief Solution status at the termination of the growth distance algorithm.
enum class SolutionStatus {
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
 * @brief Growth distance algorithm output.
 *
 * @attention When using cold start, an Output instance can be shared across
 * different pairs of convex sets. However, when using warm start, each
 * collision pair should use a different instance.
 *
 * @tparam dim Dimension of the convex sets.
 */
template <int dim>
struct Output {
  /**
   * @name Convex set simplex vertices
   * @brief Simplex vertices for the convex sets (in the local frame),
   * corresponding to the optimal inner polyhedral approximation.
   */
  ///@{
  Matr<dim, dim> s1 = Matr<dim, dim>::Zero();
  Matr<dim, dim> s2 = Matr<dim, dim>::Zero();
  ///@}

  /// @brief Barycentric coordinates corresponding to the optimal simplex.
  Vecr<dim> bc = Vecr<dim>::Zero();

  /**
   * @brief The normal vector of an optimal hyperplane (dual optimal solution).
   *
   * @note The normal vector is in the world frame of reference and is
   * pointed towards the second set (with respect to the first set). The normal
   * vector need not have unit 2-norm.
   */
  Vecr<dim> normal = Vecr<dim>::Zero();

  /// @brief (internal) support function hints.
  SupportFunctionHint<dim> hint1_{}, hint2_{};

  /**
   * @name Primal optimal solutions
   * @brief Primal optimal solutions for each convex set.
   *
   * The solutions are primal feasible, i.e., they satisfy:
   * \f[
   * p_1 - p_2 + \text{growth_dist_ub} \cdot (z_1 - p_1 - (z_2 - p_2)) = 0.
   * \f]
   *
   * @note The primal optimal solutions are in the world frame of reference.
   */
  ///@{
  Vecr<dim> z1 = Vecr<dim>::Zero();
  Vecr<dim> z2 = Vecr<dim>::Zero();
  ///@}

  /**
   * @brief Upper bound on the growth distance.
   *
   * The upper bound corresponds to the primal solution (the ray intersection on
   * the inner polyhedral approximation).
   */
  Real growth_dist_ub = kInf;

  /**
   * @brief Lower bound on the growth distance.
   *
   * The lower bound corresponds to the dual solution (normal vector).
   */
  Real growth_dist_lb = Real(0.0);

  /// @brief (internal) convex set inradii.
  Real r1_ = kEps, r2_ = kEps;

  // (test) primal infeasibility error.
  // Real prim_infeas_err = kInf;

  /// @brief Number of solver iterations.
  int iter = 0;

  /// @brief (internal) unit normal vector flag.
  bool normalize_2norm_ = true;

  /// @brief Solution status.
  SolutionStatus status = SolutionStatus::MaxIterReached;

  /// @brief (logging) iteration-wise growth distance bounds.
#ifdef DGD_EXTRACT_METRICS
  std::vector<Real> lbs;
  std::vector<Real> ubs;
#endif  // DGD_EXTRACT_METRICS
};

using Output2 = Output<2>;
using Output3 = Output<3>;

/// @brief Returns the solution status name.
inline std::string SolutionStatusName(SolutionStatus status) {
  if (status == SolutionStatus::CoincidentCenters) {
    return "Coincident centers";
  } else if (status == SolutionStatus::MaxIterReached) {
    return "Maximum iterations reached";
  } else if (status == SolutionStatus::Optimal) {
    return "Optimal solution";
  } else {
    return "";
  }
}

}  // namespace dgd

#endif  // DGD_OUTPUT_H_
