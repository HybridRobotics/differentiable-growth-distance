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
 * @brief Differentiable growth distance algorithm output.
 */

#ifndef DGD_OUTPUT_H_
#define DGD_OUTPUT_H_

#include <memory>
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

  /**
   * @brief Ill-conditioned input sets.
   *
   * The input sets are ill-conditioned if the sum of their inradii is less than
   * \f$\sqrt{\epsilon_m}/2\f$ in 2D and \f$2\sqrt{\epsilon_m}\f$ in 3D.
   */
  IllConditionedInputs,
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

/**
 * @brief Directional (Gateaux) derivatives of the growth distance optimal value
 * and solution.
 *
 * @tparam dim Dimension of the convex sets.
 */
template <int dim>
struct DirectionalDerivative {
  /**
   * @brief Orthonormal basis for the primal solution set affine hull.
   *
   * @note The null space is an overapproximation of the linear subspace
   * corresponding to the primal solution set affine hull, i.e., it represents
   * the directions along which the primal solution (of both sets) can vary
   * locally while maintaining optimality.
   *
   * @note The basis vectors are in the world frame of reference, and orthogonal
   * to the normal vector.
   *
   * @see z_nullity
   */
  Matr<dim, dim - 1> z_nullspace = Matr<dim, dim - 1>::Zero();

  /**
   * @brief Orthonormal basis for the dual solution set span.
   *
   * @note The null space is an overapproximation of the dual solution set,
   * i.e., it represents the directions along which the dual solution (normal
   * vector) can vary locally while maintaining optimality.
   *
   * @note The basis vectors are in the world frame of reference, and the first
   * column is the computed optimal normal vector.
   *
   * @see n_nullity
   */
  Matr<dim, dim> n_nullspace = Matr<dim, dim>::Zero();

  /// @brief Derivative of the optimal normal vector (dual optimal solution).
  Vecr<dim> d_normal = Vecr<dim>::Zero();

  /**
   * @name Primal optimal solution derivatives
   * @brief Derivatives of the primal optimal solution.
   */
  ///@{
  Vecr<dim> d_z1 = Vecr<dim>::Zero();
  Vecr<dim> d_z2 = Vecr<dim>::Zero();
  ///@}

  /// @brief Derivative of the growth distance.
  Real d_gd = Real(0.0);

  /**
   * @brief Dimension of the primal solution set null space.
   *
   * @see z_nullspace
   */
  int z_nullity = 0;

  /**
   * @brief Dimension of the dual solution set null space.
   *
   * @see n_nullspace
   */
  int n_nullity = 0;

  /// @brief Differentiability of the growth distance optimal value.
  bool value_differentiable = false;

  /// @brief Differentiability of the growth distance optimal solution.
  bool differentiable = false;
};

/**
 * @brief Total (Frechet) derivatives of the growth distance optimal value
 * and solution.
 *
 * @attention The derivatives depend on the twist frame of reference.
 *
 * @tparam dim Dimension of the convex sets.
 */
template <int dim>
struct TotalDerivative {
  /// @brief Gradient of a scalar function with respect to rigid body motion.
  using Gradr = Twistr<dim>;

  /// @brief Jacobian of a vector function with respect to rigid body motion.
  using Jacr = Matr<dim, SeDim<dim>()>;

  /**
   * @name Optimal normal vector Jacobians
   * @brief Jacobians of the optimal normal vector (dual optimal solution) with
   * respect to rigid body motion of the convex sets.
   */
  ///@{
  Jacr d_normal_tf1 = Jacr::Zero();
  Jacr d_normal_tf2 = Jacr::Zero();
  ///@}

  /**
   * @name Primal optimal solution Jacobians
   * @brief Jacobians of the primal optimal solution with respect to rigid body
   * motion of the convex sets.
   */
  ///@{
  Jacr d_z1_tf1 = Jacr::Zero();
  Jacr d_z1_tf2 = Jacr::Zero();
  Jacr d_z2_tf1 = Jacr::Zero();
  Jacr d_z2_tf2 = Jacr::Zero();
  ///@}

  /**
   * @name Growth distance gradients
   * @brief Gradients of the growth distance with respect to rigid body motion
   * of the convex sets.
   */
  ///@{
  Gradr d_gd_tf1 = Gradr::Zero();
  Gradr d_gd_tf2 = Gradr::Zero();
  ///@}
};

/**
 * @brief Output bundle for the differentiable growth distance algorithm.
 *
 * @note The total derivatives require the directional derivative outputs
 * (automatically allocated if total derivatives are requested).
 *
 * @tparam dim Dimension of the convex sets.
 */
template <int dim>
struct OutputBundle {
  /// @brief Growth distance algorithm output (always allocated).
  std::unique_ptr<Output<dim>> output;

  /**
   * @brief Directional derivatives of the growth distance optimal value and
   * solution (optional).
   */
  std::unique_ptr<DirectionalDerivative<dim>> dir_derivative;

  /**
   * @brief Total derivatives of the growth distance optimal value and solution
   * (optional).
   */
  std::unique_ptr<TotalDerivative<dim>> total_derivative;

  /**
   * @param alloc_directional If true, allocate memory for directional
   *                          derivatives.
   * @param alloc_total       If true, allocate memory for total derivatives.
   */
  explicit OutputBundle(bool alloc_directional = false,
                        bool alloc_total = false);
};

template <int dim>
OutputBundle<dim>::OutputBundle(bool alloc_directional, bool alloc_total)
    : output(std::make_unique<Output<dim>>()),
      dir_derivative(nullptr),
      total_derivative(nullptr) {
  if (alloc_total || alloc_directional) {
    dir_derivative = std::make_unique<DirectionalDerivative<dim>>();
  }
  if (alloc_total) {
    total_derivative = std::make_unique<TotalDerivative<dim>>();
  }
}

using Output2 = Output<2>;
using Output3 = Output<3>;

using OutputBundle2 = OutputBundle<2>;
using OutputBundle3 = OutputBundle<3>;

/// @brief Returns the solution status name.
inline std::string SolutionStatusName(SolutionStatus status) {
  if (status == SolutionStatus::CoincidentCenters) {
    return "Coincident centers";
  } else if (status == SolutionStatus::MaxIterReached) {
    return "Maximum iterations reached";
  } else if (status == SolutionStatus::Optimal) {
    return "Optimal solution";
  } else if (status == SolutionStatus::IllConditionedInputs) {
    return "Ill-conditioned input sets";
  } else {
    return "";
  }
}

}  // namespace dgd

#endif  // DGD_OUTPUT_H_
