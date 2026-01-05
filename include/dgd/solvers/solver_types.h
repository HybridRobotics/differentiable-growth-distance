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
 * @brief Growth distance solver types.
 */

#ifndef DGD_SOLVERS_SOLVER_TYPES_H_
#define DGD_SOLVERS_SOLVER_TYPES_H_

#include <string>

#include "dgd/data_types.h"

namespace dgd {

/// @brief Nonsmooth solver types for the computing the growth distance.
enum class SolverType {
  /// @brief Cutting plane solver.
  CuttingPlane,

  /**
   * @brief Proximal bundle solver with constant or adaptive regularization.
   *
   * @attention The proximal bundle solver is not implemented for the growth
   * distance problem in 3D.
   *
   * @note The proximal bundle method can have slow convergence in general.
   */
  ProximalBundle,

  /// @brief Trust region Newton solver with partial or full solution.
  TrustRegionNewton,
};

/// @brief Derivative order required for each solver type.
template <SolverType S>
inline constexpr int SolverOrder() {
  if constexpr (S == SolverType::TrustRegionNewton) {
    return 2;
  } else {
    return 1;
  }
}

/// @brief Returns the solver name.
template <SolverType S>
inline constexpr std::string SolverName() {
  if constexpr (S == SolverType::CuttingPlane) {
    return "Cutting plane";
  } else if constexpr (S == SolverType::ProximalBundle) {
    if constexpr (SolverSettings::kProxRegType ==
                  ProximalRegularization::kConstant) {
      return "Proximal bundle, constant regularization";
    } else {
      return "Proximal bundle, adaptive regularization";
    }
  } else if constexpr (S == SolverType::TrustRegionNewton) {
    if constexpr (SolverSettings::kTrnLevel ==
                  TrustRegionNewtonLevel::kPartial) {
      return "Trust region Newton, partial";
    } else {
      return "Trust region Newton, full";
    }
  } else {
    return "";
  }
}

/// @brief Barycentric coordinate solver type.
enum class BcSolverType {
  /**
   * @brief Cramer's rule, assuming nondegeneracy.
   *
   * @attention This method can only be used for the cutting plane solver.
   */
  kCramer,

  /// @brief Full pivot LU decomposition with projection, handling degeneracy.
  kLU,
};

}  // namespace dgd

#endif  // DGD_SOLVERS_SOLVER_TYPES_H_
