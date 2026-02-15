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

  /// @brief Trust region Newton solver.
  TrustRegionNewton,
};

/// @brief Derivative order required for each solver type.
template <SolverType S>
inline constexpr int SolverOrder() {
  if constexpr (S == SolverType::CuttingPlane) {
    return 1;
  } else if constexpr (S == SolverType::TrustRegionNewton) {
    return 2;
  } else {
    return -1;
  }
}

/// @brief Warm start type for the growth distance algorithm.
enum class WarmStartType {
  /// @brief Primal solution warm start.
  Primal,

  /// @brief Dual solution warm start.
  Dual
};

/// @brief Barycentric coordinate solver type.
enum class BcSolverType {
  /**
   * @brief Cramer's rule, assuming nondegeneracy.
   *
   * @attention This method can only be used for the cutting plane solver.
   */
  Cramer,

  /// @brief Full pivot LU decomposition with projection, handling degeneracy.
  LU,
};

/// @brief Returns the solver name.
template <SolverType S>
inline constexpr std::string SolverName() {
  if constexpr (S == SolverType::CuttingPlane) {
    return "Cutting plane";
  } else if constexpr (S == SolverType::TrustRegionNewton) {
    return "Trust region Newton";
  } else {
    return "";
  }
}

/// @brief Returns the initialization type name.
inline std::string InitializationName(bool warm_start) {
  return warm_start ? "Warm start" : "Cold start";
}

/// @brief Returns the warm start type name.
inline std::string WarmStartName(WarmStartType ws_type) {
  if (ws_type == WarmStartType::Primal) {
    return "Primal";
  } else if (ws_type == WarmStartType::Dual) {
    return "Dual";
  } else {
    return "";
  }
}

/// @brief Returns the barycentric coordinate solver type name.
template <BcSolverType BST>
inline constexpr std::string BcSolverName() {
  if constexpr (BST == BcSolverType::Cramer) {
    return "Cramer's rule";
  } else if constexpr (BST == BcSolverType::LU) {
    return "LU decomposition";
  } else {
    return "";
  }
}

}  // namespace dgd

#endif  // DGD_SOLVERS_SOLVER_TYPES_H_
