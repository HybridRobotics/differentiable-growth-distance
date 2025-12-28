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
 * @brief Growth distance solver types and settings.
 */

#ifndef DGD_SOLVERS_SOLVER_OPTIONS_H_
#define DGD_SOLVERS_SOLVER_OPTIONS_H_

#include <cmath>
#include <string>

#include "dgd/data_types.h"

namespace dgd {

namespace detail {

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

/// @brief Proximal regularization type.
enum class ProximalRegularization {
  /// @brief Constant regularization.
  kConstant,

  /// @brief Adaptive regularization, proportional to the iteration number.
  kAdaptive,
};

/// @brief Trust region Newton solution level.
enum class TrustRegionNewtonLevel {
  /**
   * @brief Return cutting plane solution if the Newton step does not lie in the
   * trust region.
   */
  kPartial,

  /**
   * @brief Return full trust region Newton solution.
   *
   * @attention The full solution is not implemented for the growth distance
   * problem in 3D.
   */
  kFull,
};

/// @brief Solver settings.
struct SolverSettings {
  // [Debugging]
  /// @brief Whether to print convergence information at each iteration.
  static constexpr bool kEnableDebugPrinting = false;
  /// @brief Whether to print growth distance bounds or dual function bounds.
  static constexpr bool kPrintGdBounds = false;

  // [Primal warm start]
  /// @brief Smallest nonzero barycentric coordinate value.
  static inline const Real kEpsMinBc = std::pow(kEps, Real(0.75));

  // [All 3D solvers]
  /// @brief Projected simplex area tolerance for barycentric coordinate
  /// computation.
  static inline const Real kEpsArea3 = kEps;

  // [Proximal bundle solver]
  /// @brief Type of proximal regularization.
  static constexpr auto kProxRegType = ProximalRegularization::kConstant;
  /// @brief Constants used to compute the regularization factor (>= 0).
  static inline const Real kProxKc = Real(1.0e-3);   // Constant.
  static inline const Real kProxKa = Real(0.1e-3);   // Adaptive.
  static inline const Real kProxThresh = Real(0.0);  // <= 1.0.

  // [Trust region Newton solver]
  /// @brief (3D) Trust region Newton solution level.
  static constexpr auto kTrnLevel = TrustRegionNewtonLevel::kPartial;
  /// @brief (3D) Skip the trust region Newton solution if the Hessian is
  /// singular.
  static constexpr bool kSkipTrnIfSingularHess3 = true;
  /// @brief Tolerance for the Newton step computation.
  static inline const Real kHessMin2 = kSqrtEps;
  static inline const Real kPinvTol3 = kSqrtEps;
  static inline const Real kPinvResErr3 = kEps;
};

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

}  // namespace detail

}  // namespace dgd

#endif  // DGD_SOLVERS_SOLVER_OPTIONS_H_
