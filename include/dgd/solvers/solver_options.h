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
   * @note The full solution is not implemented for the three-dimensional growth
   * distance problem.
   */
  kFull,
};

/// @brief Solver settings.
struct SolverSettings {
  // [Debug]
  // Whether to print convergence information at each iteration.
  static constexpr bool kEnableDebugPrinting = false;
  // Whether to print growth distance bounds or dual function bounds.
  static constexpr bool kPrintGdBounds = false;

  // [All solvers]
  // Smallest nonzero barycentric coordinate value.
  static constexpr Real kEpsMinBc = std::pow(kEps, Real(0.75));

  // [All 3D solvers]
  // Projected simplex area tolerance for barycentric coordinate computation.
  static constexpr Real kEpsArea3 = kEps;
  // Constant added to the normal vector to ensure dual feasibility.
  static constexpr Real kEpsNormal3 = 0;  // kEps;

  // [3D proximal bundle and trust region Newton]
  // Threshold for lower bound increase.
  static constexpr Real kEpsLb3 = std::pow(kEps, Real(0.75));

  // [Proximal bundle]
  // Type of proximal regularization.
  static constexpr auto kProxRegType = ProximalRegularization::kConstant;
  // Constants used to compute the regularization factor (>= 0).
  static constexpr Real kProxKc = Real(1.0e-3);  // Constant.
  static constexpr Real kProxKa = Real(0.1e-3);  // Adaptive.
  static constexpr Real kProxThresh = 0.0;       // <= 1.0.

  // [Trust region Newton]
  // [3D] Trust region Newton solution level.
  static constexpr auto kTrnLevel = TrustRegionNewtonLevel::kPartial;
  // [3D] Skip the trust region Newton solution if the Hessian is singular.
  static constexpr bool kSkipTrnIfSingularHess3 = true;
  // Tolerance for the Newton step computation.
  static constexpr Real kHessMin2 = kSqrtEps;
  static constexpr Real kPinvTol3 = kSqrtEps;
  static constexpr Real kPinvResErr3 = kEps;
};

/// @brief Returns the solver name.
template <SolverType S>
inline constexpr std::string SolverName() {
  if constexpr (S == SolverType::CuttingPlane) {
    return "Cutting plane";
  } else if constexpr (S == SolverType::ProximalBundle) {
    if (SolverSettings::kProxRegType == ProximalRegularization::kConstant) {
      return "Proximal bundle, constant regularization";
    } else {
      return "Proximal bundle, adaptive regularization";
    }
  } else if constexpr (S == SolverType::TrustRegionNewton) {
    if (SolverSettings::kTrnLevel == TrustRegionNewtonLevel::kPartial) {
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
