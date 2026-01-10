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
 * @brief Growth distance solver settings.
 */

#ifndef DGD_SOLVERS_SOLVER_SETTINGS_H_
#define DGD_SOLVERS_SOLVER_SETTINGS_H_

#include <cmath>

#include "dgd/data_types.h"

namespace dgd {

namespace detail {

/// @brief Solver settings.
struct SolverSettings {
  // [Debugging]
  /// @brief Whether to print convergence information at each iteration.
#ifdef DGD_VERBOSE_ITERATION
  static constexpr bool kVerboseIteration = true;
#else
  static constexpr bool kVerboseIteration = false;
#endif  // DGD_VERBOSE_ITERATION
  /// @brief Whether to print growth distance bounds or dual function bounds.
  static constexpr bool kPrintGdBounds = false;

  // [Primal warm start]
  /// @brief Smallest nonzero barycentric coordinate value.
  static inline const Real kEpsMinBc = std::pow(kEps, Real(0.75));

  // [All 3D solvers]
  /// @brief Projected simplex area tolerance for barycentric coordinate
  /// computation.
  static inline const Real kEpsArea3 = kEps;

  // [Trust region Newton solver]
  /// @brief (3D) Skip the trust region Newton solution if the Hessian is
  /// singular.
  static constexpr bool kSkipTrnIfSingularHess3 = true;
  /// @brief Tolerance for the Newton step computation.
  static inline const Real kHessMin2 = kSqrtEps;
  static inline const Real kPinvTol3 = kSqrtEps;
  static inline const Real kPinvResErr3 = kEps;
};

}  // namespace detail

}  // namespace dgd

#endif  // DGD_SOLVERS_SOLVER_SETTINGS_H_
