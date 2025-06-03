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
 * @file settings.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-02-18
 * @brief Solver settings struct.
 */

#ifndef DGD_SETTINGS_H_
#define DGD_SETTINGS_H_

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief Settings for the growth distance algorithm.
 */
struct Settings {
  /**
   * @brief Minimum distance between the centers of the convex sets (\f$> 0\f$).
   *
   * If the center positions of the two convex sets are \f$p_1\f$ and \f$p_2\f$
   * (corresponding to the rigid body transformations), and \f$|p_1 - p_2| <\f$
   * min_center_dist, then the growth distance is set to zero.
   */
  Real min_center_dist{kSqrtEps};

  /**
   * @brief Relative tolerance between the upper and lower bounds of the growth
   * distance (\f$= 1 + \epsilon\f$).
   *
   * The convergence criterion is
   * \f[
   * lb \leq ub \leq (\text{rel_tol}) \ lb.
   * \f]
   */
  Real rel_tol{Real(1.0) + kSqrtEps};

  /**
   * @brief Maximum number of solver iterations (\f$> 1\f$).
   */
  int max_iter{100};
};

}  // namespace dgd

#endif  // DGD_SETTINGS_H_
