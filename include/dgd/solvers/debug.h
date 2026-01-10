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
 * @brief Debug printing and logging functions.
 */

#ifndef DGD_SOLVERS_DEBUG_H_
#define DGD_SOLVERS_DEBUG_H_

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "dgd/data_types.h"
#include "dgd/output.h"

namespace dgd {

namespace detail {

/// @brief Sets ostream properties and prints debugging header.
inline void PrintDebugHeader(const std::string& solver_name) {
  std::cout << std::left;
  std::cout << std::scientific;
  const int line_len = 6 + 15 * 3 + 17 * 2 + 21;
  std::cout << std::string(line_len, '-') << std::endl;
  std::cout << "Solver: " << solver_name << std::endl;
  // clang-format off
  std::cout << std::setw(6) << "iter" << "| "
            << std::setw(15) << "primal" << ">= "
            << std::setw(15) << "dual" << "|| 0.0 <= "
            << std::setw(17) << "rel-err" << "<= "
            << std::setw(17) << "rel-tol" << "|| "
            << std::setw(15) << "prim-infeas-err" << std::endl;
  // clang-format on
  std::cout << std::string(line_len, '-') << std::endl;
}

/// @brief Prints debugging information at an iteration of the algorithm.
inline void PrintDebugIteration(int iter, Real primal, Real dual, Real rel_tol,
                                Real prim_infeas_err) {
  const Real rel_err = primal / dual;
  // clang-format off
  std::cout << std::setw(6) << std::setprecision(4) << iter << "| "
            << std::setw(15) << std::setprecision(7) << primal << ">= "
            << std::setw(15) << std::setprecision(7) << dual << "|| 0.0 <= "
            << std::setw(17) << std::setprecision(10) << rel_err << "<= "
            << std::setw(17) << std::setprecision(10) << rel_tol << "|| "
            << std::setw(15) << std::setprecision(7) << prim_infeas_err
            << std::endl;
  // clang-format on
}

/// @brief Unsets ostream properties.
inline void PrintDebugFooter() {
  std::cout.unsetf(std::ios::fixed | std::ios::scientific);
  std::cout << std::right;
}

/**
 * Logging.
 */

#ifdef DGD_EXTRACT_METRICS
template <int dim>
inline void InitializeLogs(int max_iter, Output<dim>& out) {
  out.lbs.assign(max_iter, Real(0.0));
  out.ubs.assign(max_iter, kInf);
}

template <int dim>
inline void LogBounds(int iter, Real lb, Real ub, Output<dim>& out) {
  out.lbs[iter] = lb;
  out.ubs[iter] = ub;
}
#endif  // DGD_EXTRACT_METRICS

}  // namespace detail

}  // namespace dgd

#endif  // DGD_SOLVERS_DEBUG_H_
