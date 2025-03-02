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
 * @file io.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-02-24
 * @brief Input/Ouput functions.
 */

#ifndef DGD_IO_H_
#define DGD_IO_H_

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "dgd/data_types.h"
#include "dgd/output.h"
#include "dgd/settings.h"

namespace dgd {

namespace io {

/**
 * @brief Prints diagnostic information at any iteration of the algorithm.
 *
 * @tparam dim          Dimension of the convex sets.
 * @tparam simplex_size Maximum number of points in the simplex tableau.
 * @param[in] lb        Growth distance lower bound.
 * @param[in] ub        Growth distance upper bound.
 * @param[in] simplex   Simplex tableau.
 * @param[in] settings  Solver settings.
 * @param[in] out       Solver output.
 */
template <int dim, int simplex_size>
void PrintSolutionDiagnostics(Real lb, Real ub,
                              const Matf<dim, simplex_size>& simplex,
                              const SolverSettings& settings,
                              const SolverOutput<dim, simplex_size>& out) {
  const int iter{out.iter};
  const Real primal{std::abs(Real(1.0) / ub)};
  const Real dual{std::abs(Real(1.0) / lb)};
  const Real rel_err{std::abs(lb / ub)};
  const Real rel_tol{settings.rel_tol};
  const Real prim_feas_err{
      (simplex.template topRows<dim - 1>() * out.bc).norm()};
  // clang-format off
  std::cout << std::setw(6) << std::setprecision(4) << iter << "| "
            << std::setw(15) << std::setprecision(7) << primal << ">= "
            << std::setw(15) << std::setprecision(7) << dual << "|| 0.0 <= "
            << std::setw(15) << std::setprecision(8) << rel_err << "<= "
            << std::setw(15) << std::setprecision(8) << rel_tol << "|| "
            << std::setw(15) << std::setprecision(7) << prim_feas_err
            << std::endl;
  // clang-format on
}

/**
 * @brief Sets ostream properties and prints diagnostics header.
 *
 * @tparam dim          Dimension of the convex sets.
 * @tparam simplex_size Maximum number of points in the simplex tableau.
 * @param[in] lb        Growth distance lower bound.
 * @param[in] ub        Growth distance upper bound.
 * @param[in] simplex   Simplex tableau.
 * @param[in] settings  Solver settings.
 * @param[in] out       Solver output.
 */
template <int dim, int simplex_size>
void PrintDiagnosticsHeader(Real lb, Real ub,
                            const Matf<dim, simplex_size>& simplex,
                            const SolverSettings& settings,
                            const SolverOutput<dim, simplex_size>& out) {
  std::cout << std::left;
  std::cout << std::scientific;
  const int line_len{6 + 15 * 4 + 13 + 21};
  std::cout << std::string(line_len, '-') << std::endl;
  // clang-format off
  std::cout << std::setw(6) << "iter" << "| "
            << std::setw(15) << "primal" << ">= "
            << std::setw(15) << "dual" << "|| 0.0 <= "
            << std::setw(15) << "rel-err" << "<= "
            << std::setw(15) << "rel-tol" << "|| "
            << std::setw(15) << "prim-feas-err" << std::endl;
  // clang-format on
  std::cout << std::string(line_len, '-') << std::endl;
  PrintSolutionDiagnostics(lb, ub, simplex, settings, out);
}

/**
 * @brief Unsets ostream properties.
 */
inline void PrintDiagnosticsFooter() {
  std::cout.unsetf(std::ios::fixed | std::ios::scientific);
  std::cout << std::right;
}

}  // namespace io

}  // namespace dgd

#endif  // DGD_IO_H_
