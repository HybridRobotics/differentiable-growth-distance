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
 * @file debug.h
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-02-24
 * @brief Debug printing functions.
 */

#ifndef DGD_DEBUG_H_
#define DGD_DEBUG_H_

#ifdef DGD_ENABLE_DEBUG_PRINTING

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

#include "dgd/data_types.h"
#include "dgd/output.h"
#include "dgd/settings.h"

namespace dgd {

// Prints debugging information at any iteration of the algorithm.
template <int dim>
void PrintDebugIteration(int iter, Real lb, Real ub,
                         const Matr<dim, dim>& simplex,
                         const Settings& settings, const Output<dim>& out) {
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

// Sets ostream properties and prints debugging header.
template <int dim>
void PrintDebugHeader(int iter, Real lb, Real ub, const Matr<dim, dim>& simplex,
                      const Settings& settings, const Output<dim>& out) {
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
  PrintDebugIteration(iter, lb, ub, simplex, settings, out);
}

// Unsets ostream properties.
inline void PrintDebugFooter() {
  std::cout.unsetf(std::ios::fixed | std::ios::scientific);
  std::cout << std::right;
}

}  // namespace dgd

#define DGD_PRINT_DEBUG_ITERATION(...) dgd::PrintDebugIteration(__VA_ARGS__)
#define DGD_PRINT_DEBUG_HEADER(...) dgd::PrintDebugHeader(__VA_ARGS__)
#define DGD_PRINT_DEBUG_FOOTER() dgd::PrintDebugFooter()

#else  // DGD_ENABLE_DEBUG_PRINTING

#define DGD_PRINT_DEBUG_ITERATION(...) (void)0
#define DGD_PRINT_DEBUG_HEADER(...) (void)0
#define DGD_PRINT_DEBUG_FOOTER() (void)0

#endif  // DGD_ENABLE_DEBUG_PRINTING

#endif  // DGD_DEBUG_H_
