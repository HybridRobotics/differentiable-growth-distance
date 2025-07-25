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
 * @file numerical_differentiation.cc
 * @author Akshay Thirugnanam (akshay_t@berkeley.edu)
 * @date 2025-07-24
 * @brief Numerical differentiation implementation.
 */

#include "dgd/utils/numerical_differentiation.h"

#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"

namespace dgd {

NumericalDifferentiator::NumericalDifferentiator(Real step_size) {
  set_step_size(step_size);
}

inline Real NumericalDifferentiator::ScaledStepSize(Real x) const {
  return std::max(Real(1.0), std::abs(x)) * h_;
}

Real NumericalDifferentiator::Derivative(RealFunctionType f, Real x) const {
  const Real h = ScaledStepSize(x);
  try {
    return (f(x + h) - f(x - h)) / (Real(2.0) * h);
  } catch (const std::exception& e) {
    std::cerr << "Error evaluating f(" << x << "): " << e.what() << std::endl;
    throw std::runtime_error("Failed to evaluate function");
  }
}

void NumericalDifferentiator::Gradient(VectorFunctionType f,
                                       const Eigen::Ref<const VecXr>& x,
                                       Eigen::Ref<VecXr>& grad) const {
  if (grad.size() != x.size()) {
    throw std::domain_error("x and grad have different sizes");
  }

  VecXr xp = x, xm = x;
  try {
    for (int i = 0; i < x.size(); ++i) {
      const Real h = ScaledStepSize(x(i));
      xp(i) += h;
      xm(i) -= h;
      grad(i) = (f(xp) - f(xm)) / (Real(2.0) * h);
    }
  } catch (const std::exception& e) {
    std::cerr << "Error evaluating gradient: " << e.what() << std::endl;
    throw std::runtime_error("Failed to evaluate function");
  }
}

void NumericalDifferentiator::Jacobian(VectorValuedFunctionType f,
                                       const Eigen::Ref<const VecXr>& x,
                                       Eigen::Ref<MatXr> jac) const {
  if (jac.cols() != x.size()) {
    throw std::domain_error("Incompatible sizes for x and jac");
  }

  VecXr xn = x;
  VecXr yp(jac.rows()), ym(jac.rows());
  try {
    for (int i = 0; i < x.size(); ++i) {
      const Real h = ScaledStepSize(x(i));
      xn(i) += h;
      f(xn, yp);
      xn(i) -= Real(2.0) * h;
      f(xn, ym);
      xn(i) = x(i);

      jac.col(i) = (yp - ym) / (Real(2.0) * h);
    }
  } catch (const std::exception& e) {
    std::cerr << "Error evaluating Jacobian: " << e.what() << std::endl;
    throw std::runtime_error("Failed to evaluate function");
  }
}

inline void NumericalDifferentiator::set_step_size(Real step_size) {
  if (step_size < kSqrtEps) {
    throw std::domain_error("Step size too small");
  }
  h_ = step_size;
}

}  // namespace dgd
