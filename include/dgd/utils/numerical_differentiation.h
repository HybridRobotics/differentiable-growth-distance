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
 * @brief Numerical differentiation class.
 */

#ifndef DGD_UTILS_NUMERICAL_DIFFERENTIATION_H_
#define DGD_UTILS_NUMERICAL_DIFFERENTIATION_H_

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "dgd/data_types.h"

namespace dgd {

/// @brief Numerical differentiator using the central difference method.
class NumericalDifferentiator {
 public:
  /// @brief Function mapping reals to reals.
  using RealFunctionType = std::function<Real(Real)>;
  /// @brief Function mapping vectors to reals.
  using VectorFunctionType =
      std::function<Real(const Eigen::Ref<const VecXr>&)>;
  /// @brief Function mapping vectors to vectors.
  using VectorValuedFunctionType =
      std::function<void(const Eigen::Ref<const VecXr>&, Eigen::Ref<VecXr>)>;

  /**
   * @brief Constructs a numerical differentiator object.
   *
   * @param step_size Step size for finite difference calculations.
   */
  explicit NumericalDifferentiator(Real step_size = kSqrtEps);

  /// @brief Returns the scaled step size for a given x.
  Real ScaledStepSize(Real x) const;

  /**
   * @brief Computes the numerical derivative of a real-valued real function at
   * a point.
   *
   * @param  f Function to differentiate.
   * @param  x Point at which to evaluate the derivative.
   * @return Derivative of f at x.
   */
  Real Derivative(RealFunctionType f, Real x) const;

  /**
   * @brief Computes the numerical gradient of a real-valued vector function at
   * a point.
   *
   * @param[in]  f    Function to differentiate.
   * @param[in]  x    Point at which to evaluate the gradient.
   * @param[out] grad Gradient vector.
   */
  void Gradient(VectorFunctionType f, const Eigen::Ref<const VecXr>& x,
                Eigen::Ref<VecXr> grad) const;

  /**
   * @brief Computes the numerical Jacobian of a vector-valued vector function
   * at a point.
   *
   * @param[in]  f   Function to differentiate.
   * @param[in]  x   Point at which to evaluate the Jacobian.
   * @param[out] jac Jacobian matrix.
   */
  void Jacobian(VectorValuedFunctionType f, const Eigen::Ref<const VecXr>& x,
                Eigen::Ref<MatXXr> jac) const;

  /**
   * @brief Computes the numerical Hessian of a real-valued vector function at
   * a point.
   *
   * @param[in]  f    Function to differentiate.
   * @param[in]  x    Point at which to evaluate the Hessian.
   * @param[out] hess Hessian matrix.
   */
  void Hessian(VectorFunctionType f, const Eigen::Ref<const VecXr>& x,
               Eigen::Ref<MatXXr> hess) const;

  /// @brief Sets the step size.
  void set_step_size(Real step_size);

 private:
  Real h_; /**< Step size. */
};

inline NumericalDifferentiator::NumericalDifferentiator(Real step_size) {
  set_step_size(step_size);
}

inline Real NumericalDifferentiator::ScaledStepSize(Real x) const {
  return std::max(Real(1.0), std::abs(x)) * h_;
}

inline Real NumericalDifferentiator::Derivative(RealFunctionType f,
                                                Real x) const {
  const Real h = ScaledStepSize(x);
  return (f(x + h) - f(x - h)) / (Real(2.0) * h);
}

inline void NumericalDifferentiator::Gradient(VectorFunctionType f,
                                              const Eigen::Ref<const VecXr>& x,
                                              Eigen::Ref<VecXr> grad) const {
  if (grad.size() != x.size()) {
    throw std::domain_error("x and grad have different sizes");
  }

  // Ensure f can be evaluated at x.
  try {
    (void)f(x);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to evaluate f(x): ") +
                             e.what());
  }

  const int n = static_cast<int>(x.size());
  VecXr xp = x;
  for (int i = 0; i < n; ++i) {
    const Real xi = x(i);
    const Real h = ScaledStepSize(xi);

    xp(i) = xi + h;
    const Real fp = f(xp);
    xp(i) = xi - h;
    const Real fm = f(xp);

    grad(i) = (fp - fm) / (Real(2.0) * h);

    xp(i) = xi;
  }
}

inline void NumericalDifferentiator::Jacobian(VectorValuedFunctionType f,
                                              const Eigen::Ref<const VecXr>& x,
                                              Eigen::Ref<MatXXr> jac) const {
  if (jac.cols() != x.size()) {
    throw std::domain_error("Incompatible sizes for x and jac");
  }

  // Ensure f can be evaluated at x.
  try {
    VecXr tmp(jac.rows());
    f(x, tmp);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to evaluate f(x): ") +
                             e.what());
  }

  const int n = static_cast<int>(x.size());
  const int m = static_cast<int>(jac.rows());

  VecXr xn = x;
  VecXr yp(m), ym(m);
  for (int i = 0; i < n; ++i) {
    const Real xi = x(i);
    const Real h = ScaledStepSize(xi);

    xn(i) += h;
    f(xn, yp);
    xn(i) = xi - h;
    f(xn, ym);

    jac.col(i) = (yp - ym) / (Real(2.0) * h);

    xn(i) = xi;
  }
}

inline void NumericalDifferentiator::Hessian(VectorFunctionType f,
                                             const Eigen::Ref<const VecXr>& x,
                                             Eigen::Ref<MatXXr> hess) const {
  const int n = static_cast<int>(x.size());
  if ((hess.rows() != n) || (hess.cols() != n)) {
    throw std::domain_error("Incompatible sizes for x and hess");
  }

  // Ensure f can be evaluated at x.
  try {
    (void)f(x);
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("Failed to evaluate f(x): ") +
                             e.what());
  }

  VecXr xp = x;

  // Precompute step sizes for each coordinate.
  VecXr h(n);
  for (int i = 0; i < n; ++i) h(i) = ScaledStepSize(x(i));

  // Compute diagonal terms.
  const Real fx = f(x);
  for (int i = 0; i < n; ++i) {
    const Real xi = x(i);
    const Real hi = h(i);

    xp(i) = xi + hi;
    const Real fp = f(xp);
    xp(i) = xi - hi;
    const Real fm = f(xp);

    hess(i, i) = (fp - Real(2.0) * fx + fm) / (hi * hi);

    xp(i) = xi;
  }

  // Compute off-diagonal terms.
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      const Real xi = x(i), xj = x(j);
      const Real hi = h(i), hj = h(j);

      xp(i) = xi + hi;
      xp(j) = xj + hj;
      const Real fpp = f(xp);
      xp(j) = xj - hj;
      const Real fpm = f(xp);
      xp(i) = xi - hi;
      const Real fmm = f(xp);
      xp(j) = xj + hj;
      const Real fmp = f(xp);

      hess(i, j) = (fpp - fpm - fmp + fmm) / (Real(4.0) * hi * hj);
      hess(j, i) = hess(i, j);

      xp(i) = xi;
      xp(j) = xj;
    }
  }
}

inline void NumericalDifferentiator::set_step_size(Real step_size) {
  if (step_size < kSqrtEps) {
    throw std::domain_error("Step size too small");
  }
  h_ = step_size;
}

}  // namespace dgd

#endif  // DGD_UTILS_NUMERICAL_DIFFERENTIATION_H_
