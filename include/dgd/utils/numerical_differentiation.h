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
#include <limits>

#include "dgd/data_types.h"

namespace dgd {

/// @brief Numerical differentiator using the central difference method.
class NumericalDifferentiator {
 public:
  // Function mapping reals to reals.
  using RealFunctionType = std::function<Real(Real)>;
  // Function mapping vectors to reals.
  using VectorFunctionType =
      std::function<Real(const Eigen::Ref<const VecXr>&)>;
  // Function mapping vectors to vectors.
  using VectorValuedFunctionType =
      std::function<void(const Eigen::Ref<const VecXr>&, Eigen::Ref<VecXr>)>;

  explicit NumericalDifferentiator(Real step_size = kSqrtEps);

  Real ScaledStepSize(Real x) const;

  Real Derivative(RealFunctionType f, Real x) const;

  void Gradient(VectorFunctionType f, const Eigen::Ref<const VecXr>& x,
                Eigen::Ref<VecXr>& grad) const;

  void Jacobian(VectorValuedFunctionType f, const Eigen::Ref<const VecXr>& x,
                Eigen::Ref<MatXr> jac) const;

  void set_step_size(Real step_size);

 private:
  // Step size.
  Real h_;
};

}  // namespace dgd

#endif  // DGD_UTILS_NUMERICAL_DIFFERENTIATION_H_
