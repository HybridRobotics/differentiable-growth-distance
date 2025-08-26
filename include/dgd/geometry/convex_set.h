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
 * @brief Convex set abstract class implementation.
 */

#ifndef DGD_GEOMETRY_CONVEX_SET_H_
#define DGD_GEOMETRY_CONVEX_SET_H_

#include <cmath>
#include <iostream>
#include <stdexcept>

#include "dgd/data_types.h"

namespace dgd {

/**
 * @brief Support function hint; used internally.
 *
 * @tparam dim Dimension of the convex sets.
 */
template <int dim>
struct SupportFunctionHint {
  /// @brief Normal vector (in local coordinates) at the previous iteration.
  Vecr<dim> n_prev = Vecr<dim>::Zero();

  /// @brief Integer hint for support function computation.
  int idx_ws = -1;
};

/**
 * @brief Second-order support function derivatives.
 *
 * @tparam dim Dimension of the convex sets.
 */
template <int dim>
struct SupportFunctionDerivatives {
  /**
   * @brief Support point derivative with respect to the normal vector.
   *
   * @attention The support point derivative is equal to the support function
   * Hessian, and it exists almost everywhere. Whenever it exists, it must be
   * positive semi-definite. Note that the normal vector lies in the null space
   * of the support point derivative.
   *
   * @see differentiable
   */
  Matr<dim, dim> Dsp;

  /// @brief Support point.
  Vecr<dim> sp;

  /// @brief Differentiability of the support point at the given normal vector.
  bool differentiable;
};

/**
 * @brief Convex set abstract class implementing the support function.
 *
 * @attention The convex set must be a compact and solid set (i.e., closed,
 * bounded, and with a nonempty interior). Also, the origin must be in
 * the interior of the set. The origin is the center point of the convex set
 * in its local frame.
 *
 * @tparam dim Dimension of the convex set (2 or 3).
 */
template <int dim>
class ConvexSet {
 public:
  virtual ~ConvexSet() {}

  /**
   * @brief Computes the support function in the local frame.
   *
   * Implements the support function for the convex set \f$C\f$, which is given
   * by: \f{align*}{
   * sv & = \max_{x \in C} \langle n, x\rangle, \\
   * sp & \in \arg \max_{x \in C} \langle n, x\rangle,
   * \f}
   * where \f$sv\f$ is the return value of the function.
   *
   * @note If the normal vector n is required to have unit 2-norm, the function
   * RequireUnitNormal should return true.
   *
   * @note Safety margins (in the 2-norm) can be directly included in the
   * support function computation (when n has unit 2-norm) as: \f{align*}{
   * sv & \leftarrow sv + m, \\
   * sp & \leftarrow sp + m \cdot n,
   * \f}
   * where \f$m\f$ is the safety margin. Note that the safety margin also
   * increases the inradius. A nonzero safety margin can result in slower
   * convergence when using the cutting plane method.
   *
   * @see RequireUnitNormal
   *
   * @param[in]     n    Normal vector.
   * @param[out]    sp   Support point. Any point at which the maximum for the
   *                     support function is attained.
   * @param[in,out] hint Additional hints.
   * @return        Support function value at the normal vector.
   */
  virtual Real SupportFunction(
      const Vecr<dim>& n, Vecr<dim>& sp,
      SupportFunctionHint<dim>* hint = nullptr) const = 0;

  /**
   * @brief Computes the support function and its higher-order derivatives.
   *
   * @see SupportFunctionDerivatives
   *
   * @param[in]     n     Normal vector.
   * @param[out]    deriv Support function derivatives.
   * @param[in,out] hint  Additional hints.
   * @return        Support function value at the normal vector.
   */
  virtual Real SupportFunction(const Vecr<dim>& n,
                               SupportFunctionDerivatives<dim>& deriv,
                               SupportFunctionHint<dim>* hint = nullptr) const;

  /**
   * @brief Returns the normalization requirement for the normal vector passed
   * to SupportFunction.
   *
   * If the return value is true, the argument n to SupportFunction will have
   * unit 2-norm.
   *
   * @see SupportFunction
   */
  virtual bool RequireUnitNormal() const;

  /// @brief Returns true if the convex set is polytopic.
  virtual bool IsPolytopic() const;

  virtual void PrintInfo() const;

  static constexpr int dimension();

  Real inradius() const;

  void set_inradius(Real inradius);

 protected:
  ConvexSet();

  ConvexSet(Real inradius);

  /**
   * @brief Support function differentiability constant.
   *
   * The support point function is considered (numerically) differentiable if
   * the support point at a given normal vector n is unique, the support point
   * function is differentiable at n, and
   * sv = n.dot(sp) >= n.dot(p) + eps_diff(),
   * where sv and sp are the support value and support point, and p is, loosely
   * speaking, any point not smoothly connected to sp.
   */
  static constexpr Real eps_diff();

  /**
   * @brief Convex set inradius at the origin.
   *
   * Radius of a ball that is centered at the origin and contained in the set.
   * Any number greater than 0 and less than the inradius will work. However,
   * larger values can help prevent singularities in simplex computations and
   * enable faster convergence.
   */
  Real inradius_;
};

template <int dim>
inline ConvexSet<dim>::ConvexSet() : ConvexSet(kEps) {}

template <int dim>
inline ConvexSet<dim>::ConvexSet(Real inradius) : inradius_(inradius) {
  if (inradius <= 0.0) {
    throw std::domain_error("Inradius is not positive");
  }
}

template <int dim>
inline Real ConvexSet<dim>::SupportFunction(
    const Vecr<dim>& n, SupportFunctionDerivatives<dim>& deriv,
    SupportFunctionHint<dim>* hint) const {
  deriv.differentiable = false;
  return SupportFunction(n, deriv.sp, hint);
}

template <int dim>
inline bool ConvexSet<dim>::RequireUnitNormal() const {
  return true;
}

template <int dim>
inline bool ConvexSet<dim>::IsPolytopic() const {
  return false;
}

template <int dim>
inline void ConvexSet<dim>::PrintInfo() const {
  std::cout << "Type: Abstract Convex Set (dim = " << dim << ")" << std::endl
            << "  Inradius: " << inradius_ << std::endl;
}

template <int dim>
constexpr int ConvexSet<dim>::dimension() {
  return dim;
}

template <int dim>
inline Real ConvexSet<dim>::inradius() const {
  return inradius_;
}

template <int dim>
inline void ConvexSet<dim>::set_inradius(Real inradius) {
  if (inradius <= 0.0) {
    throw std::domain_error("Inradius is not positive");
  }
  inradius_ = inradius;
}

template <int dim>
constexpr Real ConvexSet<dim>::eps_diff() {
  return dim * std::pow(kEps, Real(1.0 / 3.0));
}

}  // namespace dgd

#endif  // DGD_GEOMETRY_CONVEX_SET_H_
